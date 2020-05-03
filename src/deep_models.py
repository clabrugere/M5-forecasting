from math import floor
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ARCNN(torch.nn.Module):
    '''Neural network achitecture combining a linear AR model, two-scales convolutionnal layer, categorical embedding for covariates.
    Convolutions and covariates embedding are fed to a serie of dense layers with output equal to the input sequence lenght.
    Outputs are stacked and summed to get a vector of shape (output_size,)
    '''
    def __init__(self, ar_input_size, ar_size_hidden, ar_n_hidden,
                 cnn_n_kernels_global, cnn_n_kernels_local, cnn_kernel_size_local, cnn_dropout,
                 cov_embedding_sizes, cov_embedding_dropout, 
                 linear_sizes, linear_dropout, output_size):
        
        super(ARCNN, self).__init__()
        
        self.ar_net = AR(ar_input_size, output_size, ar_size_hidden, ar_n_hidden)
        
        self.cnn = MultiscaleCNN(ar_input_size, output_size, cnn_n_kernels_global, cnn_n_kernels_local, cnn_kernel_size_local, cnn_dropout)
        self.cnn_output_size = cnn_n_kernels_global + cnn_n_kernels_local
        
        self.cov_emb = CovariatesEmbedding(cov_embedding_sizes, cov_embedding_dropout)
        self.emb_output_size = np.sum([emb_size for _, emb_size in cov_embedding_sizes])
        
        # linear layers
        self.linears = nn.ModuleList([nn.Linear(self.emb_output_size + self.cnn_output_size, linear_sizes[0])])
        self.linears.extend([nn.Linear(linear_sizes[i], linear_sizes[i+1]) for i in range(len(linear_sizes) - 1)])
        self.activation = nn.ReLU()
        
        # regularization
        self.batchnorms = nn.ModuleList([nn.BatchNorm1d(size) for size in linear_sizes])
        self.dropouts = nn.ModuleList([nn.Dropout(linear_dropout) for _ in range(len(linear_sizes))])
        
        # Output Layer
        self.output = nn.Linear(linear_sizes[-1], output_size)
    
    def forward(self, x_seq, x_cov):
        """
        x_seq: (batch_size, 1, ar_input_size)
        x_cov: (batch_size, n_covariate)
        out = (batch_size, output_size)
        """
        
        out_ar = self.ar_net(x_seq)
        out_cnn = self.cnn(x_seq)
        out_emb = self.cov_emb(x_cov)
        
        out = torch.cat([out_emb, out_cnn], 1)
        
        for linear, batchnorm, dropout in zip(self.linears, self.batchnorms, self.dropouts):
            out = linear(out)
            out = batchnorm(out)
            out = self.activation(out)
            out = dropout(out)
        
        out_dense = self.output(out)
        
        # sum outputs
        out_ar = torch.unsqueeze(out_ar, -1)
        out_dense = torch.unsqueeze(out_dense, -1)
        out = torch.sum(torch.cat([out_ar, out_dense], -1), -1)
        
        return out
    

class AR(torch.nn.Module):
    '''Linear autoregressive model
    input: vector of size (input_size,)
    output: vector of size (output_size,)
    '''
    def __init__(self, input_size, output_size, size_hidden, n_hidden):
        super(AR, self).__init__()
        
        # input layer
        self.linears = nn.ModuleList([nn.Linear(input_size, size_hidden)])
        
        # hidden layers
        self.linears.extend([nn.Linear(size_hidden, size_hidden) for _ in range(n_hidden)])
        
        # output layer
        self.linears.append(nn.Linear(size_hidden, output_size))
        
    def forward(self, x):
        '''
        x: (batch_size, L)
        out: (batch_size, H)
        '''
        for l in self.linears:
            x = l(x)
        
        return x
    
    
class MultiscaleCNN(torch.nn.Module):
    '''Convolutionnal neural net with a global convolution (kernel size = to the sequence length) and local (kernel size < sequence length). Regularisation is done
    with InstanceNorm and dropout layers.
    input: vector of size (input_size,)
    output: vector of size (n_kernels_global + n_kernels_local,)
    '''
    def __init__(self, input_size, output_size, n_kernels_global, n_kernels_local, kernel_size_local, dropout):
        super(MultiscaleCNN, self).__init__()
        
        self.input_size = input_size
        
        # Global convolution
        self.global_conv = nn.Conv1d(in_channels=1, out_channels=n_kernels_global, kernel_size=input_size) # output of size (batch_size, n_kernels_global, 1)
        self.global_normalization = nn.InstanceNorm1d(n_kernels_global)
        self.global_dropout = nn.Dropout(dropout)
        
        # local convolution
        self.local_conv = nn.Conv1d(in_channels=1, out_channels=n_kernels_local, kernel_size=kernel_size_local) # output of size (batch_size, n_kernels_local, (input_size - kernel_size_local) + 1 )
        self.pooling = nn.AdaptiveMaxPool1d(1) #out of size (batch_size, n_kernels_local, 1)
        self.local_normalization = nn.InstanceNorm1d(n_kernels_local)
        self.local_dropout = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()
    
    def forward(self, x):
        '''
        x: (batch_size, L)
        out: (batch_size, n_kernels_global + n_kernels_local)
        '''
        # reshape input for conv layers Input: (batch_size, C_in, L_in)
        x = x.view(-1, 1, self.input_size)
        
        # pass on global convolution
        out_global = self.global_conv(x)
        out_global = self.global_normalization(out_global)
        out_global = self.activation(out_global)
        out_global = self.global_dropout(out_global)
        
        # pass on local convolution
        out_local = self.local_conv(x)
        out_local = self.local_normalization(out_local)
        out_local = self.activation(out_local)
        out_local = self.pooling(out_local)
        out_local = self.local_dropout(out_local)
        
        # reshape and concatenate outputs 
        out = torch.cat([out_global, out_local], 1)
        out = torch.squeeze(out, 2)
        
        return out
    
    
class CovariatesEmbedding(torch.nn.Module):
    '''
    Embedding layer with dropout to encode covariates categories.
    input: vector of size (n_covariates,)
    output: vector of size (sum(emb_size))
    '''
    def __init__(self, embedding_sizes, embedding_dropout):
        super(CovariatesEmbedding, self).__init__()
        
        # Embedding layers
        self.embeddings = nn.ModuleList([nn.Embedding(n_cat, emb_size) for n_cat, emb_size in embedding_sizes])
        self.n_emb = np.sum([emb_size for _, emb_size in embedding_sizes])
        
        # Dropout
        self.embedding_dropout = nn.Dropout(embedding_dropout)
    
    def forward(self, x):
        '''
        x: (batch_size, n_covariates)
        out: (batch_size, sum(emb_size))
        '''
        x = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        out = torch.cat(x, 1)
        out = self.embedding_dropout(out)
        
        return out
    

def conv_output_shape(L_in, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of 1D convolutions
    """
    L_out = floor((L_in + (2 * pad) - (dilation * (kernel_size - 1)) - 1)// stride + 1)
    
    return L_out