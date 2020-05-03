import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf


class TargetTransformer():
    def __init__(self):
        pass
    
    def transform(self, df):
        
        # log transform to get a contant variance
        #df['sold_qty'] = df['sold_qty'].apply(np.log1p)
        
        # differentiate to get a constant mean
        df_grouped = df.groupby('id', as_index=False)
        
        self.initial_values = df_grouped['sold_qty'].first()
        self.initial_values = self.initial_values.rename({'sold_qty': 'sold_qty_initial'}, axis=1)
        df['sold_qty'] = df_grouped['sold_qty'].transform(lambda x: x.diff())
        df = df_grouped.apply(lambda x:x.iloc[1:])
        
        return df
    
    def inverse(self, df):
        # inverse differentiation
        df['sold_qty'] = df.groupby('id')['sold_qty'].cumsum()
        df = pd.merge(df, self.initial_values, on='id', how='left')
        df['sold_qty'] = df[['sold_qty', 'sold_qty_initial']].sum(axis=1)
        df = df.drop('sold_qty_initial', axis=1)
        
        # inverse log transform
        #df['sold_qty'] = df['sold_qty'].apply(np.expm1)
        
        return df


def reduce_mem_usage(df, verbose=True):
    '''Optimize dataframe number types memory usage
    '''
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'] 
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics: 
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    return df


def categorical_to_ordinal(df):
    '''Convert pandas categorical types to ordinal integers
    '''
    # In the sample dataset, event_type_2 is always empty. It's thus useless for the models
    df = df.drop(['event_name_2', 'event_type_2'], axis=1)

    # We fill event_type_1 nans as it represents an instance of an event category: there's just no event at that time
    df.event_type_1 = df.event_type_1.cat.add_categories('none').fillna('none')

    # event_name_1 is not very informative as we take only one year of data, and most events are yearly based. Thus they appear only once in the training set.
    df = df.drop('event_name_1', axis=1)

    # convert categorical features to ordinal
    for col in df.select_dtypes('category'):
            df[col] = df[col].cat.codes

    # compress snap events representation: for each row, there can be only one true
    df['snap'] = df[['snap_CA', 'snap_TX', 'snap_WI']].sum(axis=1)
    df = df.drop(['snap_CA', 'snap_TX', 'snap_WI'], axis=1)
    
    return df


def get_best_lags(x, n_most=1, after=0):
    '''return lags corresponding to the maximum autocorrelation values
    '''
    x = np.diff(x)
    x = x[~np.isnan(x)]
    acorr = acf(x, fft=True)
    lags = np.argsort(np.abs(acorr)) + 1
    lags = np.extract(lags > after, lags)
    
    if len(lags) < n_most:
        lags = np.pad(lags, (0, n_most - len(lags)))
    else:
        return lags[:n_most]


def item_lag(df, cols, n_most, after, df_grouped=None):
    '''return value on lags corresponding to the maximum autocorrelation values
    '''
    def get_lagged_serie(x, n_most, after, col_name):
        res = pd.DataFrame(index=x.index)
        lags = get_best_lags(x, n_most, after)
        for i, lag in enumerate(lags):
            res[f'{col_name}_lag_{i}'] = x.shift(lag).T
        return res
    
    if df_grouped is None:
        df_grouped = df.groupby('id', as_index=False)
    
    for col in cols:
        df = pd.concat([
            df, df_grouped[col].apply(get_lagged_serie, n_most, after, col).astype(np.float32)
        ], axis=1)
    
    return df


def item_rolling_mean(df, agg_cols, window, shift=0, df_grouped=None):
    '''return rolling mean of a column to aggregate by a categorical column
    '''
    
    if df_grouped is None:
        df_grouped = df.groupby('id', as_index=False)
    
    for agg_col in agg_cols:
        col_name = f'item_{agg_col}_{window}d_mean'
        df[col_name] = df_grouped[agg_col].shift(shift).rolling(window).mean().astype(np.float32)
        
    return df


def item_rolling_std(df, agg_cols, window, shift=0, df_grouped=None):
    '''return rolling mean of a column to aggregate by a categorical column
    '''
    
    if df_grouped is None:
        df_grouped = df.groupby('id', as_index=False)
    
    for agg_col in agg_cols:
        col_name = f'item_{agg_col}_{window}d_mean'
        df[col_name] = df_grouped[agg_col].shift(shift).rolling(window).std().fillna(0.).astype(np.float32)
        
    return df


def hierarchy_rolling_mean(df, hierarchy, agg_cols, window, shift=0, df_grouped=None):
    '''return rolling mean of a column to aggregate by a categorical column
    '''
    if df_grouped is None:
        df_grouped = df.groupby([hierarchy, 'date'])
        
    for agg_col in agg_cols:
        col_name = f'{hierarchy}_{agg_col}_{window}d_mean'
        df[col_name] = df_grouped[agg_col].transform('sum').shift(shift).rolling(window).mean().astype(np.float32)
        
    return df


def hierarchy_rolling_std(df, hierarchy, agg_cols, window, shift=0, df_grouped=None):
    '''return rolling std of a column to aggregate by a categorical column
    '''
    if df_grouped is None:
        df_grouped = df.groupby([hierarchy, 'date'])
        
    for agg_col in agg_cols:
        col_name = f'{hierarchy}_{agg_col}_{window}d_mean'
        df[col_name] = df_grouped[agg_col].transform('sum').shift(shift).rolling(window).std().fillna(0.).astype(np.float32)
        
    return df



def hierarchy_stats(df, hierarchy, agg_cols):
    '''return aggregated statistics by a categorical column on an column to aggregate. Only previous values are used to avoid leakage
    '''
    df_grouped = df.groupby(hierarchy, as_index=False)
    for agg_col in agg_cols:
        df[f'{hierarchy}_{agg_col}_mean'] = df_grouped[agg_col].transform(lambda x: x.expanding().mean()).astype(np.float32)
        df[f'{hierarchy}_{agg_col}_std'] = df_grouped[agg_col].transform(lambda x: x.expanding().std()).fillna(0.).astype(np.float32)
        df[f'{hierarchy}_{agg_col}_max'] = df_grouped[agg_col].transform(lambda x: x.expanding().max()).astype(np.float32)
    
    return df


def exp_decay(n, decay=.01, w_min=None):
    '''exponential decay to weight samples. The idea is to penalize more the errors on recent samples
    0 =< decay, w_min <= 1
    '''
    if w_min is not None:
        decay = - np.log(w_min) / n
    
    w = np.array([np.exp(-decay * i) for i in range(n)])
    w = w[::-1]
    return w


'''TODO
- calendar:
    - days since last event
    - days until next event
    
- prices:
    - days since last price change
    - price before last change
    - last price pct change
    
    
'''
