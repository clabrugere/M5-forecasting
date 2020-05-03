import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


optuna.logging.disable_default_handler()


class LGBModel():
    '''Wrapper class to optimize lgbm hyperparameters using optuna, train the model with Kfold cross-validation and inference on Kfolds
    '''
    def __init__(self):
        pass
    
    
    def _cross_validate(self, params, X_train, y_train, sample_weights=None, callbacks=None, n_splits=3, X_test=None, predict=False):
        '''Perform cross-validation. Perform inference if 'predict' is True and average out-of-fold predictions.
        '''
        fold_generator = KFold(n_splits=n_splits)
        loss = np.zeros((n_splits))
        
        if predict:
            predictions = np.empty((X_test.shape[0], n_splits))

        for i, (train_inds, val_inds) in enumerate(fold_generator.split(X_train, y_train)):

            # split
            fold_weights = None
            if sample_weights is not None:
                fold_weights = sample_weights.iloc[train_inds]
            
            dtrain_fold = lgb.Dataset(X_train.iloc[train_inds], label=y_train.iloc[train_inds], weight=fold_weights)
            dval_fold = lgb.Dataset(X_train.iloc[val_inds], label=y_train.iloc[val_inds])
            
            # train
            model = lgb.train(
                params, 
                dtrain_fold, 
                valid_sets=[dval_fold], 
                num_boost_round=1000,
                early_stopping_rounds=10,
                verbose_eval=False,
                callbacks=callbacks
            )
            fold_loss = model.best_score['valid_0'][self.metric]
            loss[i] = fold_loss
            
            if predict:
                predictions[:, i] = model.predict(X_test, num_iteration=model.best_iteration)
                
            del dtrain_fold
            del dval_fold
            gc.collect()
        
        if predict:
            return np.clip(np.mean(predictions, axis=1), 0, None)
        else:
            return np.mean(loss)
    
    
    def _objective(self, trial):
        '''Optuna trial step. Unpromising parameter vectors are pruned using a dedicated callback
        '''
        params = {
            "objective": trial.suggest_categorical("objective", ['rmse', 'mape', 'poisson', 'tweedie']),
            "metric": trial.suggest_categorical("metric", [self.metric]),
            "verbosity": -1,
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 0.1),
            "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),
            "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction": trial.suggest_uniform("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 1000),
        }
        
        callbacks = [optuna.integration.LightGBMPruningCallback(trial, self.metric)]
        loss = self._cross_validate(params, self.X_train, self.y_train, self.sample_weights, callbacks)
        self.running_loss.append(loss)
        
        if self._trial % 5 == 0:
            print(f'   trial {self._trial} - best running loss: {np.min(self.running_loss):.3f}')
        
        self._trial += 1
        return loss

    
    def optimize(self, X, y, sample_weights=None, n_trials=10, metric='rmse'):
        '''Perform hypterparameters tuning
        '''
        self.X_train = X
        self.y_train = y
        self.sample_weights = sample_weights
        self.metric = metric
        self._trial = 1
        self.running_loss = []
        
        study = optuna.create_study(direction='minimize')
        
        print(f'   metric: {self.metric}')
        
        study.optimize(self._objective, n_jobs=-1, n_trials=n_trials)
        self.best_value = study.best_value
        self.best_params = study.best_params
        
        print(f'   best loss: {self.best_value:.3f}')
        
    def predict(self, X):
        '''Perform Inference
        '''
        return self._cross_validate(
            self.best_params,
            self.X_train,
            self.y_train,
            X_test=X,
            predict=True
        )