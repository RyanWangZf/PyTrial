'''
Implement XGBoost (a gradient-boosted decision tree method) model for tabular individual outcome
prediction in clinical trials.
'''
import pdb
import os
import joblib
import pickle

from xgboost import XGBClassifier as xgb_clf_model
from xgboost import XGBRegressor as xgb_reg_model
from sklearn.multioutput import MultiOutputClassifier # wrap xgboost classifier to be multilable predictor
import numpy as np
import pandas as pd

from pytrial.data.patient_data import TabularPatientBase
from pytrial.utils.check import check_checkpoint_file, check_model_dir, check_model_config_file, make_dir_if_not_exist
from .base import TabularIndivBase

class BuildModel:
    def __new__(self, config):
        if config['mode'] in ['binary']:
            config['objective'] = 'binary:logistic'
            model = xgb_clf_model(**config)
        
        elif config['mode'] in ['multiclass']:
            config['objective'] = 'multi:softmax'
            model = xgb_clf_model(**config)

        elif config['mode'] in ['multilabel']:
            config['objective'] = 'binary:logistic'
            xgb_est = xgb_clf_model(**config)
            model = MultiOutputClassifier(xgb_est)

        else:
            config['objective'] = 'reg:squarederror'
            model = xgb_reg_model(**config)

        return model

class XGBoost(TabularIndivBase):
    '''
    Implement XGBoost model for tabular individual outcome
    prediction in clinical trials.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds.

    max_depth : int
        Maximum tree depth for base learners.

    mode: str
        The task's objectives, in `binary`, `multiclass`, `multilabel`, or `regression`.
        Do not support early stopping when `multilabel`.

    n_jobs: int
        Number of parallel threads used to run xgboost. When used with other Scikit-Learn algorithms like grid search, 
        you may choose which algorithm to parallelize and balance the threads. Creating thread contention will significantly slow down both algorithms.

    reg_alpha: float
        L1 regularization term on weights (xgb's alpha).
    
    reg_lambda: float
        L2 regularization term on weights (xgb's lambda).

    experiment_id: str, optional (default='test')
        The name of current experiment. Decide the saved model checkpoint name.
    '''
    def __init__(self,
        mode,
        n_estimators=100,
        max_depth=8,
        n_jobs=0,
        reg_alpha=0,
        reg_lambda=0,
        num_class=None,
        experiment_id='test',
        ) -> None:
        super().__init__(experiment_id=experiment_id)
        if mode in ['multclass']:
            assert num_class is not None, 'Should specify `num_class` if `mode` is `multiclass`!'

        self.config = {
            'mode':mode,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'reg_alpha': reg_alpha,
            'experiment_id': experiment_id,
            'reg_lambda': reg_lambda,
            'model_name': 'xgboost',
            'n_jobs': n_jobs,
            'num_class':num_class,
        }
        self._save_config(self.config)

    def fit(self, train_data, valid_data=None):
        '''Train logistic regression model to predict patient outcome
        with tabular input data.

        Parameters
        ----------
        train_data: dict
            {
            'x': TabularPatientBase or pd.DataFrame,
            'y': pd.Series or np.ndarray
            }

            - 'x' contain all patient features; 
            - 'y' contain labels for each row.

        valid_data: same as train_data.
            Validation set for early stopping.
        '''
        self._input_data_check(train_data)
        x_feat, y = self._parse_input_data(train_data)
        eval_set = None

        if valid_data is not None and self.config['mode'] != 'multilabel':
            self._input_data_check(train_data)
            x_val, y_val = self._parse_input_data(valid_data)
            eval_set = [(x_val, y_val)]

        self._build_model()
        self._fit_model(x_feat, y, eval_set=eval_set)

    def predict(self, test_data):
        '''
        Make prediction probability based on the learned model.
        Save to `self.result_dir`.

        Parameters
        ----------
        test_data: dict
            {
            'x': TabularPatientBase or pd.DataFrame,
            'y': pd.Series or np.ndarray
            }
            
            - 'x' contain all patient features; 
            - 'y' contain labels for each row. Ignored for prediction function.

        Returns
        -------
        ypred: np.ndarray
            For binary classification, return shape (n, );
            For multiclass classification, return shape (n, n_class).

        '''
        self._input_data_check(test_data)
        x_feat, y = self._parse_input_data(test_data)
        ypred = self.model.predict_proba(x_feat)

        if isinstance(ypred, list): # multilabel prediction
            ypred = np.stack(ypred,0)
            ypred = ypred[...,1].T # n_sample, n_labels

        if ypred.shape[1] == 2: # binary
            ypred = ypred[:,1]

        # save results to dir
        pickle.dump(ypred, open(os.path.join(self.result_dir, 'pred.pkl'), 'wb'))
        if isinstance(test_data, dict):
            if 'y' in test_data:
                pickle.dump(test_data['y'], open(os.path.join(self.result_dir, 'label.pkl'), 'wb'))

        return ypred

    def save_model(self, output_dir=None):
        '''
        Save the learned logistic regression model to the disk.

        Parameters
        ----------
        output_dir: str or None
            The dir to save the learned model.
            If set None, will save model to `self.checkout_dir`.
        '''
        if output_dir is not None:
            make_dir_if_not_exist(output_dir)
        else:
            output_dir = self.checkout_dir

        self._save_config(self.config, output_dir=output_dir)
        ckpt_path = os.path.join(output_dir, 'indiv-tabular.model')
        joblib.dump(self.model, ckpt_path)

    def load_model(self, checkpoint=None):
        '''
        Save the learned logistic regression model to the disk.

        Parameters
        ----------
        checkpoint: str or None
            - If a directory, the only checkpoint file `.model` will be loaded.
            - If a filepath, will load from this file;
            - If None, will load from `self.checkout_dir`.
        '''
        if checkpoint is None:
            checkpoint = self.checkout_dir

        checkpoint_filename = check_checkpoint_file(checkpoint, suffix='model')
        config_filename = check_model_config_file(checkpoint)
        self.model = joblib.load(checkpoint_filename)
        self.config = self._load_config(config_filename)

    def _build_model(self):
        self.model = BuildModel(self.config)

    def _fit_model(self, x_feat, y, eval_set):
        self.model.fit(x_feat, y, eval_set=eval_set)

    def _parse_input_data(self, inputs):
        if isinstance(inputs, TabularPatientBase):
            x_feat = inputs.df
            y = None

        if isinstance(inputs, pd.DataFrame):
            x_feat = inputs
            y = None

        if isinstance(inputs, dict):
            if isinstance(inputs['x'], TabularPatientBase):
                dataset = inputs['x']
                x_feat = dataset.df
            else:
                x_feat = inputs['x']
            y = inputs['y'] if 'y' in inputs else None

        return x_feat, y