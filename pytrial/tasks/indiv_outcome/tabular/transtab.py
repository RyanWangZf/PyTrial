import pdb
from copy import deepcopy

import transtab
import numpy as np
import pandas as pd
import torch

from pytrial.data.patient_data import TabularPatientBase
from pytrial.utils.check import (
    check_checkpoint_file, check_model_dir, check_model_config_file, make_dir_if_not_exist
)
from .base import TabularIndivBase, IndivTabDataset

class BuildModel:
    def __new__(self, config):
        contrastive_pretrain = config.pop('contrastive_pretrain')
        if not contrastive_pretrain:
            clf = transtab.build_classifier(**config)
            collate_fn = None
        else:
            clf, collate_fn = transtab.build_contrastive_learner(**config)
        return clf, collate_fn

class TransTab(TabularIndivBase):
    '''
    Implement transtab model for tabular individual outcome
    prediction in clinical trials [1]_.

    Parameters
    ----------
    mode: str
        The task's objectives, in `binary`, `multiclass`. # TODO: `multilabel`, or `regression`
        Can be ignored if `contrastive_pretrain` is set True.
    
    categorical_columns: list 
        a list of categorical feature names.

    numerical_columns: list
        a list of numerical feature names.

    binary_columns: list
        a list of binary feature names, accept binary indicators like (yes,no); (true,false); (0,1).
    
    contrastive_pretrain: bool(default=False)
        whether or not take a contrastive pretraining. If set true,
        `num_class` will be ignored.

    num_class: int
        number of output classes to be predicted.

    hidden_dim: int
        the dimension of hidden embeddings.
    
    num_layer: int
        the number of transformer layers used in the encoder.
    
    num_attention_head: int
        the numebr of heads of multihead self-attention layer in the transformers.

    hidden_dropout_prob: float
        the dropout ratio in the transformer encoder.

    ffn_dim: int
        the dimension of feed-forward layer in the transformer layer.
    
    activation: str
        the name of used activation functions, support ``"relu"``, ``"gelu"``, ``"selu"``, ``"leakyrelu"``.

    learning_rate: float
        Learning rate for optimization based on SGD. Use torch.optim.Adam by default.

    weight_decay: float
        Regularization strength for l2 norm; must be a positive float.
        Smaller values specify weaker regularization.
    
    batch_size: int
        Batch size when doing SGD optimization.

    epochs: int
        Maximum number of iterations taken for the solvers to converge.

    num_worker: int
        Number of workers used to do dataloading during training.

    device: str
        Target device to train the model, as `cuda:0` or `cpu`.

    experiment_id: str, optional (default='test')
        The name of current experiment. Decide the saved model checkpoint name.
    
    Notes
    -----
    .. [1] Wang, Z., & Sun, J. (2022). TransTab: Learning Transferable Tabular Transformers Across Tables. NeurIPS'22.
    '''
    def __init__(self, 
        mode=None,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        contrastive_pretrain=False,
        num_class=2,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=8,
        hidden_dropout_prob=0,
        ffn_dim=256,
        activation='relu',
        learning_rate=1e-4,
        weight_decay=1e-4,
        batch_size=64,
        epochs=10,
        num_worker=0,
        device='cuda:0',
        experiment_id='test'):
        super().__init__(experiment_id)

        mode = mode.lower()
        if not contrastive_pretrain:
            assert mode in ['binary', 'multiclass', 'regression', 'multilabel'], 'Must specify `mode` for supervised classifcation.'

        self.config = {
            'categorical_columns':categorical_columns,
            'numerical_columns':numerical_columns,
            'binary_columns':binary_columns,
            'contrastive_pretrain':contrastive_pretrain,
            'num_class':num_class,
            'hidden_dim':hidden_dim,
            'num_layer':num_layer,
            'num_attention_head':num_attention_head,
            'hidden_dropout_prob':hidden_dropout_prob,
            'ffn_dim':ffn_dim,
            'activation':activation,
            'device':device,
            'mode':mode,
            'learning_rate':learning_rate,
            'weight_decay':weight_decay,
            'batch_size':batch_size,
            'epochs':epochs,
            'num_worker':num_worker,
        }
        self._save_config(self.config)
        self._build_model()
        self.device = device

    def fit(self, train_data, valid_data=None):
        '''Train TransTab model to predict patient outcome
        with tabular input data.

        Parameters
        ----------
        train_data: list[dict]
            a list of patient data, each patient is a dict of 
            {
            
            'x': TabularPatientBase or pd.DataFrame, 
            
            'y': pd.Series or np.ndarray
            
            }.

            TransTab can learn from multiple different tabular datasets.

        valid_data: dict
            Validation data during the training for early stopping. 
            valid_data = 
            
            {
            
            'x': TabularPatientBase or pd.DataFrame,

            'y': pd.Series or np.ndarray
            
            }
            
        '''
        self._input_data_check(train_data)
        if valid_data is not None: self._input_data_check(valid_data)
        self._fit_model(train_data=train_data, valid_data=valid_data)

    def predict(self, test_data):
        '''
        Make prediction probability based on the learned model.

        Parameters
        ----------
        test_data: TabularPatientBase or pd.DataFrame
            Contain all patient features.

        Returns
        -------
        ypred: np.ndarray or torch.Tensor

            - For binary classification, return shape (n, );
            - For multiclass classification, return shape (n, n_class).

        '''
        self._input_data_check(test_data)
        data = self._parse_input_data(test_data)
        ypred = transtab.predict(self.model, x_test=data)
        return ypred
        
    def save_model(self, output_dir=None):
        '''
        Save the learned transtab model to the disk.

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
        self.model.save(output_dir)

    def load_model(self, checkpoint):
        '''
        Load the learned transtab model from the given checkpoint dir.

        Parameters
        ----------
        checkpoint: str
            The input dir that stores the pretrained model.

            - If a directory, the only checkpoint file `*.pth.tar` will be loaded.
            - If a filepath, will load from this file.
        '''
        config_filename = check_model_config_file(checkpoint)
        if config_filename is not None:
            config = self._load_config(config_filename)
            self.config.update(config)        
        self.model.load(checkpoint)

    def update(self, config):
        '''Update the configuration of feature extractor's column map for *cat*, *num*, and *bin* cols.
        Or update the number of classes for the output classifier layer.

        Parameters
        ----------
        config: dict
            a dict of configurations: keys `cat:list`, `num:list`, `bin:list` are to specify the new column names;
            key `num_class:int` is to specify the number of classes for finetuning on a new dataset.

        '''
        self.model.update(config)

    def _build_model(self):
        config = deepcopy(self.config)
        self.model, self.collate_fn = BuildModel(config)

    def _fit_model(self, train_data, valid_data=None):
        train_data = self._parse_input_data(train_data)
        if valid_data is not None: valid_data = self._parse_input_data(valid_data)
        
        if self.config['mode'] == 'binary': eval_metric = 'auc'
        elif self.config['mode'] == 'multiclass': eval_metric = 'acc'

        transtab.train(
            self.model,
            train_data, 
            valset=valid_data, 
            eval_metric=eval_metric,
            output_dir=self.checkout_dir,
            collate_fn = self.collate_fn,
            **self.config
            )
        
    def _parse_input_data(self, inputs):
        def _check_input(input):
            if isinstance(input, pd.DataFrame):
                return input
            if isinstance(input, TabularPatientBase):
                return input.df

        if isinstance(inputs, list):
            data = []
            for input in inputs:
                x = _check_input(input['x'])
                data.append((x, input['y']))

        if isinstance(inputs, dict):
            data = (_check_input(inputs['x']), inputs['y'])

        if isinstance(inputs, pd.DataFrame):
            data = inputs
        
        if isinstance(inputs, TabularPatientBase):
            data = inputs.df

        return data

    def _input_data_check(self, inputs):
        '''
        Check the training / testing data fits the formats.
        Target to (1) check if inputs valid,
                    if not, give tips about the data problem.

        Parameters
        ----------
        inputs: [{
                'x': TabularPatientBase or pd.DataFrame,
                'y': pd.Series or np.ndarray
                },...]
                'x' contain all patient features; 'y' contain labels
                for each row.
        '''
        def _check_input(input):
            if isinstance(input, dict):
                assert 'x' in input, 'No input patient data found in inputs.'
                assert isinstance(input['x'], pd.DataFrame) or isinstance(input['x'], TabularPatientBase), 'Get unaccepted input data format, expect `pd.DataFrame` or `TabularPatientBase`, get {} instead.'.format(type(inputs['x']))
                if 'y' in input:
                    assert isinstance(input['y'], pd.Series) or isinstance(input['y'], np.ndarray)
                    assert not pd.isnull(input['y']).any(), 'Find NaN in input targets, please check.'
            if isinstance(input['x'], pd.DataFrame):
                assert not input['x'].isnull().values.any(), 'Find NaN in input dataframe, please check your input, or try to pass `TabularPatientBase` as inputs.'
            if isinstance(input['x'], TabularPatientBase):
                assert not input['x'].df.isnull().values.any(), 'Find NaN in input dataset, please check your input, or try to pass `TabularPatientBase` as inputs.'

        if isinstance(inputs, list):
            for input in inputs:
                _check_input(input)
        else:
            _check_input(inputs)
