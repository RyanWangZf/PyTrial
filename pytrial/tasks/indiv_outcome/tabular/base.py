import abc
import os
import json
import pdb

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from pytrial.data.patient_data import TabularPatientBase
from pytrial.utils.check import check_model_dir
from pytrial.utils.check import make_dir_if_not_exist
from pytrial.utils.parallel import batch_to_device

from ..losses import BinaryXentLoss, XentLoss, MSELoss, MultilabelBinaryXentLoss
from ..trainer import IndivTabTrainer


class TabularIndivBase(abc.ABC):
    '''Abstract class for all individual outcome predictions
    based on tabular patient data.

    Parameters
    ----------
    experiment_id: str, optional (default = 'test')
        The name of current experiment.
    '''
    @abc.abstractmethod
    def __init__(self, experiment_id='test'):
        check_model_dir(experiment_id)
        self.checkout_dir = os.path.join('./experiments_records', experiment_id,
                                         'checkpoints')
        self.result_dir = os.path.join('./experiments_records', experiment_id,
                                       'results')
        make_dir_if_not_exist(self.checkout_dir)
        make_dir_if_not_exist(self.result_dir)

    @abc.abstractmethod
    def fit(self, train_data, valid_data):
        '''
        Fit function needs to be implemented after subclass.

        Parameters
        ----------
        train_data: Any
            Training data.
        
        valid_data: Any
            Validation data.
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, test_data):
        '''
        Prediction function needs to be implemented after subclass.

        Parameters
        ----------
        test_data: Any
            Testing data.
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def load_model(self, checkpoint):
        '''
        Load the pretrained model from disk, needs to be implemented after subclass.
        
        Parameters
        ----------
        checkpoint: str
            The path to the checkpoint file.
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def save_model(self, output_dir):
        '''
        Save the model to disk, needs to be implemented after subclass.

        Parameters
        ----------
        output_dir: str
            The path to the output directory.
        '''
        raise NotImplementedError

    def train(self, mode=True):
        '''Set the model in training mode. Work similar to `model.train()` in PyTorch.
        
        Parameters
        ----------
        mode: bool, optional (default = True)
            Whether to set the model in training mode.
            ``False`` means the model is in evaluation mode.
            ``True`` means the model is in training mode.
        '''
        self.training = mode
        self.model.train()
        return self
    
    def eval(self, mode=False):
        '''Set the model in evaluation mode. Work similar to `model.eval()` in PyTorch.

        Parameters
        ----------
        mode: bool, optional (default = False)
            Whether to set the model in evaluation mode.
            ``False`` means the model is in evaluation mode.
            ``True`` means the model is in training mode.
        '''
        self.training = mode
        self.model.eval()
        return self

    def get_train_dataloader(self, inputs):
        dataset = self._build_dataset(inputs)
        dataloader = DataLoader(dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True, 
            num_workers=self.config['num_worker'],
            pin_memory=True,
            )
        return dataloader

    def _build_dataset(self, inputs):
        df = inputs['x']
        
        if 'y' in inputs: label = inputs['y']
        else: label = None

        return IndivTabDataset(df, label)

    def _build_loss_model(self):
        mode = self.config['mode']
        if mode == 'multiclass':
            return XentLoss(self.model)

        elif mode == 'binary':
            return BinaryXentLoss(self.model)

        elif mode == 'regression':
            return MSELoss(self.model)

        elif mode == 'multilabe':
            return MultilabelBinaryXentLoss(self.model)

        else:
            raise ValueError(f'Do not recognize mode `{mode}`, please correct.')

    def _fit_model(self, train_data, valid_data=None):
        train_dataloader = self.get_train_dataloader(train_data)
        loss_model = self._build_loss_model()
        train_objectives = [(train_dataloader, loss_model)]

        mode = self.config['mode']
        test_metric_dict = {
            'binary': 'auc',
            'multiclass': 'acc',
            'regression': 'mse',
            'multilabel': 'f1', # take average of F1 scores
        }

        trainer = IndivTabTrainer(
            model=self,
            train_objectives=train_objectives,
            test_data=valid_data,
            test_metric=test_metric_dict[mode],
        )

        trainer.train(
            **self.config
        )

    def _parse_input_data(self, inputs):
        if isinstance(inputs, dict):
            if isinstance(inputs['x'], TabularPatientBase):
                dataset = inputs['x']
                x_feat = dataset.df
                y = inputs['y']
            else:
                x_feat = inputs['x']
                y = inputs['y']                
            
            if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
                y = y.values
        
        if isinstance(inputs, pd.DataFrame) or isinstance(inputs, torch.Tensor):
            x_feat, y = inputs, None
        
        if isinstance(inputs, TabularPatientBase):
            x_feat, y = inputs.df, None

        return x_feat, y

    def _predict_on_dataloader(self, dataloader):
        pred_list, label_list = [], []
        for batch in dataloader:
            x_feat = batch['x']
            x_feat = x_feat.to(self.device)
            if 'y' in batch: label_list.append(batch.pop('y'))
            pred = self.model(x_feat)
            pred_list.append(pred)
        pred = torch.cat(pred_list)
        label = torch.cat(label_list) if len(label_list) > 0 else None
        return {'pred':pred, 'label':label}

    def _save_config(self, config, output_dir=None):
        if output_dir is None:
            output_dir = self.checkout_dir

        temp_path = os.path.join(output_dir, 'config.json')
        if os.path.exists(temp_path):
            os.remove(temp_path)
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(
                json.dumps(config, indent=4)
            )

    def _load_config(self, checkpoint=None):
        '''
        Load model config from the given directory.

        Parameters
        ----------
        checkpoint: str
            The given filepath (e.g., ./checkpoint/config.json)
            to load the model config.
        '''
        if checkpoint is None:
            temp_path = os.path.join(checkpoint, 'config.json')
        else:
            temp_path = checkpoint

        assert os.path.exists(temp_path), 'Cannot find `config.json` under {}'.format(self.checkout_dir)
        with open(temp_path, 'r') as f:
            config = json.load(f)
        return config

    def _save_checkpoint(self, state,
                        epoch_id=0,
                        is_best=False,
                        output_dir=None,
                        filename='checkpoint.pth.tar'):
        if output_dir is None:
            output_dir = self.checkout_dir

        if epoch_id < 1:
            filepath = os.path.join(output_dir, 'latest.' + filename)
        elif is_best:
            filepath = os.path.join(output_dir, 'best.' + filename)
        else:
            filepath = os.path.join(self.checkout_dir,
                                    str(epoch_id) + '.' + filename)
        torch.save(state, filepath)

    def _input_data_check(self, inputs):
        '''
        Check the training / testing data fits the formats.
        Target to (1) check if inputs valid,
                    if not, give tips about the data problem.

        Parameters
        ----------
        inputs: {
                'x': TabularPatientBase or pd.DataFrame,
                'y': pd.Series or np.ndarray
                }
                'x' contain all patient features; 'y' contain labels
                for each row.
        '''
        if isinstance(inputs, dict):
            assert 'x' in inputs, 'No input patient data found in inputs.'
            assert isinstance(inputs['x'], pd.DataFrame) or isinstance(inputs['x'], TabularPatientBase), 'Get unaccepted input data format, expect `pd.DataFrame` or `TabularPatientBase`, get {} instead.'.format(type(inputs['x']))
            if 'y' in inputs:
                assert isinstance(inputs['y'], pd.Series) or isinstance(inputs['y'], np.ndarray)
                assert not pd.isnull(inputs['y']).any(), 'Find NaN in input targets, please check.'
        if isinstance(inputs['x'], pd.DataFrame):
            assert not inputs['x'].isnull().values.any(), 'Find NaN in input dataframe, please check your input, or try to pass `TabularPatientBase` as inputs.'
        if isinstance(inputs['x'], TabularPatientBase):
            assert not inputs['x'].df.isnull().values.any(), 'Find NaN in input dataset, please check your input, or try to pass `TabularPatientBase` as inputs.'


class IndivTabDataset(Dataset):
    def __init__(self, df, label=None):
        self.df = df
        self.label = label
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if self.label is not None: return {'x':self.df.iloc[idx].values, 'y': self.label[idx]}
        else: return {'x':self.df.iloc[idx].values}
