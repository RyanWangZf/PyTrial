import pdb
import os

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from pytrial.data.patient_data import TabularPatientBase
from pytrial.utils.check import (
    check_checkpoint_file, check_model_dir, check_model_config_file, make_dir_if_not_exist
)

from .base import TabularIndivBase, IndivTabDataset

class BuildModel(nn.Module):
    def __init__(self,
        input_dim,
        output_dim,
        hidden_dim=128,
        num_layer=2,
        ) -> None:
        super().__init__()
        if num_layer == 1:
            self.mlp = nn.ModuleList([nn.Linear(input_dim, output_dim)])
        else:
            self.mlp = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])

        for _ in range(num_layer-2):
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Linear(hidden_dim, hidden_dim))
        
        if num_layer > 1:
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Linear(hidden_dim, output_dim))
    
    def forward(self, inputs):
        if isinstance(inputs, dict):
            h = inputs['x']
        elif isinstance(inputs, torch.Tensor):
            h = inputs
        else:
            raise ValueError('inputs should be dict or torch.Tensor')

        h = h.float()
        for layer in self.mlp:
            h = layer(h)
        return h

class MLP(TabularIndivBase):
    '''
    Implement multi-layer perceptron model for tabular individual outcome
    prediction in clinical trials.

    Parameters
    ----------
    input_dim: int
        Dimension of the input features.

    output_dim: int
        Dimension of the outputs. When doing classification, it equals to number of classes.

    mode: str
        The task's objectives, in `binary`, `multiclass`, `multilabel`, or `regression`

    hidden_dim: int
        Hidden dimensions of neural networks.

    num_layer: int
        Number of hidden layers.

    learning_rate: float
        Learning rate for optimization based on SGD. Use torch.optim.Adam by default.

    weigth_decay: float
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
    '''
    def __init__(self,
        input_dim,
        output_dim,
        mode,
        hidden_dim=128,
        num_layer=2,
        learning_rate=1e-4,
        weight_decay=1e-4,
        batch_size=64,
        epochs=10,
        num_worker=0,
        device='cuda:0',
        experiment_id='test'):
        super().__init__(experiment_id)

        mode = mode.lower()
        assert mode in ['binary', 'multiclass', 'regression', 'multilabel']

        self.config = {
            'input_dim':input_dim,
            'output_dim':output_dim,
            'hidden_dim':hidden_dim,
            'num_layer':num_layer,
            'learning_rate':learning_rate,
            'batch_size':batch_size,
            'weight_decay':weight_decay,
            'epochs':epochs,
            'num_worker':num_worker,
            'experiment_id':experiment_id,
            'model_name': 'MLP',
            'device':device,
            'mode':mode,
        }
        self._save_config(self.config)
        self.device = device

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

        valid_data: same as `train_data`.
            Validation data during the training for early stopping.
        '''

        self._input_data_check(train_data)
        self._build_model()
        
        x_feat, y = self._parse_input_data(train_data)
        train_data={'x':x_feat, 'y':y}

        if valid_data is not None:
            x_feat_va, y_va = self._parse_input_data(valid_data)
            valid_data = {'x':x_feat_va, 'y':y_va}
                
        self._fit_model(train_data=train_data, valid_data=valid_data)
    
    def predict(self, test_data):
        '''
        Make prediction probability based on the learned model.

        Parameters
        ----------
        test_data: Dict or TabularPatientBase or pd.DataFrame or torch.Tensor
            {'x': TabularPatientBase or pd.DataFrame}
            
            'x' contain all patient features.

        Returns
        -------
        ypred: np.ndarray or torch.Tensor
            Prediction probability for each patient.

            - For binary classification, return shape (n, );
            - For multiclass classification, return shape (n, n_class).

        '''

        if isinstance(test_data, torch.Tensor):
            return self.model(test_data)
        
        x_feat, y = self._parse_input_data(test_data)
        test_data = {'x':x_feat, 'y': y}
        dataset = self._build_dataset(test_data)
        dataloader = DataLoader(dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False, 
            num_workers=self.config['num_worker'],
            pin_memory=True,
            )
        ypred = self._predict_on_dataloader(dataloader)
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
        self._save_checkpoint({'model':self.model}, output_dir=output_dir)

    def load_model(self, checkpoint):
        '''
        Load model and the pre-encoded trial embeddings from the given
        checkpoint dir.

        Parameters
        ----------
        checkpoint: str
            The input dir that stores the pretrained model.
            - If a directory, the only checkpoint file `*.pth.tar` will be loaded.
            - If a filepath, will load from this file.
        '''
        checkpoint_filename = check_checkpoint_file(checkpoint)
        config_filename = check_model_config_file(checkpoint)
        state_dict = torch.load(checkpoint_filename)
        if config_filename is not None:
            config = self._load_config(config_filename)
            self.config.update(config)
        self.model = state_dict['model']

    def _build_model(self):
        self.model = BuildModel(
            input_dim=self.config['input_dim'],
            output_dim=self.config['output_dim'],
            hidden_dim=self.config['hidden_dim'],
            )
        self.model.to(self.config['device'])



    




