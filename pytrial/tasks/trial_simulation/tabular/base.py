import abc
import os
import json
import pandas as pd

from pytrial.utils.check import check_model_dir
from pytrial.utils.check import make_dir_if_not_exist
from pytrial.data.patient_data import TabularPatientBase


class TabularSimulationBase(abc.ABC):
    '''Abstract class for all tabular simulations
    based on tabular patient data.
    
    Parameters
    ----------
    experiment_id: str, optional (default = 'test')
        The name of current experiment.
    '''
    @abc.abstractmethod
    def __init__(self, experiment_id='trial_simulation.tabular'):
        check_model_dir(experiment_id)
        self.checkout_dir = os.path.join('./experiments_records', experiment_id,
                                         'checkpoints')
        self.result_dir = os.path.join('./experiments_records', experiment_id,
                                       'results')
        make_dir_if_not_exist(self.checkout_dir)
        make_dir_if_not_exist(self.result_dir)

    @abc.abstractmethod
    def fit(self, train_data):
        '''Fit the model to the given training data. Need to be implemented by subclass.

        Parameters
        ----------
        train_data: Any
            The training data.
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, number_of_predictions):
        '''Generate synthetic data after the model trained. Need to be implemented by subclass.

        Parameters
        ----------
        number_of_predictions: Any
            Number of synthetic data to be generated.
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def save_model(self, output_dir):
        '''Save the model to the given directory. Need to be implemented by subclass.
        
        Parameters
        ----------
        output_dir: str
            The given directory to save the model.
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def load_model(self, checkpoint):
        '''
        Load model from the given directory. Need to be implemented by subclass.
        
        Parameters
        ----------
        checkpoint: str
            The given filepath (e.g., ./checkpoint/model.pth)
        '''
        raise NotImplementedError

    def _input_data_check(self, inputs):
        assert (isinstance(inputs, TabularPatientBase) or isinstance(inputs, dict)), 'Wrong input type.'

    def _save_config(self, config, output_dir=None):
        '''
        Save model config to the given directory.

        Parameters
        ----------
        config: dict
            Hyperparameters and model infomation
            
        checkpoint: str
            The given filepath (e.g., ./checkpoint/config.json)
            to load the model config.
        '''
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
