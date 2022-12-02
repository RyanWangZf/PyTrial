import abc
import pdb
import os
import torch
import json

from pytrial.utils.check import check_model_dir
from pytrial.utils.check import make_dir_if_not_exist

class SiteSelectionBase(abc.ABC):
    '''Abstract class for all sequential patient data simulations.
    
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
    def fit(self, train_data):
        '''
        Fit function needs to be implemented after subclass.

        Parameters
        ----------
        train_data: Any
            Training data.
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
    def save_model(self, output_dir):
        '''
        Save the model to disk, needs to be implemented after subclass.

        Parameters
        ----------
        output_dir: str
            The path to the output directory.
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

    def train(self, mode=True):
        '''
        Swith the model to the `training` mode. Work samely as `model.train()` in pytorch.

        Parameters
        ----------
        mode: bool, optional (default = True)
            If True, switch to the `training` mode.
        '''
        self.training = mode
        self.model.train()
        return self
    
    def eval(self, mode=False):
        '''
        Swith the model to the `validation` mode. Work samely as `model.eval()` in pytorch.

        Parameters
        ----------
        mode: bool, optional (default = False)
            If False, switch to the `validation` mode.
        '''
        self.training = mode
        self.model.eval()
        return self

    def _input_data_check(self, inputs):
        assert isinstance(inputs, TrialSiteSimple), 'Wrong input type.'
    
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