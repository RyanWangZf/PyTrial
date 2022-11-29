import abc
from collections import defaultdict
import pdb
import os
import json
import math

import torch
from torch import nn
import numpy as np

from pytrial.utils.check import check_model_dir
from pytrial.utils.check import make_dir_if_not_exist
from ..data import TrialData, PatientData

class PatientTrialMatchBase(abc.ABC):
    '''Abstract class for all sequential patient data simulations.
    
    Parameters
    ----------
    experiment_id: str, optional (default = 'test')
        The name of current experiment.
    '''
    training=False
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
        assert 'patient' in inputs, 'Do not find patient data in inputs!'
        assert 'trial' in inputs, 'Do not find trial data in inputs!'
        assert isinstance(inputs['trial'], TrialData), 'Input trial data is not a `trial_patient_match.data.TrialData` instance!'
        assert isinstance(inputs['patient'], PatientData), 'Input patient data is not a `trial_patient_match.data.PatientData` instance!'
    
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

    def _tuple_result_to_dict(self, pred_res):
        res = defaultdict(list)
        for nctid, pred in pred_res:
            if isinstance(nctid, list):
                for i,nctid_ in enumerate(nctid):
                    res[nctid_] = pred[i].tolist()
            else:
                res[nctid] = pred.flatten().tolist()
        return res

    def _translate_dense_visits_to_sparse(self, visits):
        total_vocab_size = sum(self.config['vocab_size'])
        num_visits = len(visits[self.config['order'][0]])
        outputs = np.zeros((num_visits, total_vocab_size))

        for i, o in enumerate(self.config['order']):
            for j in range(num_visits):
                raw = visits[o][j]
                if isinstance(raw, torch.Tensor): raw = raw.detach().cpu().numpy()
                if i > 0: 
                    voc_size = sum(self.config['vocab_size'][:i-1])
                    if isinstance(raw, list):
                        raw = [r + voc_size for r in raw]
                    else:
                        raw += voc_size
                outputs[j, raw] = 1

        return outputs

    def _match_trial_for_patients(self, pred_inc, criteria_type='inc'):
        pred_label_ec = torch.max(pred_inc, -1)[1] # 1，num_patients, num_ec
        num_ec = pred_label_ec.shape[-1]

        # prediction logits
        # 0 is unmatch
        # 1 is match
        # 2 is unknown

        if criteria_type == 'inc':
            # prediction
            pred_label_trial_inc = (pred_label_ec == 1).sum(-1) # 1 is match
            pred_label_trial_inc[pred_label_trial_inc<num_ec] = 0 # does not satisfy all criteria, not matched
            pred_label_trial_inc[pred_label_trial_inc==num_ec] = 1 # 1, num_patients
            return pred_label_trial_inc

        else:
            pred_label_trial_exc = (pred_label_ec == 0).sum(-1) # 0 is unmatch, do not match all exclusion criteria then sum == num_ec
            pred_label_trial_exc[pred_label_trial_exc<num_ec] = 0 # has more than 1 exclusion criteria satisfied, not matched
            pred_label_trial_exc[pred_label_trial_exc==num_ec] = 1 # 1, num_patients
            return pred_label_trial_exc