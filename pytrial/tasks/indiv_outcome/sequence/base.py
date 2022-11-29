import abc
import pdb
import os
import json

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

from pytrial.utils.check import check_model_dir
from pytrial.utils.check import make_dir_if_not_exist
from pytrial.data.patient_data import SequencePatientBase
from pytrial.data.patient_data import SeqPatientCollator
from ..losses import XentLoss, BinaryXentLoss, MSELoss, MultilabelBinaryXentLoss


class SequenceIndivBase(abc.ABC):
    '''Abstract class for all individual outcome predictions
        based on sequential patient data.

    Parameters
    ----------
    experiment_id: str, optional (default = 'test')
        The name of current experiment.
    '''
    _mode_list = ['binary','multiclass','multilabel','regression']
    training=False
    @abc.abstractmethod
    def __init__(self, experiment_id='test', mode=None, output_dim=None):
        check_model_dir(experiment_id)
        self.checkout_dir = os.path.join('./experiments_records', experiment_id,
                                         'checkpoints')
        self.result_dir = os.path.join('./experiments_records', experiment_id,
                                       'results')
        make_dir_if_not_exist(self.checkout_dir)
        make_dir_if_not_exist(self.result_dir)
        self._check_mode_and_output_dim(mode, output_dim)

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

    def get_test_dataloader(self, test_data):
        dataloader = DataLoader(test_data,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_worker'],
            pin_memory=True,
            shuffle=False,
            collate_fn=SeqPatientCollator(
                config={
                    'visit_mode':test_data.metadata['visit']['mode'],
                    'label_mode':test_data.metadata['label']['mode'],
                    }
                ),
            )
        return dataloader

    def get_train_dataloader(self, train_data):
        dataloader = DataLoader(train_data,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_worker'],
            pin_memory=False,
            shuffle=True,
            collate_fn=SeqPatientCollator(
                config={
                    'visit_mode':train_data.metadata['visit']['mode'],
                    'label_mode':train_data.metadata['label']['mode'],
                    }
                ),
            )
        return dataloader

    def _input_data_check(self, inputs):
        assert isinstance(inputs, SequencePatientBase), 'Wrong input type.'

    def _check_mode_and_output_dim(self, mode, output_dim):
        mode = mode.lower()
        assert mode in self._mode_list, f'Input mode `{mode}` does not belong to the supported mode list {self._mode_list}.'
        if output_dim is None:
            if mode not in ['binary','regression']:
                raise ValueError('`output_dim` should be given when `mode` is not `binary` or `regression`.')
            else:
                output_dim = 1
        self.mode = mode
        self.output_dim = output_dim
    
    def _save_config(self, config, output_dir):
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

    def _save_checkpoint(self, 
        state_dict,
        epoch_id=0,
        is_best=False,
        output_dir=None, 
        filename='checkpoint.pth.tar'
        ):
        if output_dir is None:
            output_dir = self.checkout_dir

        if epoch_id < 1:
            filepath = os.path.join(output_dir, 'latest.' + filename)
        elif is_best:
            filepath = os.path.join(output_dir, 'best.' + filename)
        else:
            filepath = os.path.join(self.checkout_dir,
                                    str(epoch_id) + '.' + filename)
        torch.save(state_dict, filepath)

    @torch.no_grad()
    def _predict_on_dataloader(self, test_dataloader):
        pred_list, label_list = [], []
        for batch in test_dataloader:
            inputs = self._prepare_input(batch)
            logits = self.model(inputs)
            pred_list.append(logits)
            if 'y' in batch: label_list.append(batch.pop('y'))
        pred = torch.cat(pred_list, dim=0)
        if self.config['mode'] in ['binary','multilabel']:
            pred = torch.sigmoid(pred)
        if self.config['mode'] == 'multiclass':
            pred = torch.softmax(pred, dim=1)
        pred = pred.cpu().numpy()
        label = torch.cat(label_list) if len(label_list) > 0 else None
        return {'pred':pred,'label':label}

    def _build_loss_model(self):
        mode = self.config['mode']
        if mode == 'binary':
            return [BinaryXentLoss(self.model)]

        if mode == 'multiclass':
            return [XentLoss(self.model)]

        if mode == 'multilabel':
            return [MultilabelBinaryXentLoss(self.model)]

        if mode == 'binary':
            return [MSELoss(self.model)]

    def _prepare_input(self, data):
        '''
        Prepare inputs for sequential patient record predictive models.

        Parameters
        ----------
        data: dict[list]
            A batch of patient records.

        '''
        visits = data['v']
        feature = data['x']

        if not isinstance(feature, torch.Tensor): feature = torch.tensor(feature)
        feature = feature.to(self.device)

        inputs = {
            'v':{},
            'v_lengths':[],
            'x':feature, # baseline feature
            }
        
        if self.config['max_visit'] is None:
            v_lengths = [len(visits[self.config['orders'][0]][idx]) for idx in range(len(visits[self.config['orders'][0]]))]
        else:
            max_visit = self.config['max_visit']
            v_lengths = [len(visits[self.config['orders'][0]][idx][:max_visit]) for idx in range(len(visits[self.config['orders'][0]]))]

        inputs['v_lengths'] = v_lengths

        v = torch.zeros(len(v_lengths), max(v_lengths), self.config['total_vocab_size'])
        for idx in range(len(v_lengths)):
            v[idx,:v_lengths[idx]] = torch.tensor(self._translate_dense_visits_to_sparse({k: visits[k][idx][:self.config['max_visit']] for k in visits}))
        v = v.to(self.device)
        inputs['v'] = v

        if 'y' in data: # target labels
            target = data['y']
            if not isinstance(target, torch.Tensor): target = torch.tensor(target)
            target = target.to(self.device)
            inputs['y'] = target
        return inputs

    def _translate_dense_visits_to_sparse(self, visits):
        total_vocab_size = sum(self.config['vocab_size'])
        num_visits = len(visits[self.config['orders'][0]])
        outputs = np.zeros((num_visits, total_vocab_size))

        for i, o in enumerate(self.config['orders']):
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



class InputEventEmbedding(nn.Module):
    def __init__(self, orders, vocab_size, emb_size, padding_idx) -> None:
        super().__init__()
        # build input embeddings
        emb_dict = {}
        for i, order in enumerate(orders):
            emb_dict[order] = nn.Embedding(vocab_size[i], embedding_dim=emb_size, padding_idx=padding_idx)
        self.embeddings = nn.ModuleDict(emb_dict)
    
    def forward(self, inputs):
        emb_list = []
        for k, v in inputs['v'].items():
            emb = self.embeddings[k](v)
            emb_list.append(emb)
        embs = torch.cat(emb_list, 1)
        return embs

class RNNModel(nn.Module):
    RNN_TYPE = {
        'rnn':nn.RNN,
        'lstm':nn.LSTM,
        'gru':nn.GRU,
    }
    def __init__(self, 
        rnn_type,
        emb_size,
        num_layer,
        bidirectional,
        ) -> None:
        super().__init__()
        self.model = self.RNN_TYPE[rnn_type](
            input_size=emb_size,
            hidden_size=emb_size,
            num_layers=num_layer,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.bidirectional = bidirectional
    
    def forward(self, x):
        outputs = self.model(x)[0]
        return outputs