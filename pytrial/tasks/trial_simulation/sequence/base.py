import abc
import pdb
import os
import json
import math

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from pytrial.utils.check import check_model_dir
from pytrial.utils.check import make_dir_if_not_exist
from pytrial.tasks.trial_simulation.data import SequencePatient

class SequenceSimulationBase(abc.ABC):
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
        '''Fit the model to the given training data. Need to be implemented by the child class.

        Parameters
        ----------
        train_data: SequencePatient
            The training data.
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, number_of_predictions):
        '''Generate synthetic sequence patient data. Need to be implemented by the child class.
        
        Parameters
        ----------
        number_of_predictions: int
            The number of synthetic sequence patient data to generate.
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def save_model(self, output_dir):
        '''Save the model to the given directory. Need to be implemented by the child class.

        Parameters
        ----------
        output_dir: str
            The given directory to save the model.
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def load_model(self, checkpoint):
        '''Load the model from the given directory. Need to be implemented by the child class.
        
        Parameters
        ----------
        checkpoint: str
            The given directory to load the model.
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

    def _input_data_check(self, inputs):
        assert isinstance(inputs, SequencePatient), 'Wrong input type.'
    
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
    
    def _compute_n_per_sample(self, n_test_sample, n=None, n_per_sample=None):
        if n_per_sample is not None:
            n_total = n_test_sample*n_per_sample
            if n is not None:
                n_total = min(n_total, n)
            return n_total, n_per_sample
        else:
            return n, math.ceil(n_test_sample / n)

    def _get_num_visit(self, data, idx):
        visits = data['v']
        num_visit_list = []
        for k,v in visits.items():
            num_visit_list.append(len(v[idx]))
        
        num_visit_uq = list(set(num_visit_list))
        assert len(num_visit_uq) == 1, f'Find mismatch in the number of visit events {num_visit_list}, please check the input data {visits}.'
        return num_visit_uq[0]

    def _prepare_input(self, data, idx, vdx):
        '''
        Prepare inputs for sequence simulation models.

        Parameters
        ----------
        data: dict[list]
            A single patient records.

        idx: int
            The patient index.

        vdx: int
            The target visit index.

        '''
        visits = data['v']
        feature = data['x']
        if not isinstance(feature, torch.Tensor): feature = torch.tensor(feature)

        feature = feature.to(self.device)

        inputs = {
            'v':{},
            'y':{},
            'x':feature[idx], # baseline feature
            }

        for k in visits.keys():
            v = pad_sequence(list(map(torch.tensor,visits[k][idx][:vdx])), 
                    batch_first=True)
            
            y = torch.tensor(visits[k][idx][vdx])

            v = v.to(self.device)
            y = y.to(self.device)

            inputs['v'][k] = v
            inputs['y'][k] = y

        return inputs

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

class GAN(nn.Module):
    '''
    Build generative adversarial network module referring to MedGAN.
    '''
    def __init__(self,
        emb_size,
        total_vocab_size,
        gen_dims,
        dis_dims,
        ) -> None:
        super().__init__()

        # generator and discriminator part
        self.gen_module = nn.ModuleDict({
            'fc1': nn.Linear(emb_size*2, gen_dims[0]), # from z+s -> hidden feat
            'fc2': nn.Linear(gen_dims[0], gen_dims[1]),
            'fc3': nn.Linear(gen_dims[1], total_vocab_size),
        })

        self.dis_module = nn.ModuleDict({
            'fc1': nn.Linear(total_vocab_size, dis_dims[0]), # x,\bar{x} -> {0,1}
            'fc2': nn.Linear(dis_dims[0], 1),
        })

    def forward(self, s):
        '''forward makes generation taking the state embeddings as inputs.
        '''
        z_random = torch.randn(s.size()).to(s.device)
        return self.infer_generator(s, z_random)
        
    def infer_generator(self, s, z):
        '''do generator inference, x
        args:
            s: state vector [bs, emb_size]
            z: visit or random [bs, emb_size]
        '''
        inputs = torch.cat([z,s], -1) # bs, emb_size*2
        tempVec = inputs
        h = self.gen_module['fc1'](tempVec) # bs, 256
        h = nn.ReLU()(h)

        tempVec = h + tempVec

        h = self.gen_module['fc2'](tempVec)
        h = nn.ReLU()(h)
        tempVec = h + tempVec # bs, 256

        h = self.gen_module['fc3'](tempVec)
        outputs = nn.Sigmoid()(h)
        return outputs

    def infer_discriminator(self, x):
        '''do discriminator inference
        args:
            x: the decoded fake records or the true records x
        '''
        h = self.dis_module['fc1'](x)
        h = nn.ReLU()(h)
        h = self.dis_module['fc2'](h)
        pred = h.sigmoid().squeeze(1)
        return pred


class RNN(nn.Module):
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