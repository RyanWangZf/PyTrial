import pdb

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import LSTMCell
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np

from pytrial.utils.check import (
    check_checkpoint_file, check_model_dir, check_model_config_file, make_dir_if_not_exist
)
from .base import SequenceIndivBase
from ..trainer import IndivSeqTrainer

class RaimExtract(nn.Module):

    def __init__(self, input_size, window_size, hidden_size):
        super(RaimExtract, self).__init__()

        self.input_size = input_size
        self.window_size = window_size
        self.hidden_size = hidden_size

        self.w_h_alpha = nn.Linear(self.hidden_size, self.window_size)
        self.w_a_alpha = nn.Linear(self.input_size, 1)

        self.w_h_beta = nn.Linear(self.hidden_size, self.input_size)
        self.w_a_beta = nn.Linear(self.window_size, 1)

        self.activate_func = nn.Tanh()
        self.weights_func = nn.Softmax(dim = -1)

    def forward(self, input_data, h_t_1):
        # shape of input_data : <n_batch, window_size, input_size> 
        # shape of h_t_1      : <n_batch, hidden_size>
        # shape of _alpha_weights : <n_batch, window_size> 
        _alpha_weights = self.activate_func(self.w_h_alpha(h_t_1) +\
                                            self.w_a_alpha(input_data.reshape(-1, self.input_size)).\
                                            squeeze().\
                                            reshape(-1, self.window_size)
                                           )
        _alpha_weights = self.weights_func(_alpha_weights)

        # shape of _beta_weights : <n_batch, input_size> 
        _beta_weights = self.activate_func(self.w_h_beta(h_t_1) +\
                                           self.w_a_beta(input_data.permute(0, 2, 1).\
                                                         reshape(-1, self.window_size)).\
                                           squeeze().\
                                           reshape(-1, self.input_size)
                                          )
        _beta_weights = self.weights_func(_beta_weights)

        # shape of _alpha_weights_v : <n_batch, window_size, 1> 
        _alpha_weights_v = _alpha_weights.unsqueeze(-1)
        # shape of _beta_weights_v  : <n_batch, 1, input_size> 
        _beta_weights_v = _beta_weights.unsqueeze(1)

        # shape of weight_A  : <n_batch, window_size, input_size> 
        weight_A = _alpha_weights_v * _beta_weights_v

        # shape of output_v  : <n_batch, input_size> 
        output_v = torch.sum(weight_A * input_data, dim = 1)
        
        return output_v

class BuildModel(nn.Module):
    def __init__(self, 
                 input_size = None,
                 window_size = 3,
                 hidden_size = 16,
                 output_size = 8,
                 label_size = 1,
                 device = None):
        super(BuildModel, self).__init__()
        assert input_size != None and isinstance(input_size, int), 'fill in correct input_size' 
        self.input_size = input_size
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.label_size = label_size
        self.rnn_raimf = RaimExtract(input_size, window_size, hidden_size)
        self.rnn_unit = LSTMCell(input_size, hidden_size) 
        self.predict_func = nn.Linear(self.output_size, self.label_size)
        self.pad = nn.ConstantPad1d((self.window_size-1, 0), 0.)
        self.device = device

    def forward(self, input_data):
        """
        
        Parameters
        
        ----------
        input_data = {
                      'X': shape (batchsize, n_timestep, n_featdim)
                      'M': shape (batchsize, n_timestep), the first visit to the current visit are all 1, the later are 0, i.e., [1,1,1,0,0].
                      'cur_M': shape (batchsize, n_timestep), only the current visit is 1 others are 0, i.e.,[0,0,1,0,0]
                      'T': shape (batchsize, n_timestep), the time interval between visits, i.e., for timestamp [0, 1, 2, 3, 4], T = [0, 1, 1, 1, 1].
                     }
        
        Return
        
        ----------
        
        all_output, shape (batchsize, n_timestep, n_labels)
            
            predict output of each time step
            
        cur_output, shape (batchsize, n_labels)
        
            predict output of last time step
        
        """
        X = input_data['v']
        
        if 'v_lengths' in input_data:
            v_len = input_data['v_lengths']
            mask, cur_mask = self._create_mask_from_visit_length(v_len=v_len)
            mask = mask.to(X.device)
            cur_mask = cur_mask.to(X.device)
        else:
            mask = cur_mask = None

        batchsize, n_timestep, n_f = X .shape
        # shape of X     : <batchsize, n_timestep, n_featdim>        
        # shape of pad_X : <batchsize, n_timestep + window_size - 1, n_featdim>        
        pad_X = self.pad(X.permute(0,2,1)).permute(0,2,1)
        
        h0 = Variable(torch.zeros(batchsize, self.hidden_size)).to(self.device)
        c0 = Variable(torch.zeros(batchsize, self.hidden_size)).to(self.device)
        outs = []        
        hn = h0
        cn = c0
        for i in range(self.window_size-1, n_timestep+self.window_size-1):
            cur_x = pad_X[ : , i - self.window_size + 1 : i + 1, : ]
            z_value = self.rnn_raimf(cur_x, hn)
            hn, cn = self.rnn_unit(z_value, (hn, cn))
            outs.append(hn)
        outputs = torch.stack(outs, dim =1)
        n_batchsize, n_timestep, n_featdim = outputs.shape
        all_output =  self.predict_func(outputs.reshape(n_batchsize*n_timestep, n_featdim)).\
                    reshape(n_batchsize, n_timestep, self.label_size) # bs, timestep, 1

        if mask is not None and cur_mask is not None:
            all_output *= mask.unsqueeze(-1)
            cur_output = (all_output * cur_mask.unsqueeze(-1)).sum(dim=1)
        else:
            cur_output = all_output.sum(1) 

        # TODO: all_output is the per visit label prediction, currently we only support patient-level prediction.
        return cur_output

    def _create_mask_from_visit_length(self, v_len):
        mask = torch.zeros(len(v_len),max(v_len))
        cur_mask = torch.zeros(len(v_len),max(v_len))

        for i in range(len(v_len)):
            mask[i, :v_len[i]] = 1
            cur_mask[i, v_len[i]-1] = 1
        return mask, cur_mask

class RAIM(SequenceIndivBase):
    '''
    Implment RAIM for longitudinal patient records predictive modeling [1]_.

    Parameters
    ----------
    vocab_size: list[int]
        A list of vocabulary size for different types of events, e.g., for diagnosis, procedure, medication.
    
    orders: list[str]
        A list of orders when treating inputs events. Should have the same shape of `vocab_size`.

    mode: str
        Prediction traget in ['binary','multiclass','multilabel','regression'].

    output_dim: int
        Output dimension of the model.

        - If binary classification, output_dim=1;
        - If multiclass/multilabel classification, output_dim=n_class;
        - If regression, output_dim=1.

    max_visit: int
        The maximum number of visits for input event codes.

    window_size: int
        The window size in the recurrent part of RAIM.
    
    hidden_size: int
        Embedding size for encoding input event codes.
    
    hidden_output_size: int
        The output size of the intermediate layers. Not the output_dim.

    dropout: float
        Dropout rate in the intermediate layers.
    
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
        The model device.
    
    experiment_id: str
        The prefix when saving the checkpoints during the training.
    
    Notes
    -----
    .. [1] Xu, Y., Biswal, S., Deshpande, S. R., Maher, K. O., & Sun, J. (2018, July). 
        Raim: Recurrent attentive and intensive model of multimodal patient monitoring data. 
        In Proceedings of the 24th ACM SIGKDD international conference on Knowledge Discovery & Data Mining (pp. 2565-2573).

    '''
    def __init__(self, 
        vocab_size,
        orders,
        mode=None, 
        output_dim=None,
        max_visit=None,
        window_size=3,
        hidden_size=8,
        hidden_output_size=8,
        dropout=0.5,
        learning_rate=1e-4,
        weight_decay=1e-4,
        batch_size=64,
        epochs=10,
        num_worker=0,
        device='cuda:0',
        experiment_id='test', 
        ):
        super().__init__(experiment_id, mode, output_dim)
        self.config = {
            'mode':self.mode,
            'vocab_size':vocab_size,
            'max_visit':max_visit,
            'window_size':window_size,
            'hidden_size':hidden_size,
            'hidden_output_size':hidden_output_size,
            'output_dim':self.output_dim,
            'dropout':dropout,
            'device':device,
            'learning_rate':learning_rate,
            'batch_size':batch_size,
            'weight_decay':weight_decay,
            'epochs':epochs,
            'num_worker':num_worker,
            'orders':orders,
            }
        self.config['total_vocab_size'] = sum(vocab_size)
        self.device = device
        self._build_model()

    def load_model(self, checkpoint):
        '''
        Load pretrained model from the disk.

        Parameters
        ----------
        checkpoint: str
            The input directory that stores the trained pytorch model and configuration.
        '''
        checkpoint_filename = check_checkpoint_file(checkpoint)
        config_filename = check_model_config_file(checkpoint)
        state_dict = torch.load(checkpoint_filename)
        if config_filename is not None:
            config = self._load_config(config_filename)
            self.config.update(config)
        
        self.model.load_state_dict(state_dict['model'])

    def save_model(self, output_dir):
        '''
        Save the pretrained model to the disk.

        Parameters
        ----------
        output_dir: str
            The output directory that stores the trained pytorch model and configuration.
        '''
        if output_dir is not None:
            make_dir_if_not_exist(output_dir)
        else:
            output_dir = self.checkout_dir
        
        self._save_config(self.config, output_dir=output_dir)
        self._save_checkpoint({'model':self.model.state_dict()}, output_dir=output_dir)

    def fit(self, train_data, valid_data=None):
        '''
        Train model with sequential patient records.
        
        Parameters
        ----------
        train_data: SequencePatientBase
            A `SequencePatientBase` contains patient records where 'v' corresponds to 
            visit sequence of different events; 'y' corresponds to labels.

        valid_data: SequencePatientBase
            A `SequencePatientBase` contains patient records used to make early stopping of the
            model.
        '''
        self._input_data_check(train_data)
        if valid_data is not None: self._input_data_check(valid_data)
        self._fit_model(train_data, valid_data=valid_data)


    def predict(self, test_data):
        '''
        Predict patient outcomes using longitudinal trial patient sequences.
        
        Parameters
        ----------
        test_data: SequencePatient
            A `SequencePatient` contains patient records where 'v' corresponds to 
            visit sequence of different events.

        '''
        test_dataloader = self.get_test_dataloader(test_data)
        outputs = self._predict_on_dataloader(test_dataloader)
        return outputs['pred']

    def _build_model(self):
        self.model = BuildModel(
            input_size=self.config['total_vocab_size'],
            window_size=self.config['window_size'],
            hidden_size=self.config['hidden_size'],
            output_size=self.config['hidden_output_size'],
            label_size=self.config['output_dim'],
            device=self.config['device'],
            )
        self.model.to(self.device)

    def _fit_model(self, train_data, valid_data=None):
        test_metric_dict = {
            'binary': 'auc',
            'multiclass': 'acc',
            'regression': 'mse',
            'multilabel': 'f1', # take average of F1 scores
            }
        train_dataloader = self.get_train_dataloader(train_data)
        loss_models = self._build_loss_model()
        train_objectives = [(train_dataloader, loss_model) for loss_model in loss_models]
        trainer = IndivSeqTrainer(model=self,
            train_objectives=train_objectives,
            test_data=valid_data,
            test_metric=test_metric_dict[self.config['mode']],
            )
        trainer.train(**self.config)