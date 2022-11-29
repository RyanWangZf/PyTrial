import pdb

import torch
from torch import nn

from pytrial.utils.check import (
    check_checkpoint_file, check_model_config_file, make_dir_if_not_exist
)
from .base import SequenceIndivBase
from ..trainer import IndivSeqTrainer

class LocationAttention(nn.Module):

    def __init__(self, hidden_size, device):
        super(LocationAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_value_ori_func = nn.Linear(self.hidden_size, 1)
        self.device = device

    def forward(self, input_data):
        # shape of input_data: <n_batch, n_seq, hidden_size>         
        n_batch, n_seq, hidden_size = input_data.shape
        # shape of reshape_feat: <n_batch*n_seq, hidden_size>       
        reshape_feat = input_data.reshape(n_batch*n_seq, hidden_size)
        # shape of attention_value_ori: <n_batch*n_seq, 1>       
        attention_value_ori = torch.exp(self.attention_value_ori_func(reshape_feat))
        # shape of attention_value_format: <n_batch, 1, n_seq>       
        attention_value_format = attention_value_ori.reshape(n_batch, n_seq).unsqueeze(1)        
        # shape of ensemble flag format: <1, n_seq, n_seq> 
        # if n_seq = 3, ensemble_flag_format can get below flag data
        #  [[[ 0  0  0 
        #      1  0  0
        #      1  1  0 ]]]
        ensemble_flag_format = torch.triu(torch.ones([n_seq, n_seq]), diagonal = 1).permute(1, 0).unsqueeze(0).to(self.device)
        # shape of accumulate_attention_value: <n_batch, n_seq, 1>
        accumulate_attention_value = torch.sum(attention_value_format * ensemble_flag_format, -1).unsqueeze(-1) + 1e-9
        # shape of each_attention_value: <n_batch, n_seq, n_seq>
        each_attention_value = attention_value_format * ensemble_flag_format
        # shape of attention_weight_format: <n_batch, n_seq, n_seq>
        attention_weight_format = each_attention_value/accumulate_attention_value
        # shape of _extend_attention_weight_format: <n_batch, n_seq, n_seq, 1>
        _extend_attention_weight_format = attention_weight_format.unsqueeze(-1)
        # shape of _extend_input_data: <n_batch, 1, n_seq, hidden_size>
        _extend_input_data = input_data.unsqueeze(1)
        # shape of _weighted_input_data: <n_batch, n_seq, n_seq, hidden_size>
        _weighted_input_data = _extend_attention_weight_format * _extend_input_data
        # shape of weighted_output: <n_batch, n_seq, hidden_size>
        weighted_output = torch.sum(_weighted_input_data, 2)
        return weighted_output

class GeneralAttention(nn.Module):

    def __init__(self, hidden_size, device):
        super(GeneralAttention, self).__init__()
        self.hidden_size = hidden_size
        self.correlated_value_ori_func = nn.Linear(self.hidden_size, self.hidden_size)
        self.device = device

    def forward(self, input_data):
        # shape of input_data: <n_batch, n_seq, hidden_size>         
        n_batch, n_seq, hidden_size = input_data.shape
        # shape of reshape_feat: <n_batch*n_seq, hidden_size>       
        reshape_feat = input_data.reshape(n_batch*n_seq, hidden_size)
        # shape of correlated_value_ori: <n_batch, n_seq, hidden_size>       
        correlated_value_ori = self.correlated_value_ori_func(reshape_feat).reshape(n_batch, n_seq, hidden_size)
        # shape of _extend_correlated_value_ori: <n_batch, n_seq, 1, hidden_size>   
        _extend_correlated_value_ori = correlated_value_ori.unsqueeze(-2)
        # shape of _extend_input_data: <n_batch, 1, n_seq, hidden_size>   
        _extend_input_data = input_data.unsqueeze(1)
        # shape of _extend_input_data: <n_batch, n_seq, n_seq, hidden_size> 
        _correlat_value = _extend_correlated_value_ori * _extend_input_data
        # shape of attention_value_format: <n_batch, n_seq, n_seq>       
        attention_value_format = torch.exp(torch.sum(_correlat_value, dim = -1))
        # shape of ensemble flag format: <1, n_seq, n_seq> 
        # if n_seq = 3, ensemble_flag_format can get below flag data
        #  [[[ 0  0  0 
        #      1  0  0
        #      1  1  0 ]]]
        ensemble_flag_format = torch.triu(torch.ones([n_seq, n_seq]), diagonal = 1).permute(1, 0).unsqueeze(0).to(self.device)
        # shape of accumulate_attention_value: <n_batch, n_seq, 1>
        accumulate_attention_value = torch.sum(attention_value_format * ensemble_flag_format, -1).unsqueeze(-1) + 1e-10
        # shape of each_attention_value: <n_batch, n_seq, n_seq>
        each_attention_value = attention_value_format * ensemble_flag_format
        # shape of attention_weight_format: <n_batch, n_seq, n_seq>
        attention_weight_format = each_attention_value/accumulate_attention_value
        # shape of _extend_attention_weight_format: <n_batch, n_seq, n_seq, 1>
        _extend_attention_weight_format = attention_weight_format.unsqueeze(-1)
        # shape of _extend_input_data: <n_batch, 1, n_seq, hidden_size>
        _extend_input_data = input_data.unsqueeze(1)
        # shape of _weighted_input_data: <n_batch, n_seq, n_seq, hidden_size>
        _weighted_input_data = _extend_attention_weight_format * _extend_input_data
        # shape of weighted_output: <n_batch, n_seq, hidden_size>
        weighted_output = torch.sum(_weighted_input_data, 2)
        return weighted_output

class ConcatenationAttention(nn.Module):
    def __init__(self, hidden_size, attention_dim = 16, device = None):
        super(ConcatenationAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_dim = attention_dim        
        self.attention_map_func = nn.Linear(2 * self.hidden_size, self.attention_dim)
        self.activate_func = nn.Tanh()
        self.correlated_value_ori_func = nn.Linear(self.attention_dim, 1)
        self.device = device
				
    def forward(self, input_data):
        # shape of input_data: <n_batch, n_seq, hidden_size>         
        n_batch, n_seq, hidden_size = input_data.shape
        # shape of _extend_input_data: <n_batch, n_seq, 1, hidden_size>       
        _extend_input_data_f = input_data.unsqueeze(-2)
        # shape of _repeat_extend_correlated_value_ori: <n_batch, n_seq, n_seq, hidden_size>   
        _repeat_extend_input_data_f = _extend_input_data_f.repeat(1,1,n_seq,1)
        # shape of _extend_input_data: <n_batch, 1, n_seq, hidden_size>   
        _extend_input_data_b = input_data.unsqueeze(1)
        # shape of _repeat_extend_input_data: <n_batch, n_seq, n_seq, hidden_size>   
        _repeat_extend_input_data_b = _extend_input_data_b.repeat(1,n_seq,1,1)
        # shape of _concate_value: <n_batch, n_seq, n_seq, 2 * hidden_size>           
        _concate_value = torch.cat([_repeat_extend_input_data_f, _repeat_extend_input_data_b], dim = -1)        
        # shape of _correlat_value: <n_batch, n_seq, n_seq> 
        _correlat_value = self.activate_func(self.attention_map_func(_concate_value.reshape(-1, 2 * hidden_size)))
        _correlat_value = self.correlated_value_ori_func(_correlat_value).reshape(n_batch, n_seq, n_seq)
        # shape of attention_value_format: <n_batch, n_seq, n_seq>       
        attention_value_format = torch.exp(_correlat_value)
        # shape of ensemble flag format: <1, n_seq, n_seq> 
        # if n_seq = 3, ensemble_flag_format can get below flag data
        #  [[[ 0  0  0 
        #      1  0  0
        #      1  1  0 ]]]
        ensemble_flag_format = torch.triu(torch.ones([n_seq, n_seq]), diagonal = 1).permute(1, 0).unsqueeze(0).to(self.device)
        # shape of accumulate_attention_value: <n_batch, n_seq, 1>
        accumulate_attention_value = torch.sum(attention_value_format * ensemble_flag_format, -1).unsqueeze(-1) + 1e-10
        # shape of each_attention_value: <n_batch, n_seq, n_seq>
        each_attention_value = attention_value_format * ensemble_flag_format
        # shape of attention_weight_format: <n_batch, n_seq, n_seq>
        attention_weight_format = each_attention_value/accumulate_attention_value
        # shape of _extend_attention_weight_format: <n_batch, n_seq, n_seq, 1>
        _extend_attention_weight_format = attention_weight_format.unsqueeze(-1)
        # shape of _extend_input_data: <n_batch, 1, n_seq, hidden_size>
        _extend_input_data = input_data.unsqueeze(1)
        # shape of _weighted_input_data: <n_batch, n_seq, n_seq, hidden_size>
        _weighted_input_data = _extend_attention_weight_format * _extend_input_data
        # shape of weighted_output: <n_batch, n_seq, hidden_size>
        weighted_output = torch.sum(_weighted_input_data, 2)
        return weighted_output

class BuildModel(nn.Module):
    def __init__(self, 
                 input_size = None,
                 embed_size = 16,
                 hidden_size = 8,
                 output_size = 10,
                 bias = True,
                 dropout = 0.5,
                 batch_first = True,
                 label_size = 1,
                 attention_type = 'location_based',
                 attention_dim = 8,
                 device = None):
        super(BuildModel, self).__init__()
        assert input_size != None and isinstance(input_size, int), 'fill in correct input_size' 
 
        self.input_size = input_size        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.label_size = label_size
				
        self.embed_func = nn.Linear(self.input_size, self.embed_size)
        self.rnn_model = nn.GRU(input_size = embed_size,
                                 hidden_size = hidden_size,
                                 bias = bias,
                                 dropout = dropout,
                                 bidirectional = True,
                                 batch_first = batch_first)
        if attention_type == 'location_based':
            self.attention_func = LocationAttention(2*hidden_size, device)
        elif attention_type == 'general':
            self.attention_func = GeneralAttention(2*hidden_size, device)
        elif attention_type == 'concatenation_based':
            self.attention_func = ConcatenationAttention(2*hidden_size, attention_dim = attention_dim, device = device)
        else:
            raise Exception('fill in correct attention_type, [location_based, general, concatenation_based]')
        self.output_func = nn.Linear(4 * hidden_size, self.output_size) 
        self.output_activate = nn.Tanh()
        self.predict_func = nn.Linear(self.output_size, self.label_size)
            
    def forward(self, input_data):
        
        """
        
        Parameters
        
        ----------
        input_data = {
                      'X': shape (batchsize, n_timestep, n_featdim)
                      'M': shape (batchsize, n_timestep)
                      'cur_M': shape (batchsize, n_timestep)
                      'T': shape (batchsize, n_timestep)
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

        batchsize, n_timestep, n_orifeatdim = X.shape
        _ori_X = X.view(-1, n_orifeatdim)
        _embed_X = self.embed_func(_ori_X)
        _embed_X = _embed_X.reshape(batchsize, n_timestep, self.embed_size)
        _embed_F, _ = self.rnn_model(_embed_X)
        _embed_F_w = self.attention_func(_embed_F)
        _mix_F = torch.cat([_embed_F, _embed_F_w], dim = -1)
        _mix_F_reshape = _mix_F.view(-1, 4 * self.hidden_size)
        outputs = self.output_activate(self.output_func(_mix_F_reshape)).reshape(batchsize, n_timestep, self.output_size)
        n_batchsize, n_timestep, output_size = outputs.shape

        all_output = self.predict_func(outputs.reshape(n_batchsize*n_timestep, output_size)).\
                         reshape(n_batchsize, n_timestep, self.label_size)

        if mask is not None and cur_mask is not None:
            all_output *= mask.unsqueeze(-1)
            cur_output = (all_output * cur_mask.unsqueeze(-1)).sum(dim=1)
        else:
            cur_output = all_output.sum(1)

        # return all_output, cur_output
        # TODO: all_output is the per visit label prediction, currently we only support patient-level prediction.
        return cur_output

    def _create_mask_from_visit_length(self, v_len):
        mask = torch.zeros(len(v_len),max(v_len))
        cur_mask = torch.zeros(len(v_len),max(v_len))

        for i in range(len(v_len)):
            mask[i, :v_len[i]] = 1
            cur_mask[i, v_len[i]-1] = 1
        return mask, cur_mask

class Dipole(SequenceIndivBase):
    '''
    Implement Dipole for longitudinal patient records predictive modeling [1]_.
    
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
        - If multiclass/multilabel classification, output_dim=n_class
        - If regression, output_dim=1.

    max_visit: int
        The maximum number of visits for input event codes.

    attention_type: {'general', 'concatenation_based', 'location_based'}

        Apply attention mechnism to derive a context vector that captures relevant information to 
        help predict target.

        - 'location_based': Location-based Attention. Alocation-based attention function is to calculate the weights solely from hidden state 
        
        - 'general': General Attention. An easy way to capture the relationship between two hidden states
        
        - 'concatenation_based': Concatenation-based Attention. Via concatenating two hidden states, then use multi-layer perceptron(MLP) to calculatethe contextvector

    attention_dim: int
        It is the latent dimensionality used for attention weight computing just for for concatenation_based attention mechnism 

    emb_size: int
        Embedding size for encoding input event codes.

    hidden_size : int, optional (default = 8)
        The number of features of the hidden state h
 
    hidden_output_size : int, optional (default = 8)
        The number of mix features

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
    
    Notes
    -----
    ..  [1] Ma, F., Chitta, R., Zhou, J., You, Q., Sun, T., & Gao, J. (2017, August). Dipole: Diagnosis prediction in healthcare via attention-based bidirectional recurrent neural networks. In Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining (pp. 1903-1911).
    '''
    def __init__(self,
        vocab_size,
        orders,
        mode,
        output_dim=None,
        max_visit=None,
        attention_type='location_based',
        attention_dim=8,
        emb_size=16,
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
        super().__init__(experiment_id, mode=mode, output_dim=output_dim)
        self.config = {
            'mode':self.mode,
            'vocab_size':vocab_size,
            'max_visit':max_visit,
            'attention_type':attention_type,
            'attention_dim':attention_dim,
            'emb_size':emb_size,
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

    def fit(self, train_data, valid_data):
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

    def _build_model(self):
        self.model = BuildModel(
            input_size=self.config['total_vocab_size'],
            embed_size=self.config['emb_size'],
            hidden_size=self.config['hidden_size'],
            output_size=self.config['hidden_output_size'],
            dropout=self.config['dropout'],
            label_size=self.config['output_dim'],
            attention_type=self.config['attention_type'],
            attention_dim=self.config['attention_dim'],
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