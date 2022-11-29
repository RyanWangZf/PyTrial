import pdb

import torch
from torch import nn

from pytrial.utils.check import (
    check_checkpoint_file, check_model_config_file, make_dir_if_not_exist
)
from .base import SequenceIndivBase
from ..trainer import IndivSeqTrainer


class BuildModel(nn.Module):
    def __init__(self, 
        input_dim = None, 
        hidden_dim = 384, 
        conv_size = 10, 
        levels = 3, 
        dropconnect = 0.3, 
        dropout = 0.3, 
        dropres = 0.3,
        label_size = None,
        device = None
        ):
        super(BuildModel, self).__init__()
        assert hidden_dim % levels == 0
        self.dropout = dropout
        self.dropconnect = dropconnect
        self.dropres = dropres
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv_dim = hidden_dim
        self.conv_size = conv_size
        self.output_dim = label_size
        self.levels = levels
        self.chunk_size = hidden_dim // levels
        self.device = device

        self.kernel = nn.Linear(int(input_dim+1), int(hidden_dim*4+levels*2))
        nn.init.xavier_uniform_(self.kernel.weight)
        nn.init.zeros_(self.kernel.bias)
        self.recurrent_kernel = nn.Linear(int(hidden_dim+1), int(hidden_dim*4+levels*2))
        nn.init.orthogonal_(self.recurrent_kernel.weight)
        nn.init.zeros_(self.recurrent_kernel.bias)

        self.nn_scale = nn.Linear(int(hidden_dim), int(hidden_dim // 6))
        self.nn_rescale = nn.Linear(int(hidden_dim // 6), int(hidden_dim))
        self.nn_conv = nn.Conv1d(int(hidden_dim), int(self.conv_dim), int(conv_size), 1)
        self.nn_output = nn.Linear(int(self.conv_dim), int(self.output_dim))

        if self.dropconnect:
            self.nn_dropconnect = nn.Dropout(p=dropconnect)
            self.nn_dropconnect_r = nn.Dropout(p=dropconnect)
        if self.dropout:
            self.nn_dropout = nn.Dropout(p=dropout)
            self.nn_dropres = nn.Dropout(p=dropres)

    def cumax(self, x, mode='l2r'):

        if mode == 'l2r':
            x = torch.softmax(x, dim=-1)
            x = torch.cumsum(x, dim=-1)
            return x
        elif mode == 'r2l':
            x = torch.flip(x, [-1])
            x = torch.softmax(x, dim=-1)
            x = torch.cumsum(x, dim=-1)
            return torch.flip(x, [-1])
        else:
            return x

    def step(self, inputs, c_last, h_last, interval):

        x_in = inputs
        # Integrate inter-visit time intervals
        interval = interval.unsqueeze(-1)
        x_out1 = self.kernel(torch.cat((x_in, interval), dim=-1))
        x_out2 = self.recurrent_kernel(torch.cat((h_last, interval), dim=-1))

        if self.dropconnect:
            x_out1 = self.nn_dropconnect(x_out1)
            x_out2 = self.nn_dropconnect_r(x_out2)
        x_out = x_out1 + x_out2
        f_master_gate = self.cumax(x_out[:, :self.levels], 'l2r')
        f_master_gate = f_master_gate.unsqueeze(2)
        i_master_gate = self.cumax(x_out[:, self.levels:self.levels*2], 'r2l')
        i_master_gate = i_master_gate.unsqueeze(2)
        x_out = x_out[:, self.levels*2:]
        x_out = x_out.reshape(-1, self.levels*4, self.chunk_size)
        f_gate = torch.sigmoid(x_out[:, :self.levels])
        i_gate = torch.sigmoid(x_out[:, self.levels:self.levels*2])
        o_gate = torch.sigmoid(x_out[:, self.levels*2:self.levels*3])
        c_in = torch.tanh(x_out[:, self.levels*3:])
        c_last = c_last.reshape(-1, self.levels, self.chunk_size)
        overlap = f_master_gate * i_master_gate
        c_out = overlap * (f_gate * c_last + i_gate * c_in) + (f_master_gate - overlap) * c_last + (i_master_gate - overlap) * c_in
        h_out = o_gate * torch.tanh(c_out)
        c_out = c_out.reshape(-1, self.hidden_dim)
        h_out = h_out.reshape(-1, self.hidden_dim)
        out = torch.cat([h_out, f_master_gate[..., 0], i_master_gate[..., 0]], 1)
        return out, c_out, h_out

    def forward(self, input_data):
        
        """
        
        Parameters
        
        ----------
        input_data = {
                      'X': shape (batchsize, n_timestep, n_featdim)
                      'M': shape (batchsize, n_timestep)
                      'cur_M': shape (batchsize, n_timestep)
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

        if 'timestamp' not in input_data:
            # if no timestamp is provided, we assume the time interval between visits is 1.
            T = torch.ones_like(X[:, :, 0])
            T[:,0] = 0
            T = T.to(X.device)
        else:
            T = input_data['timestamp']
            # differentiate between timestamp
            T = T[:,1:] - T[:,:-1]
            T[:,0] = 0

        if 'v_lengths' in input_data:
            v_len = input_data['v_lengths']
            mask, cur_mask = self._create_mask_from_visit_length(v_len=v_len)
            mask = mask.to(X.device)
            cur_mask = cur_mask.to(X.device)
        else:
            mask = cur_mask = None

        batch_size, time_step, feature_dim = X.size()
        c_out = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        h_out = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        tmp_h = torch.zeros_like(h_out, dtype=torch.float32).view(-1).repeat(self.conv_size).view(self.conv_size, batch_size, self.hidden_dim).to(self.device)
        tmp_dis = torch.zeros((self.conv_size, batch_size)).to(self.device)
        h = []
        origin_h = []
        distance = []

        for t in range(time_step):
            out, c_out, h_out = self.step(X[:, t, :], c_out, h_out, T[:, t])
            cur_distance = 1 - torch.mean(out[..., self.hidden_dim:self.hidden_dim+self.levels], -1)
            cur_distance_in = torch.mean(out[..., self.hidden_dim+self.levels:], -1)
            origin_h.append(out[..., :self.hidden_dim])
            tmp_h = torch.cat((tmp_h[1:], out[..., :self.hidden_dim].unsqueeze(0)), 0)
            tmp_dis = torch.cat((tmp_dis[1:], cur_distance.unsqueeze(0)), 0)
            distance.append(cur_distance)
            #Re-weighted convolution operation
            local_dis = tmp_dis.permute(1, 0)
            local_dis = torch.cumsum(local_dis, dim=1)
            local_dis = torch.softmax(local_dis, dim=1)
            local_h = tmp_h.permute(1, 2, 0)
            local_h = local_h * local_dis.unsqueeze(1)
            #Re-calibrate Progression patterns
            local_theme = torch.mean(local_h, dim=-1)
            local_theme = self.nn_scale(local_theme)
            local_theme = torch.relu(local_theme)
            local_theme = self.nn_rescale(local_theme)
            local_theme = torch.sigmoid(local_theme)
            local_h = self.nn_conv(local_h).squeeze(-1)
            local_h = local_theme * local_h
            h.append(local_h)  

        origin_h = torch.stack(origin_h).permute(1, 0, 2)
        rnn_outputs = torch.stack(h).permute(1, 0, 2)
        if self.dropres > 0.0:
            origin_h = self.nn_dropres(origin_h)
        rnn_outputs = rnn_outputs + origin_h
        rnn_outputs = rnn_outputs.contiguous().view(-1, rnn_outputs.size(-1))
        if self.dropout > 0.0:
            rnn_outputs = self.nn_dropout(rnn_outputs)
        output = self.nn_output(rnn_outputs)
        output = output.contiguous().view(batch_size, time_step, self.output_dim)

        cur_output = (output * cur_mask.unsqueeze(-1)).sum(dim=1)
        all_output = torch.sigmoid(output)

        # TODO: all_output is the per visit label prediction, currently we only support patient-level prediction.
        # return all_output, cur_output
        return cur_output

    def _create_mask_from_visit_length(self, v_len):
        mask = torch.zeros(len(v_len),max(v_len))
        cur_mask = torch.zeros(len(v_len),max(v_len))

        for i in range(len(v_len)):
            mask[i, :v_len[i]] = 1
            cur_mask[i, v_len[i]-1] = 1
        return mask, cur_mask

class StageNet(SequenceIndivBase):
    '''
    Implement StageNet for longitudinal patient records predictive modeling [1]_.

    Parameters
    ----------
    vocab_size: list[int]
        A list of vocabulary size for different types of events, e.g., for diagnosis, procedure, medication.

    orders: list[str]
        A list of orders when treating inputs events. Should have the same shape of `vocab_size`.

    mode: str
        Prediction traget in ['binary','multiclass','multilabel','regression'].

    output_dim: int
        If binary classification, output_dim=1;
        If multiclass/multilabel classification, output_dim=n_class
        If regression, output_dim=1.

    max_visit: int
        The maximum number of visits for input event codes.

    hidden_size : int, optional (default = 8)
        The number of features of the hidden state h.
    
    conv_size : int, optional (default = 10)
        The number of convolution kernels.
    
    levels : int, optional (default = 3)
        The number of levels for the master gate.
    
    dropconnect : float, optional (default = 0.3)
        The dropout rate for the input of the convolutional layer.

    dropout : float, optional (default = 0.3)
        The dropout rate for the output of the RNN layer.
    
    dropres : float, optional (default = 0.3)
        The dropout rate for the residual connection.
    
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
    .. [1] Gao, J., Xiao, C., Wang, Y., Tang, W., Glass, L. M., & Sun, J. (2020, April). Stagenet: Stage-aware neural networks for health risk prediction. 
        In Proceedings of The Web Conference 2020 (pp. 530-540).
    '''
    def __init__(self,
        vocab_size,
        orders,
        mode,
        output_dim=None,
        max_visit=None,
        hidden_size=384,
        conv_size=10,
        levels=3,
        dropconnect=0.3,
        dropout=0.3,
        dropres=0.3,
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
            'hidden_size':hidden_size,
            'output_dim':self.output_dim,
            'dropout':dropout,
            'dropres':dropres,
            'dropconnect':dropconnect,
            'conv_size':conv_size,
            'levels':levels,
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
            input_dim=self.config['total_vocab_size'],
            hidden_dim=self.config['hidden_size'],
            conv_size=self.config['conv_size'],
            levels=self.config['levels'],
            dropout=self.config['dropout'],
            dropres=self.config['dropres'],
            dropconnect=self.config['dropconnect'],
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