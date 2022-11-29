import pdb

import torch
from torch import nn

from pytrial.utils.check import (
    check_checkpoint_file, check_model_config_file, make_dir_if_not_exist
)
from .base import SequenceIndivBase
from ..trainer import IndivSeqTrainer


class RETAINLayer(nn.Module):
    """The separate callable RETAIN layer.
    Args:
        input_size: the embedding size of the input
        output_size: the embedding size of the output
        num_layers: the number of layers in the RNN
        dropout: dropout rate
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.5,
        ):
        super(RETAINLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        self.alpha_gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.beta_gru = nn.GRU(input_size, hidden_size, batch_first=True)

        self.alpha_li = nn.Linear(hidden_size, 1)
        self.beta_li = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs):
        """Using the sum of the embedding as the output of the transformer
        Args:
            x: [batch size, seq len, input_size]
            mask: [batch size, seq len]
        Returns:
            outputs [batch size, seq len, hidden_size]
        """
        x = inputs['x']
        mask = inputs['mask']

        # rnn will only apply dropout between layers
        x = self.dropout_layer(x)

        g, _ = self.alpha_gru(x)  # (patient, seq_len, hidden_size)
        h, _ = self.beta_gru(x)  # (patient, seq_len, hidden_size)

        # TOFIX: mask out the visit (by adding a large negative number 1e10)
        # however, it does not work better than not mask out
        attn_g = self.alpha_li(g)

        if mask is not None:
            attn_g = self._apply_mask_to_attention(attn_g, mask)

        attn_g = torch.softmax(attn_g, dim=1)  # (patient, seq_len, 1)
        attn_h = torch.tanh(self.beta_li(h))  # (patient, seq_len, hidden_size)
        c = attn_g * attn_h * x  # (patient, seq_len, hidden_size)
        # c = torch.sum(c, dim=1)  # (patient, hidden_size)
        return {'x':c, 'mask':mask}
    
    def _apply_mask_to_attention(self, attn_g, mask):
        mask_offset = torch.zeros(mask.shape, device=mask.device)
        mask_offset[mask==0] = -1e9
        mask_offset[mask==1] = 0
        attn_g += mask_offset.unsqueeze(-1)
        return attn_g


class BuildModel(nn.Module):
    def __init__(self,
        emb_size,
        hidden_dim,
        vocab_size,
        n_rnn_layer,
        output_dim,
        **kwargs,
        ) -> None:
        super().__init__()
        if not isinstance(vocab_size, list): vocab_size = [vocab_size]
        self.emb_size = emb_size
        self.vocab_size = vocab_size
        self.total_vocab_size = sum(vocab_size)
        self.embedding_matrix = nn.Linear(self.total_vocab_size, emb_size, bias=False)
        rnn_list = []
        for i in range(n_rnn_layer):
            if i == 0:
                rnn = RETAINLayer(
                        input_size=emb_size,
                        hidden_size=hidden_dim,
                        dropout=0.5,
                    )
            
            else:
                rnn = RETAINLayer(
                    input_size=hidden_dim,
                    hidden_size=hidden_dim,
                    dropout=0.5,
                )
            rnn_list.append(rnn)
        self.rnn = nn.Sequential(*rnn_list)
        self.predictor = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        v = inputs['v']
        visit_emb = self.embedding_matrix(v) # bs, num_visit, emb_size

        if 'v_lengths' in inputs:
            v_len = inputs['v_lengths']
            mask = self._create_mask_from_visit_length(v_len=v_len)
            mask = mask.to(visit_emb.device)
        else:
            mask = None

        output = self.rnn({'x':visit_emb, 'mask':mask}) # bs, seq_len, hidden_size
        output = output['x']
        output = torch.sum(output, dim=1) # bs, emd_dim
        return self.predictor(output)

    def _create_mask_from_visit_length(self, v_len):
        mask = torch.zeros(len(v_len),max(v_len))
        for i in range(len(v_len)):
            mask[i, :v_len[i]] = 1
        return mask


class RETAIN(SequenceIndivBase):
    '''
    Implement RETAIN for longitudinal patient records predictive modeling [1]_.

    Parameters
    ----------
    vocab_size: list[int]
        A list of vocabulary size for different types of events, e.g., for diagnosis, procedure, medication.

    orders: list[str]
        A list of orders when treating inputs events. Should have the same shape of `vocab_size`.

    output_dim: int
        The dimension of the output.

        - If binary classification, output_dim=1;
        - If multiclass/multilabel classification, output_dim=n_class
        - If regression, output_dim=1.

    mode: str
        Prediction traget in ['binary','multiclass','multilabel','regression'].

    max_visit: int
        The maximum number of visits for input event codes.

    emb_size: int
        Embedding size for encoding input event codes.

    n_rnn_layer: int
        Number of RETAIN layers for encoding historical events.
    
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
    .. [1] Choi, E., Bahadori, M. T., Sun, J., Kulas, J., Schuetz, A., & Stewart, W. (2016). Retain: An interpretable predictive model for healthcare using reverse time attention mechanism. Advances in neural information processing systems, 29.
    '''
    def __init__(self,
        vocab_size,
        orders,
        mode,
        output_dim=None,
        max_visit=None,
        emb_size=64,
        n_rnn_layer=2,
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
            'emb_size':emb_size,
            'hidden_dim':emb_size,
            'n_rnn_layer':n_rnn_layer,
            'output_dim':self.output_dim,
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
            emb_size=self.config['emb_size'],
            hidden_dim=self.config['hidden_dim'],
            n_rnn_layer=self.config['n_rnn_layer'],
            vocab_size=self.config['vocab_size'],
            output_dim=self.config['output_dim'],
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