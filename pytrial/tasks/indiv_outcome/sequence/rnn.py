import pdb

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np

from pytrial.data.patient_data import SeqPatientCollator
from pytrial.utils.check import (
    check_checkpoint_file, check_model_dir, check_model_config_file, make_dir_if_not_exist
)
from .base import SequenceIndivBase, InputEventEmbedding, RNNModel
from ..data import SequencePatient
from ..losses import XentLoss, BinaryXentLoss, MSELoss, MultilabelBinaryXentLoss
from ..trainer import IndivSeqTrainer


class BuildModel(nn.Module):
    def __init__(self,
        rnn_type,
        emb_size,
        bidirectional,
        vocab_size,
        orders,
        n_rnn_layer,
        output_dim,
        **kwargs,
        ) -> None:
        super().__init__()
        if not isinstance(vocab_size, list): vocab_size = [vocab_size]
        self.rnn = RNNModel(
            rnn_type=rnn_type,
            emb_size=emb_size,
            num_layer=n_rnn_layer,
            bidirectional=bidirectional,
            )
        self.emb_size = emb_size
        self.vocab_size = vocab_size
        self.orders = orders
        self.total_vocab_size = sum(vocab_size)
        self.embedding_matrix = nn.Linear(self.total_vocab_size, emb_size, bias=False)
        if bidirectional:
            self.predictor = nn.Linear(2*emb_size, output_dim)
        else:
            self.predictor = nn.Linear(emb_size, output_dim)

    def forward(self, inputs):
        v = inputs['v']
        v_len = inputs['v_lengths']
        visit_emb = self.embedding_matrix(v) # bs, num_visit, emb_size
        packed_input = pack_padded_sequence(visit_emb, v_len, batch_first=True, enforce_sorted=False)
        packed_output = self.rnn(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        out_forward = output[range(len(output)), [l - 1 for l in v_len], :self.emb_size]
        out_reverse = output[:, 0, self.emb_size:]
        out_combined = torch.cat((out_forward, out_reverse), 1) # bs, emb_size or emb_size*2(bidirectional)
        return self.predictor(out_combined)

class RNN(SequenceIndivBase):
    '''
    Implement an RNN based model for longitudinal patient records predictive modeling.

    Parameters
    ----------
    vocab_size: list[int]
        A list of vocabulary size for different types of events, e.g., for diagnosis, procedure, medication.

    orders: list[str]
        A list of orders when treating inputs events. Should have the same shape of `vocab_size`.

    output_dim: int
        Output dimension of the model.

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
        Number of RNN layers for encoding historical events.

    rnn_type: str
        Pick RNN types in ['rnn','lstm','gru']

    bidirectional: bool
        If True, it encodes historical events in bi-directional manner.
    
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
        The model device.

    experiment_id: str
        The prefix when saving the checkpoints during the training.
    '''
    def __init__(self,
        vocab_size,
        orders,
        mode,
        output_dim=None,
        max_visit=20,
        emb_size=64,
        n_rnn_layer=2,
        rnn_type='lstm',
        bidirectional=False,
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
            'n_rnn_layer':n_rnn_layer,
            'rnn_type':rnn_type,
            'bidirectional':bidirectional,
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
        test_data: SequencePatientBase
            A `SequencePatientBase` contains patient records where 'v' corresponds to 
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

    def _build_model(self):
        self.model = BuildModel(
            rnn_type=self.config['rnn_type'],
            emb_size=self.config['emb_size'],
            max_visit=self.config['max_visit'],
            n_rnn_layer=self.config['n_rnn_layer'],
            bidirectional=self.config['bidirectional'],
            vocab_size=self.config['vocab_size'],
            orders=self.config['orders'],
            output_dim=self.config['output_dim'],
            )
        self.model.to(self.device)

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

    def _prepare_input(self, data):
        '''
        Prepare inputs for sequential patient record predictive models.

        Parameters
        ----------
        data: dict[list]
            A batch of patient records.

        '''
        visits = data['v']

        if 'x' in data:
            feature = data['x']
            if not isinstance(feature, torch.Tensor): feature = torch.tensor(feature)
            feature = feature.to(self.device)
        else:
            feature = None

        inputs = {
            'v':{},
            'v_lengths':[],
            'x':feature, # baseline feature
            }
        
        v_lengths = [len(visits[self.config['orders'][0]][idx][:self.config['max_visit']]) for idx in range(len(visits[self.config['orders'][0]]))]
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
