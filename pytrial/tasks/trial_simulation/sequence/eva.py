import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import numpy as np
from collections import defaultdict

from pytrial.data.patient_data import SequencePatientBase, SeqPatientCollator
from pytrial.utils.check import (
    check_checkpoint_file, check_model_dir, check_model_config_file, make_dir_if_not_exist
)
from .base import SequenceSimulationBase
from ..losses import MultilabelBinaryXentLossWithKLDivergence
from ..trainer import SeqSimEvaTrainer
from ..data import SequencePatient

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super(CausalConv1d, self).__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.pad, dilation=dilation, **kwargs)

    def forward(self, input):
        return self.conv(input)[:,:,:-self.conv.padding[0]]

def connector(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return eps * std + mu

class Encoder(nn.Module):
    def __init__(self, emb_size, latent_dim, n_rnn_layer, total_vocab_size):
        super(Encoder, self).__init__()
        self.hidden_dim = emb_size
        self.embedding_matrix = nn.Linear(total_vocab_size, emb_size, bias=False)
        self.lstm = nn.LSTM(input_size=emb_size,
                            hidden_size=emb_size,
                            num_layers=n_rnn_layer,
                            bidirectional=True,
                            batch_first=True)
        self.latent_encoder = nn.Linear(2*emb_size, 2*latent_dim)

    def forward(self, input, lengths):
        visit_emb = self.embedding_matrix(input)
        packed_input = pack_padded_sequence(visit_emb, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        out_forward = output[range(len(output)), [l - 1 for l in lengths], :self.hidden_dim]
        out_reverse = output[:, 0, self.hidden_dim:]
        out_combined = torch.cat((out_forward, out_reverse), 1)
        mean_logvar = self.latent_encoder(out_combined)
        return mean_logvar

class Decoder(nn.Module):
    def __init__(self, emb_size, latent_dim, total_vocab_size, max_visit):
        super(Decoder, self).__init__()
        self.max_visit = max_visit
        self.deconv_list = nn.ModuleList([nn.ConvTranspose1d(latent_dim if i == 0 else emb_size, emb_size, 3, stride=3) for i in range(int(np.ceil(np.power(max_visit, 1/3))))])
        self.causal_conv1 = CausalConv1d(emb_size, emb_size, 5, dilation=2)
        self.causal_conv2 = CausalConv1d(emb_size, 2*emb_size, 5, dilation=2)
        self.causal_conv3 = CausalConv1d(2*emb_size, total_vocab_size, 5, dilation=2)

    def forward(self, input):
        out = input.unsqueeze(2)
        for deconv in self.deconv_list:
          out = deconv(out)
        out = out[:,:,:self.max_visit]
        out = self.causal_conv1(out)
        out = self.causal_conv2(out)
        out = self.causal_conv3(out)
        out = out.transpose(1, 2)
        return out
      
class BuildModel(nn.Module):
    def __init__(self,
        max_visit,
        emb_size,
        latent_dim,
        n_rnn_layer,
        total_vocab_size,
        device,
        **kwargs,
        ) -> None:
        super().__init__()
        
        self.latent_dim = latent_dim
        self.device = device

        self.encoder_module = Encoder(
            emb_size=emb_size, 
            latent_dim=latent_dim, 
            n_rnn_layer=n_rnn_layer, 
            total_vocab_size=total_vocab_size,
        )

        self.decoder_module = Decoder(
            emb_size=emb_size, 
            latent_dim=latent_dim, 
            total_vocab_size=total_vocab_size, 
            max_visit=max_visit
        )
      
    def forward(self, inputs):
        x, input_lengths = inputs['v'], inputs['v_lengths']
        mean_logvar = self.encoder_module(x, input_lengths)
        mu = mean_logvar[:,:self.latent_dim]
        log_var = mean_logvar[:,self.latent_dim:]
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        decoder_inputs = connector(mu, log_var)
        code_logits = self.decoder_module(decoder_inputs)
        for idx in range(len(inputs['v_lengths'])):
            code_logits[idx,inputs['v_lengths'][idx]:] = 0
        return code_logits[:,:inputs['v'].size(1)], inputs['v'], kl_loss
      
    def sample(self, n_samples):
        decoder_inputs = torch.randn((n_samples, self.latent_dim)).to(self.device)
        code_logits = self.decoder_module(decoder_inputs)
        sig = nn.Sigmoid()
        code_probs = sig(code_logits)
        patient_records = torch.bernoulli(code_probs)
        return patient_records
    
    def sample_from(self, x, input_lengths, n_per_sample):
        mean_logvar = self.encoder_module(x, input_lengths)
        mu = mean_logvar[:,:self.latent_dim]
        log_var = mean_logvar[:,self.latent_dim:]
        code_probs = []
        sig = nn.Sigmoid()
        for _ in range(n_per_sample):
            decoder_inputs = connector(mu, log_var)
            code_logits = self.decoder_module(decoder_inputs)
            code_probs.append(sig(code_logits))
        code_probs = torch.cat(code_probs)
        patient_records = torch.bernoulli(code_probs)
        return patient_records

class EVA(SequenceSimulationBase):
    '''
    Implement a VAE based model for longitudinal patient records simulation [1]_.
    
    Parameters
    ----------
    vocab_size: list[int]
        A list of vocabulary size for different types of events, e.g., for diagnosis, procedure, medication.

    order: list[str]
        The order of event types in each visits, e.g., ``['diag', 'prod', 'med']``.
        Visit = [diag_events, prod_events, med_events], each event is a list of codes.

    max_visit: int
        Maximum number of visits.

    emb_size: int
        Embedding size for encoding input event codes.
        
    latent_dim: int
        Size of final latent dimension between the encoder and decoder
        
    n_rnn_layer: int
        Number of RNN layers for encoding historical events.

    learning_rate: float
        Learning rate for optimization based on SGD. Use torch.optim.Adam by default.

    batch_size: int
        Batch size when doing SGD optimization.

    epochs: int
        Maximum number of iterations taken for the solvers to converge.

    num_worker: int
        Number of workers used to do dataloading during training.

    Notes
    -----
    .. [1] Biswal, S., et al. (2020, December). EVA: Generating Longitudinal Electronic Health Records Using Conditional Variational Autoencoders.
    '''
    def __init__(self,
        vocab_size,
        order,
        max_visit=20,
        emb_size=64,
        latent_dim=32,
        n_rnn_layer=2,
        learning_rate=1e-3,
        batch_size=64,
        epochs=20,
        num_worker=0,
        device='cpu',# 'cuda:0',
        experiment_id='trial_simulation.sequence.eva',
        ):
        super().__init__(experiment_id)
        self.config = {
            'vocab_size':vocab_size,
            'max_visit':max_visit,
            'emb_size':emb_size,
            'latent_dim':latent_dim,
            'n_rnn_layer':n_rnn_layer,
            'device':device,
            'learning_rate':learning_rate,
            'batch_size':batch_size,
            'epochs':epochs,
            'num_worker':num_worker,
            'orders':order,
            }
        self.config['total_vocab_size'] = sum(vocab_size)
        self.device = device
        self._build_model()
        
    def _build_model(self):
        self.model = BuildModel(
            max_visit=self.config['max_visit'],
            emb_size=self.config['emb_size'],
            latent_dim=self.config['latent_dim'],
            n_rnn_layer=self.config['n_rnn_layer'],
            total_vocab_size=self.config['total_vocab_size'],
            orders=self.config['orders'],
            device=self.device
            )
        self.model = self.model.to(self.device)
        
    def fit(self, train_data):
        '''
        Train model with sequential patient records.

        Parameters
        ----------
        train_data: SequencePatientBase
            A `SequencePatientBase` contains patient records where 'v' corresponds to 
            visit sequence of different events.
        '''
        self._input_data_check(train_data)
        self._fit_model(train_data)
    
    def predict(self, n, return_tensor=False):
        '''
        Generate synthetic records

        Parameters
        ----------
        n: int
            How many samples in total will be generated.
        
        return_tensor: bool
            - If `True`, return output generated records in tensor format (n, n_visit, n_event), good for later predictive modeling.
            - If `False, return records in `SequencePatient` format.
        '''
        assert isinstance(n, int), 'Input `n` should be integer.'
        outputs = self._predict(n)
        if not return_tensor:
            outputs = self._translate_sparse_visits_to_dense(outputs)
        return outputs

    def save_model(self, output_dir):
        '''
        Save the learned simulation model to the disk.

        Parameters
        ----------
        output_dir: str or None
            The dir to save the learned model.
            If set None, will save model to `self.checkout_dir`.
        '''
        if output_dir is not None:
            make_dir_if_not_exist(output_dir)
        else:
            output_dir = self.checkout_dir
            
        self._save_config(self.config, output_dir=output_dir)
        self._save_checkpoint({
                  'encoder': self.model.encoder_module.state_dict(),
                  'decoder': self.model.decoder_module.state_dict()
              }, output_dir=output_dir)

    def load_model(self, checkpoint):
        '''
        Load model and the pre-encoded trial embeddings from the given
        checkpoint dir.
        
        Parameters
        ----------
        checkpoint: str
            The input dir that stores the pretrained model.

            - If a directory, the only checkpoint file `*.pth.tar` will be loaded.
            - If a filepath, will load from this file.
        '''
        checkpoint_filename = check_checkpoint_file(checkpoint)
        config_filename = check_model_config_file(checkpoint)
        state_dict = torch.load(checkpoint_filename, map_location=self.config['device'])
        if config_filename is not None:
            config = self._load_config(config_filename)
            self.config.update(config)
        self.model.encoder_module.load_state_dict(state_dict['encoder'])
        self.model.decoder_module.load_state_dict(state_dict['decoder'])
    
    def get_train_dataloader(self, train_data):
        dataloader = DataLoader(train_data,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_worker'],
            pin_memory=True,
            shuffle=True,
            collate_fn=SeqPatientCollator(
                config={
                    'visit_mode':train_data.metadata['visit']['mode'],
                    'label_mode':train_data.metadata['label']['mode'],
                    }
                ),
            )
        return dataloader
    
    def get_test_dataloader(self, train_data):
        dataloader = DataLoader(train_data,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_worker'],
            pin_memory=False,
            shuffle=False,
            collate_fn=SeqPatientCollator(
                config={
                    'visit_mode':train_data.metadata['visit']['mode'],
                    'label_mode':train_data.metadata['label']['mode'],
                    }
                ),
            )
        return dataloader
    
    def _build_loss_model(self):
        return MultilabelBinaryXentLossWithKLDivergence(self.model)

    def _fit_model(self, train_data):
        train_dataloader = self.get_train_dataloader(train_data)
        loss_model = self._build_loss_model()
        train_objectives = [(train_dataloader, loss_model)]
        trainer = SeqSimEvaTrainer(
            model=self,
            train_objectives=train_objectives
            )
        trainer.train(**self.config)

    @torch.no_grad()
    def _predict(self, n):
        return self.model.sample(n)
    
    @torch.no_grad()
    def _predict_on_dataloader(self, test_dataloader, n, n_per_sample):
        data_iterator = iter(test_dataloader)
        outputs = []
        for data in data_iterator:
            outputs.append(self.model.sample_from(data['v'], data['v_lengths'], n_per_sample))
            
        return torch.cat(outputs)[:n]

    def _translate_sparse_visits_to_dense(self, visits):
        def _map_func(x):
            res = np.where(x > 0)[0].tolist()
            return [0] if len(res) == 0 else res # pad if nothing happened

        outputs = defaultdict(list)
        for batchv in visits:
            voc_offset = 0
            for i, o in enumerate(self.config['orders']):
                voc_size = self.config['vocab_size'][i]
                visit = batchv[...,voc_offset:voc_offset+voc_size]
                visit = visit.cpu().numpy()
                res = list(map(_map_func, visit))
                outputs[o].append(res)
                voc_offset += voc_size

        n_total = len(outputs[o])
        sample_list = []        
        for i in range(n_total):
            sample = []
            for numv in range(len(outputs[o][i])):
                visit = []
                for o in self.config['orders']:
                    visit.append(outputs[o][i][numv])
                sample.append(visit)
            sample_list.append(sample)

        # create seqpatient data
        return SequencePatient(
            data={'v':sample_list},
            metadata={
                'visit':{'mode':'dense','order':self.config['orders']},
                }
            )

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

    def _pad_multiple_tensor_visits(self, visits):
        new_list = []
        for v in visits:
            new_list.extend([torch.tensor(x).squeeze(0) for x in np.array_split(v, len(v))])
        return pad_sequence(new_list, batch_first=True)
    
    def _prepare_input(self, data):
        '''
        Prepare inputs for sequence simulation models.

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
        
        v_lengths = [len(visits[self.config['orders'][0]][idx][:self.config['max_visit']]) for idx in range(len(visits[self.config['orders'][0]]))]
        inputs['v_lengths'] = v_lengths

        v = torch.zeros(len(v_lengths), max(v_lengths), self.config['total_vocab_size'])
        for idx in range(len(v_lengths)):
            v[idx,:v_lengths[idx]] = torch.tensor(self._translate_dense_visits_to_sparse({k: visits[k][idx][:self.config['max_visit']] for k in visits}))
        v = v.to(self.device)
        inputs['v'] = v

        return inputs
    
    def _input_data_check(self, inputs):
        assert isinstance(inputs, SequencePatientBase), f'`trial_simulation.sequence` models require input training data in `SequencePatientBase`, find {type(inputs)} instead.'
        