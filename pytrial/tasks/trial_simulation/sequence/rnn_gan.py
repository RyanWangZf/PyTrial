from collections import defaultdict
import pdb
import time

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from pytrial.data.patient_data import SequencePatientBase, SeqPatientCollator
from pytrial.utils.check import (
    check_checkpoint_file, check_model_dir, check_model_config_file, make_dir_if_not_exist
)
from .base import SequenceSimulationBase
from .base import InputEventEmbedding, GAN, RNN
from ..losses import GeneratorLoss, DiscriminatorLoss, DiscriminatorLossGP
from ..trainer import SeqSimGANTrainer
from ..data import SequencePatient

class BuildModel(nn.Module):
    def __init__(self,
        rnn_type,
        emb_size,
        bidirectional,
        vocab_size,
        orders,
        n_rnn_layer,
        padding_idx,
        **kwargs,
        ) -> None:
        super().__init__()
        if not isinstance(vocab_size, list): vocab_size = [vocab_size]

        # generator dim size
        # emb_size*2, gendim[0], gendim[2]: default
        # emb_size*4, gendim[0], gendim[1]: directional
        if bidirectional:
            gan_input_emb_size = 2*emb_size
            gen_dims=[gan_input_emb_size, 2*emb_size]
        else:
            gan_input_emb_size = emb_size
            gen_dims=[2*emb_size, 2*emb_size]

        # discriminator dim size
        dis_dims=[2*emb_size]

        if bidirectional:
            gen_dims = [g*2 for g in gen_dims]

        self.gan_module = GAN(
            emb_size=gan_input_emb_size,
            total_vocab_size=sum(vocab_size),
            gen_dims=gen_dims,
            dis_dims=dis_dims,
        )

        self.rnn_module = RNN(
            rnn_type=rnn_type,
            emb_size=emb_size,
            num_layer=n_rnn_layer,
            bidirectional=bidirectional,
        )

        self.vocab_size = vocab_size
        self.orders = orders

        self.embeddings = InputEventEmbedding(orders=orders, vocab_size=vocab_size, emb_size=emb_size, padding_idx=padding_idx)

    def forward(self, inputs, n=1):
        '''
        inputs is a dict
        {
            'v': {'eventA':[],'eventB':[],...,},
            'x': tensor(),
            'y': {'eventA:[], 'eventB':[],...,} # optional
        }
        '''
        embs = self.embeddings(inputs) # [num_visit, num_event, emb_size]
        embs = torch.sum(embs, 1) # [num_visit, emb_size]
        embs = self.rnn_module(embs) # [num_visit, emb_size] or [num_visit, emb_size*2] (bidirectional)
        last_visit_emb = embs[-1] # [emb_size]

        if len(last_visit_emb.shape) == 1:
            last_visit_emb = last_visit_emb.unsqueeze(0)

        if n > 1:
            # generate more than one visits
            last_visit_emb = last_visit_emb.expand(n, -1)

        # infer generator
        z_random = torch.randn(last_visit_emb.size()).to(last_visit_emb.device)
        x_fake = self.gan_module.infer_generator(z_random, last_visit_emb) # 1, vocab_size
        
        # infer discriminator for real records
        if 'y' in inputs:
            # infer discriminator for fake records
            y_fake = self.gan_module.infer_discriminator(x_fake)
            target = self._create_multilabel_target(inputs)
            target = target.to(x_fake.device)
            y_real = self.gan_module.infer_discriminator(target.float())
            return {'x_fake':x_fake, 'y_real':y_real, 'y_fake':y_fake, 'y':target}
        else:
            return {'x_fake':x_fake}
    
    def infer_discriminator(self, x):
        return self.gan_module.infer_discriminator(x)

    def _create_multilabel_target(self, inputs):
        # for one patient one visit only `inputs`
        target = inputs['y']
        target_mo_list = []
        for i,o in enumerate(self.orders):
            target_mo = torch.zeros(self.vocab_size[i], dtype=torch.long)
            target_mo[target[o]] = 1
            target_mo_list.append(target_mo)
        targets = torch.cat(target_mo_list)
        if len(targets.shape) == 1: targets = targets.unsqueeze(0)
        return targets


class RNNGAN(SequenceSimulationBase):
    '''
    Implement an RNN based GAN model for longitudinal patient records simulation. The GAN part was proposed by Choi et al. [1]_.
    
    Parameters
    ----------
    vocab_size: list[int]
        A list of vocabulary size for different types of events, e.g., for diagnosis, procedure, medication.

    order: list[str]
        The order of event types in each visits, e.g., ``['diag', 'prod', 'med']``.
        Visit = [diag_events, prod_events, med_events], each event is a list of codes.

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
    
    padding_idx: int(default=None)
        Set the padding index for input events embedding. If set None, then no
        padding index will be specified.

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

    Notes
    -----
    .. [1] Choi, E., et al. (2017, November). Generating multi-label discrete patient records using generative adversarial networks. In ML4HC (pp. 286-305). PMLR.
    '''
    def __init__(self,
        vocab_size,
        order,
        max_visit=20,
        emb_size=64,
        n_rnn_layer=2,
        rnn_type='lstm',
        bidirectional=False,
        padding_idx=None,
        learning_rate=1e-4,
        weight_decay=1e-4,
        batch_size=64,
        epochs=10,
        num_worker=0,
        device='cuda:0',
        experiment_id='trial_simulation.sequence.rnn_gan',
        ):
        super().__init__(experiment_id)
        self.config = {
            'vocab_size':vocab_size,
            'max_visit':max_visit,
            'emb_size':emb_size,
            'n_rnn_layer':n_rnn_layer,
            'rnn_type':rnn_type,
            'bidirectional':bidirectional,
            'padding_idx':padding_idx,
            'device':device,
            'learning_rate':learning_rate,
            'batch_size':batch_size,
            'weight_decay':weight_decay,
            'epochs':epochs,
            'num_worker':num_worker,
            'orders':order,
            }
        self.config['total_vocab_size'] = sum(vocab_size)
        self.device = device
        self._build_model()
    
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

    def predict(self, test_data, n=None, n_per_sample=None, return_tensor=True):
        '''
        Generate synthetic records based on input real patient seq data.

        Parameters
        ----------
        test_data: SequencePatientBase
            A `SequencePatientBase` contains patient records where 'v' corresponds to 
            visit sequence of different events.

        n: int
            How many samples in total will be generated.

        n_per_sample: int
            How many samples generated based on each indivudals.
        
        return_tensor: bool
            If `True`, return output generated records in tensor format (n, n_visit, n_event), good for later predictive modeling.
            If `False, return records in `SequencePatient` format.
        '''
        if n is not None: assert isinstance(n, int), 'Input `n` should be integer.'
        if n_per_sample is not None: assert isinstance(n_per_sample, int), 'Input `n_per_sample` should be integer.'
        n, n_per_sample = self._compute_n_per_sample(len(test_data), n, n_per_sample)
        test_dataloader = self.get_test_dataloader(test_data)
        outputs = self._predict_on_dataloader(test_dataloader, n, n_per_sample)
        if not return_tensor:
            outputs = self._translate_sparse_visits_to_dense(outputs)
        else:
            # pad all to same shape
            outputs = self._pad_multiple_tensor_visits(outputs)
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
        self._save_checkpoint({'model':self.model}, output_dir=output_dir)

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
        state_dict = torch.load(checkpoint_filename)
        if config_filename is not None:
            config = self._load_config(config_filename)
            self.config.update(config)
        self.model = state_dict['model']
    
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

    def _build_model(self):
        self.model = BuildModel(
            rnn_type=self.config['rnn_type'],
            emb_size=self.config['emb_size'],
            max_visit=self.config['max_visit'],
            n_rnn_layer=self.config['n_rnn_layer'],
            bidirectional=self.config['bidirectional'],
            vocab_size=self.config['vocab_size'],
            orders=self.config['orders'],
            padding_idx=self.config['padding_idx'],
            )
        self.model.to(self.device)
    
    def _build_loss_model(self):
        # update discriminator two times v.s. generator one time
        return [DiscriminatorLoss(self.model), GeneratorLoss(self.model)]

    def _fit_model(self, train_data):
        train_dataloader = self.get_train_dataloader(train_data)
        loss_models = self._build_loss_model()
        train_objectives = [(train_dataloader, loss_model) for loss_model in loss_models]
        trainer = SeqSimGANTrainer(
            model=self,
            train_objectives=train_objectives
            )
        trainer.train(**self.config)

    @torch.no_grad()
    def _predict_on_dataloader(self, test_dataloader, n, n_per_sample):
        data_iterator = iter(test_dataloader)
        total_number = 0
        fake_visit_list = []
        while total_number < n:
            try:
                data = next(data_iterator)
            except:
                data_iterator = iter(test_dataloader)
                data = next(data_iterator)
            
            for idx, _ in enumerate(data['x']):
                num_visit = self._get_num_visit(data, idx)
                if num_visit < 2: # deal with more than one visit only
                    continue
                fake_visits = []
                for vdx in range(1, num_visit):
                    inputs = self._prepare_input(data, idx, vdx)
                    if 'y' in inputs: inputs.pop('y')
                    x_fake = self.model(inputs, n=n_per_sample)['x_fake']
                    fake_visits.append(x_fake.cpu().numpy())
                
                fake_visits = np.stack(fake_visits, 1) # n_sample, n_visit, total_vocab_size
                fake_visits[fake_visits>0.5]=1
                fake_visits[fake_visits<=0.5]=0

                outputs = self._prepare_input(data, idx, 1)
                first_visit = self._translate_dense_visits_to_sparse(outputs['v'])
                first_visit = np.tile(first_visit[None],(len(fake_visits),1,1))
                fake_visits = np.concatenate([first_visit, fake_visits], 1)
                fake_visit_list.append(fake_visits) # add one synthetic record
                total_number += len(fake_visits)
                if total_number >= n: break

        return fake_visit_list

    def _translate_sparse_visits_to_dense(self, visits):
        def _map_func(x):
            res = np.where(x > 0)[0].tolist()
            return [0] if len(res) == 0 else res # pad if nothing happened

        outputs = defaultdict(list)
        for batchv in visits:
            voc_offset = 0
            for i, o in enumerate(self.config['orders']):
                voc_size = self.config['vocab_size'][i]
                visit = batchv[...,voc_offset:voc_offset+voc_size] # 10, 7, 5
                for visit_ in visit:
                    res = list(map(_map_func, visit_))
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
    
    def _input_data_check(self, inputs):
        assert isinstance(inputs, SequencePatientBase), f'`trial_simulation.sequence` models require input training data in `SequencePatientBase`, find {type(inputs)} instead.'

    