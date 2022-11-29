import pdb
import math
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
from tqdm import trange

from pytrial.data.patient_data import SequencePatientBase, SeqPatientCollator
from pytrial.utils.check import (
    check_checkpoint_file, check_model_dir, check_model_config_file, make_dir_if_not_exist
)
from .base import SequenceSimulationBase
from ..losses import GeneratorLoss, DiscriminatorLossGP, MultilabelBinaryXentLoss
from ..trainer import SeqSimSyntegTrainer
from ..data import SequencePatient

class Embedding(nn.Module):
    def __init__(self, total_vocab_size, emb_size):
        """Construct an embedding matrix to embed sparse codes"""
        super(Embedding, self).__init__()
        self.code_embed = nn.Embedding(total_vocab_size+3, emb_size)

    def forward(self, codes): # batch_size * visits * codes
        code_embeds = self.code_embed(codes)
        return code_embeds

class SingleVisitTransformer(nn.Module):
    """An Encoder Transformer to turn code embeddings into a visit embedding"""
    def __init__(self, emb_size, n_head, hidden_dim):
        super(SingleVisitTransformer, self).__init__()
        encoderLayer = nn.TransformerEncoderLayer(emb_size, n_head, 
                        dim_feedforward=hidden_dim, dropout=0.1, activation="relu", 
                        layer_norm_eps=1e-08, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoderLayer, 2)

    def forward(self, code_embeddings, visit_lengths):
        bs, vs, cs, ed = code_embeddings.shape
        mask = torch.ones((bs, vs, cs)).to(code_embeddings.device)
        for i in range(bs):
            for j in range(vs):
                mask[i,j,:visit_lengths[i,j]] = 0
        visits = torch.reshape(code_embeddings, (bs*vs,cs,ed))
        mask = torch.reshape(mask, (bs*vs,cs))
        encodings = self.transformer(visits, src_key_padding_mask=mask)
        encodings = torch.reshape(encodings, (bs,vs,cs,ed))
        visit_representations = encodings[:,:,0,:]
        return visit_representations

class RecurrentLayer(nn.Module):
    """An Recurrent Layer to predict the next visit based on the visit embeddings"""
    def __init__(self, hidden_dim, n_rnn_layer):
        super(RecurrentLayer, self).__init__()
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=n_rnn_layer, dropout=0.1)

    def forward(self, visit_embeddings):   
        output, _ = self.lstm(visit_embeddings)
        return output

class DependencyModel(nn.Module):
    """The entire Dependency Model component of SynTEG"""
    def __init__(self, emb_size, hidden_dim, condition_dim, total_vocab_size, n_head, n_rnn_layer):
        super(DependencyModel, self).__init__()
        self.embeddings = Embedding(total_vocab_size, emb_size)
        self.visit_att = SingleVisitTransformer(emb_size, n_head, hidden_dim)
        self.proj1 = nn.Linear(emb_size, hidden_dim)
        self.lstm = RecurrentLayer(hidden_dim, n_rnn_layer)
        self.proj2 = nn.Linear(hidden_dim, condition_dim)
        self.proj3 = nn.Linear(condition_dim, total_vocab_size)
        
    def forward(self, inputs_word, visit_lengths, export=False):  # bs * visits * codes, bs * visits * 1 
        inputs = self.embeddings(inputs_word) # bs * visits * codes * emb_size
        inputs = self.visit_att(inputs, visit_lengths) # bs * visits * emb_size
        inputs = self.proj1(inputs) # bs * visits * hidden_dim
        output = self.lstm(inputs) # bs * visits * hidden_dim
        if export:
            return self.proj2(output)[:, :-1, :] # bs * visit * condition
        else:
            output = self.proj3(torch.relu(self.proj2(output))) # bs * visits * total_vocab_size
            diagnosis_output = output[:, :-1, :]
            return diagnosis_output

#######
### Conditional GAN Model
#######

class PointWiseLayer(nn.Module):
    def __init__(self, num_outputs):
        """Construct an embedding matrix to embed sparse codes"""
        super(PointWiseLayer, self).__init__()
        self.bias = nn.Parameter(torch.zeros(num_outputs).uniform_(-math.sqrt(num_outputs), math.sqrt(num_outputs)))

    def forward(self, x1, x2):
        return x1 * x2 + self.bias

class Generator(nn.Module):
    def __init__(self, g_dims, z_dim, condition_dim):
        super(Generator, self).__init__()
        self.dense_layers = nn.Sequential(*[nn.Linear(g_dims[i-1] if i > 0 else z_dim, g_dims[i]) for i in range(len(g_dims[:-1]))])
        self.batch_norm_layers = nn.Sequential(*[nn.BatchNorm1d(dim, eps=1e-5) for dim in g_dims[:-1]])
        self.output_layer = nn.Linear(g_dims[-2], g_dims[-1])
        self.output_sigmoid = nn.Sigmoid()
        self.condition_layers = nn.Sequential(*[nn.Linear(condition_dim, dim) for dim in g_dims[:-1]])
        self.pointwiselayers = nn.Sequential(*[PointWiseLayer(dim) for dim in g_dims[:-1]])

    def forward(self, x, condition):
        for i in range(len(self.dense_layers)):
            h = self.dense_layers[i](x)
            x = nn.functional.relu(self.pointwiselayers[i](self.batch_norm_layers[i](h), self.condition_layers[i](condition)))
        x = self.output_layer(x)
        x = self.output_sigmoid(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, d_dims, g_dims, condition_dim):
        super(Discriminator, self).__init__()
        self.dense_layers = nn.Sequential(*[nn.Linear(d_dims[i-1] if i > 0 else g_dims[-1] + 1, d_dims[i]) for i in range(len(d_dims))])
        self.layer_norm_layers = nn.Sequential(*[nn.LayerNorm(dim, eps=1e-5) for dim in d_dims])
        self.output_layer = nn.Linear(d_dims[-1], 1)
        self.condition_layers = nn.Sequential(*[nn.Linear(condition_dim, dim) for dim in d_dims])
        self.pointwiselayers = nn.Sequential(*[PointWiseLayer(dim) for dim in d_dims])

    def forward(self, x, condition):
        a = (2 * x) ** 15
        sparsity = torch.sum(a / (a + 1), axis=-1, keepdim=True)
        x = torch.cat((x, sparsity), axis=-1)
        for i in range(len(self.dense_layers)):
            h = self.dense_layers[i](x)
            x = self.pointwiselayers[i](self.layer_norm_layers[i](h), self.condition_layers[i](condition))
        x = self.output_layer(x)
        return x
      
class BuildModel(nn.Module):
    def __init__(self,
        emb_size,
        hidden_dim,
        condition_dim,
        n_head,
        n_rnn_layer,
        total_vocab_size,
        z_dim,
        g_dims,
        d_dims,
        device,
        max_visit,
        max_code_per_visit,
        **kwargs,
        ) -> None:
        super().__init__()
        
        self.device = device
        self.total_vocab_size = total_vocab_size
        self.max_visit = max_visit
        self.max_code_per_visit = max_code_per_visit
        self.z_dim = z_dim
        
        self.dependency_module = DependencyModel(
            emb_size=emb_size,
            hidden_dim=hidden_dim,
            condition_dim=condition_dim,
            total_vocab_size=total_vocab_size,
            n_head=n_head,
            n_rnn_layer=n_rnn_layer
        )

        self.generator_module = Generator(
            z_dim=z_dim,
            g_dims=g_dims,
            condition_dim=condition_dim
        )
        
        self.discriminator_module = Discriminator(
            d_dims=d_dims,
            g_dims=g_dims,
            condition_dim=condition_dim
        )
        
    def dependency_forward(self, inputs, export=False):
        logits = self.dependency_module(inputs['v'], inputs['c_lengths'], export)
        if export:
            return logits
        
        labels = torch.sum(nn.functional.one_hot(inputs['v'].long(), num_classes=self.total_vocab_size+2), dim=2).float()[:, 1:, :-2]
        for idx in range(len(inputs['v_lengths'])):
            logits[idx, inputs['v_lengths'][idx]:] = 0
        
        return logits, labels
    
    def gan_forward(self, inputs):
        x_real = torch.sum(nn.functional.one_hot(inputs['v'].long(), num_classes=self.total_vocab_size+2), dim=1).float()[:, :-2]
        z = torch.randn((len(x_real), self.z_dim)).to(self.device)
        x_fake = self.generator_module(z, inputs['conditions'])
        y_fake = self.discriminator_module(x_fake, inputs['conditions'])
        y_real = self.discriminator_module(x_real, inputs['conditions'])
        alpha = torch.rand((x_real.size(0), 1)).to(self.device)
        interpolates = (alpha * x_real + (1-alpha)*x_fake).requires_grad_(True).float()
        d_interpolates = self.discriminator_module(interpolates, inputs['conditions'])
        return {
            'y_fake': y_fake,
            'y_real': y_real,
            'interpolates': interpolates,
            'd_interpolates': d_interpolates
        }
    
    def forward(self, inputs, export=False):
        if 'conditions' in inputs:
            return self.gan_forward(inputs)
        else:
            return self.dependency_forward(inputs, export)
        
    def add_condition(self, inputs):
        condition = self.dependency_forward(inputs, True)
        visits = inputs['v'][:,1:]
        v_new = []
        condition_new = []
        for i in range(len(inputs['v_lengths'])):
            for j in range(inputs['v_lengths'][i]):
                v_new.append(visits[i, j, :])
                condition_new.append(condition[i,j,:])
        
        inputs['v'] = torch.stack(v_new)
        inputs['conditions'] = torch.stack(condition_new)
        return inputs
      
    def sample(self, n_samples):
        ehr = torch.zeros((n_samples, 1, self.total_vocab_size), device=self.device)
        batch_ehr = torch.ones((n_samples, self.max_visit+1, self.max_code_per_visit+1), device=self.device, dtype=torch.int) * (self.total_vocab_size + 1)
        batch_ehr[:,:,0] = self.total_vocab_size
        batch_lens = torch.zeros((n_samples, self.max_visit+1, 1), dtype=torch.int, device=self.device)
        batch_lens[:,0,0] = 1
        with torch.no_grad():
            for j in trange(self.max_visit):
                for i in range(n_samples):
                    codes = torch.nonzero(ehr[i,j]).squeeze(1)
                    batch_ehr[i,j,:min(len(codes), self.max_code_per_visit)] = codes[1:min(len(codes), self.max_code_per_visit) + 1]
                    batch_lens[i,j] = 1 + min(len(codes), self.max_code_per_visit)
                
                condition_vector = self.dependency_module(batch_ehr, batch_lens, export=True)
                condition = condition_vector[:,j,:]
                z = torch.randn((n_samples, self.z_dim), device=self.device)
                visit = self.generator_module(z, condition)
                visit = torch.bernoulli(visit).unsqueeze(1)
                ehr = torch.cat((ehr, visit), dim=1)
        return ehr[:,1:]
    

class SynTEG(SequenceSimulationBase):
    '''
    Implement a GAN based model for longitudinal patient records simulation [1]_.
    
    Parameters
    ----------
    vocab_size: list[int]
        A list of vocabulary size for different types of events, e.g., for diagnosis, procedure, medication.

    order: list[str]
        The order of event types in each visits, e.g., ``['diag', 'prod', 'med']``.
        Visit = [diag_events, prod_events, med_events], each event is a list of codes.

    max_visit: int
        Maximum number of visits.
        
    max_code_per_visit: int
        Maximum number of medical codes in a single visit.

    emb_size: int
        Embedding size for encoding input event codes.
        
    hidden_dim: int
        Size of intermediate hidden dimension for RNN and Feed Forward layers
        
    condition_dim: int
        Size of intermediate dimension for encoding medical history to condition the GAN
        
    n_head: int
        Number of attention heads
        
    n_rnn_layer: int
        Number of RNN layers for encoding historical events.
        
    z_dim: int
        Dimension of noise vector passed into the GAN Generator
        
    g_dims: list
        List of ints for intermediate GAN Generator dimensionalities
        
    d_dims: list
        List of ints for intermediate GAN Discriminator dimensionalities

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
    .. [1] Zhang, Ziqi, et al. (2021, March). SynTEG: a framework for temporal structured electronic health data simulation. Journal of the American Medical Informatics Association 28.3.
    '''
    def __init__(self,
        vocab_size,
        order,
        max_visit=20,
        max_code_per_visit=20,
        emb_size=64,
        hidden_dim=32,
        condition_dim=32,
        n_head=4,
        n_rnn_layer=2,
        z_dim=64,
        g_dims=[32, 32, 64, 64],
        d_dims=[64, 32, 32],
        learning_rate=1e-3,
        batch_size=64,
        epochs=20,
        num_worker=0,
        device='cpu',# 'cuda:0',
        experiment_id='trial_simulation.sequence.synteg',
        ):
        super().__init__(experiment_id)
        self.config = {
            'vocab_size': vocab_size, 
            'orders': order, 
            'max_visit': max_visit,
            'max_code_per_visit': max_code_per_visit,
            'emb_size': emb_size,
            'hidden_dim': hidden_dim,
            'condition_dim': condition_dim,
            'n_head': n_head,
            'n_rnn_layer': n_rnn_layer,
            'z_dim': z_dim,
            'g_dims': g_dims,
            'd_dims': d_dims,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs,
            'num_worker': num_worker,
            'device':device,
            }
        self.config['total_vocab_size'] = sum(vocab_size)
        self.config['g_dims'].append(self.config['total_vocab_size'])
        self.device = device
        self._build_model()
        
    def _build_model(self):
        self.model = BuildModel(
            emb_size = self.config['emb_size'],
            hidden_dim = self.config['hidden_dim'],
            condition_dim = self.config['condition_dim'],
            n_head = self.config['n_head'],
            n_rnn_layer = self.config['n_rnn_layer'],
            total_vocab_size = self.config['total_vocab_size'],
            z_dim = self.config['z_dim'],
            g_dims = self.config['g_dims'],
            d_dims = self.config['d_dims'],
            device = self.device,
            max_visit = self.config['max_visit'],
            max_code_per_visit = self.config['max_code_per_visit']
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

    def predict(self, n, return_tensor=True):
        '''
        Generate synthetic records

        Parameters
        ----------
        n: int
            How many samples in total will be generated.
        
        return_tensor: bool
            If `True`, return output generated records in tensor format (n, n_visit, n_event), good for later predictive modeling.
            If `False, return records in `SequencePatient` format.
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
                  'dependency': self.model.dependency_module.state_dict(),
                  'generator': self.model.generator_module.state_dict(),
                  'discriminator': self.model.discriminator_module.state_dict()
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
        state_dict = torch.load(checkpoint_filename, map_location='cpu')
        if config_filename is not None:
            config = self._load_config(config_filename)
            self.config.update(config)
        self.model.dependency_module.load_state_dict(state_dict['dependency'])
        self.model.generator_module.load_state_dict(state_dict['generator'])
        self.model.discriminator_module.load_state_dict(state_dict['discriminator'])
    
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
    
    def get_test_dataloader(self, test_data):
        dataloader = DataLoader(test_data,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_worker'],
            pin_memory=False,
            shuffle=False,
            collate_fn=SeqPatientCollator(
                config={
                    'visit_mode':test_data.metadata['visit']['mode'],
                    'label_mode':test_data.metadata['label']['mode'],
                    }
                ),
            )
        return dataloader
    
    def _build_simulation_loss_model(self):
        return [MultilabelBinaryXentLoss(self.model)]
    
    def _build_gan_loss_model(self):
        return [DiscriminatorLossGP(self.model), GeneratorLoss(self.model)]

    def _fit_model(self, train_data):
        # PHASE 1: Simulation
        train_dataloader = self.get_train_dataloader(train_data)
        simulation_loss_models = self._build_simulation_loss_model()
        simulation_train_objectives = [(train_dataloader, loss_model) for loss_model in simulation_loss_models]
        trainer = SeqSimSyntegTrainer(
            model=self,
            train_objectives=simulation_train_objectives
            )
        trainer.train(**self.config)

        # PHASE 2: GAN
        gan_loss_models = self._build_gan_loss_model()
        gan_train_objectives = [(train_dataloader, loss_model) for loss_model in gan_loss_models]
        trainer = SeqSimSyntegTrainer(
            model=self,
            train_objectives=gan_train_objectives,
            condition=True
            )
        trainer.train(**self.config)

    @torch.no_grad()
    def _predict(self, n):
        return self.model.sample(n)
    
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
            'c_lengths':{},
            'v_lengths':[],
            'x':feature, # baseline feature
            }
        
        v_lengths = [len(visits[self.config['orders'][0]][idx][:self.config['max_visit']]) for idx in range(len(visits[self.config['orders'][0]]))]
        inputs['v_lengths'] = v_lengths

        v = torch.ones(len(v_lengths), max(v_lengths) + 1, self.config['max_code_per_visit'] + 1, dtype=torch.int) * (self.config['total_vocab_size'] + 1)
        c_lengths = torch.ones(len(v_lengths), max(v_lengths) + 1, 1, dtype=torch.int)
        for i in range(len(v_lengths)):
            for j in range(min(v_lengths[i], self.config['max_visit'])):
                visit = torch.IntTensor([r + sum(self.config['vocab_size'][:n-1]) if n > 0 else r for n, o in enumerate(self.config['orders']) for r in visits[o][i][j]][:self.config['max_code_per_visit']])
                v[i, j, 1:len(visit)+1] = visit
                c_lengths[i, j] = len(visit) + 1
        v[:,:,0] = self.config['total_vocab_size']
        c_lengths[:,0] = 1
        v = v.to(self.device)
        c_lengths = c_lengths.to(self.device)
        inputs['v'] = v
        inputs['c_lengths'] = c_lengths
        
        return inputs
    
    def _add_condition(self, inputs):
        return self.model.add_condition(inputs)
    
    def _input_data_check(self, inputs):
        assert isinstance(inputs, SequencePatientBase), f'`trial_simulation.sequence` models require input training data in `SequencePatientBase`, find {type(inputs)} instead.'
        