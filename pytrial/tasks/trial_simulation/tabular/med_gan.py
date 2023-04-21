import os
import pdb
import joblib

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import BatchNorm1d, Linear, Module, Sequential
from torch.nn.functional import cross_entropy, mse_loss, sigmoid
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from pytrial.data.patient_data import TabularPatientBase
# from pytrial.utils.check import check_checkpoint_file, check_model_dir, check_model_config_file, make_dir_if_not_exist
from pytrial.utils.tabular_utils import get_transformer
from pytrial.utils.check import check_checkpoint_file, check_model_dir, check_model_config_file, make_dir_if_not_exist


from .base import TabularSimulationBase

class ResidualFC(Module):
    def __init__(self, input_dim, output_dim, activate, bn_decay):
        super(ResidualFC, self).__init__()
        self.seq = Sequential(
            Linear(input_dim, output_dim),
            BatchNorm1d(output_dim, momentum=bn_decay),
            activate()
        )

    def forward(self, input):
        residual = self.seq(input)
        return input + residual


class Generator(Module):
    def __init__(self, random_dim, hidden_dim, bn_decay):
        super(Generator, self).__init__()

        dim = random_dim
        seq = []
        for item in list(hidden_dim)[:-1]:
            assert item == dim
            seq += [ResidualFC(dim, dim, nn.ReLU, bn_decay)]
        assert hidden_dim[-1] == dim
        seq += [
            Linear(dim, dim),
            BatchNorm1d(dim, momentum=bn_decay),
            nn.ReLU()
        ]
        self.seq = Sequential(*seq)

    def forward(self, input):
        return self.seq(input)


class Discriminator(Module):
    def __init__(self, data_dim, hidden_dim):
        super(Discriminator, self).__init__()
        dim = data_dim * 2
        seq = []
        for item in list(hidden_dim):
            seq += [
                Linear(dim, item),
                nn.ReLU() if item > 1 else nn.Sigmoid()
            ]
            dim = item
        self.seq = Sequential(*seq)

    def forward(self, input):
        mean = input.mean(dim=0, keepdim=True)
        mean = mean.expand_as(input)
        inp = torch.cat((input, mean), dim=1)
        return self.seq(inp)


class Encoder(Module):
    def __init__(self, data_dim, compress_dims, embedding_dim):
        super(Encoder, self).__init__()
        dim = data_dim
        seq = []
        for item in list(compress_dims) + [embedding_dim]:
            seq += [
                Linear(dim, item),
                nn.ReLU()
            ]
            dim = item
        self.seq = Sequential(*seq)

    def forward(self, input):
        return self.seq(input)


class Decoder(Module):
    def __init__(self, embedding_dim, decompress_dims, data_dim):
        super(Decoder, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [
                Linear(dim, item),
                nn.ReLU()
            ]
            dim = item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input, output_info):
        return self.seq(input)


def aeloss(fake, real, output_info):
    st = 0
    loss = []
    for item in output_info:
        if item[1] == 'sigmoid':
            ed = st + item[0]
            loss.append(mse_loss(sigmoid(fake[:, st:ed]), real[:, st:ed], reduction='sum'))
            st = ed
        elif item[1] == 'softmax':
            ed = st + item[0]
            loss.append(cross_entropy(
                fake[:, st:ed], torch.argmax(real[:, st:ed], dim=-1), reduction='sum'))
            st = ed
        else:
            assert 0
    return sum(loss) / fake.size()[0]


class MedGANSynthesizer:
    """docstring for MedGAN."""

    def __init__(
        self,
        embedding_dim=128,
        random_dim=128,
        generator_dims=(128, 128),          # 128 -> 128 -> 128
        discriminator_dims=(256, 128, 1),   # datadim * 2 -> 256 -> 128 -> 1
        compress_dims=(),                   # datadim -> embedding_dim
        decompress_dims=(),                 # embedding_dim -> datadim
        bn_decay=0.99,
        l2scale=0.001,
        pretrain_epoch=200,
        batch_size=1000,
        epochs=2000,
        device='cpu',
        verbose=False,
        ):

        self.embedding_dim = embedding_dim
        self.random_dim = random_dim
        self.generator_dims = generator_dims
        self.discriminator_dims = discriminator_dims

        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims
        self.bn_decay = bn_decay
        self.l2scale = l2scale

        self.pretrain_epoch = pretrain_epoch
        self.batch_size = batch_size
        self.epochs = epochs

        self.device = device
        self.verbose = verbose

    def _get_metadata(self, data):
        self.output_info = []
        for k, v in data.metadata['transformed_col2col'].items():
            if len(v) == 1:
                self.output_info.append(
                    (1, 'sigmoid')
                )
            else:
                self.output_info.append(
                    (len(v), 'softmax')
                )

    def fit(self, data):
        data_val = data.df.values
        dataset = TensorDataset(torch.from_numpy(data_val.astype('float32')).to(self.device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        
        self._get_metadata(data)
        
        # data_dim = self.transformer.output_dim
        data_dim = data_val.shape[1]
        encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self.device)
        self.decoder = Decoder(self.embedding_dim, self.compress_dims, data_dim).to(self.device)
        optimizerAE = Adam(
            list(encoder.parameters()) + list(self.decoder.parameters()),
            weight_decay=self.l2scale
        )

        for i in range(self.pretrain_epoch):
            if self.verbose:
                print('Pretrain Epoch: {} / {}'.format(i+1, self.pretrain_epoch))
            for id_, data in enumerate(loader):
                optimizerAE.zero_grad()
                real = data[0].to(self.device)
                emb = encoder(real)
                rec = self.decoder(emb, self.output_info)
                loss = aeloss(rec, real, self.output_info)

                loss.backward()
                optimizerAE.step()

        self.generator = Generator(
            self.random_dim, self.generator_dims, self.bn_decay).to(self.device)
        discriminator = Discriminator(data_dim, self.discriminator_dims).to(self.device)
        optimizerG = Adam(
            list(self.generator.parameters()) + list(self.decoder.parameters()),
            weight_decay=self.l2scale
        )
        optimizerD = Adam(discriminator.parameters(), weight_decay=self.l2scale)

        mean = torch.zeros(self.batch_size, self.random_dim, device=self.device)
        std = mean + 1
        for i in range(self.epochs):
            n_d = 2
            n_g = 1
            for id_, data in enumerate(loader):
                real = data[0].to(self.device)
                noise = torch.normal(mean=mean, std=std)
                emb = self.generator(noise)
                fake = self.decoder(emb, self.output_info)

                optimizerD.zero_grad()
                y_real = discriminator(real)
                y_fake = discriminator(fake)
                real_loss = -(torch.log(y_real + 1e-4).mean())
                fake_loss = (torch.log(1.0 - y_fake + 1e-4).mean())
                loss_d = real_loss - fake_loss
                loss_d.backward()
                optimizerD.step()

                if i % n_d == 0:
                    for _ in range(n_g):
                        noise = torch.normal(mean=mean, std=std)
                        emb = self.generator(noise)
                        fake = self.decoder(emb, self.output_info)
                        optimizerG.zero_grad()
                        y_fake = discriminator(fake)
                        loss_g = -(torch.log(y_fake + 1e-4).mean())
                        loss_g.backward()
                        optimizerG.step()

            if self.verbose:
                print(f'epoch {i} loss_d: {loss_d.item()} loss_g: {loss_g.item()}')


    def sample(self, n):
        self.generator.eval()
        self.decoder.eval()

        steps = n // self.batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self.batch_size, self.random_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(self.device)
            emb = self.generator(noise)
            fake = self.decoder(emb, self.output_info)
            fake = torch.sigmoid(fake)
            data.append(fake.detach().cpu().numpy())
        data = np.concatenate(data, axis=0)
        data = data[:n]
        return data
    
# utils
# ------------

# main functions
# ------------

class BuildModel:
    def __new__(self, config) -> MedGANSynthesizer:
        model = MedGANSynthesizer(
            embedding_dim=config['embedding_dim'],
            random_dim=config['random_dim'],
            generator_dims=config['generator_dims'],
            discriminator_dims=config['discriminator_dims'],
            compress_dims=config['compress_dims'],
            decompress_dims=config['decompress_dims'],
            bn_decay=config['bn_decay'],
            l2scale=config['l2scale'],
            pretrain_epoch=config['pretrain_epoch'],
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            device=config['device'],
            verbose=config['verbose'],
        )
        return model


class MedGAN(TabularSimulationBase):
    '''
    Implement MedGAN model for patient level tabular data generation [1]_.

    Parameters
    ----------
    embedding_dim : int, default 128
        Dimension of embedding layer.
    
    random_dim : int, default 128
        Dimension of random noise.
    
    generator_dims : tuple, default (128, 128)
        Dimension of generator layers.
    
    discriminator_dims : tuple, default (256, 128, 1)
        Dimension of discriminator layers.
    
    compress_dims : tuple, default ()
        Dimension of compressed embedding layer. datadim -> embedding_dim
    
    decompress_dims : tuple, default ()
        Dimension of decompressed embedding layer. embedding_dim -> datadim
    
    bn_decay : float, default 0.99
        Decay rate of batch normalization.
    
    l2scale : float, default 0.001
        L2 regularization scale.
    
    pretrain_epoch : int, default 200
        Number of pretrain epochs.
    
    batch_size : int, default 1000
        Batch size for training.
    
    epochs : int, default 1000
        Number of epochs for training.

    experiment_id: str
        Experiment id for logging.

    verbose: bool
        Whether to print training information.
    
    Notes
    -----
    .. [1] Choi, E., Biswal, S., Malin, B., Duke, J., Stewart, W. F., & Sun, J. (2017, November). Generating multi-label discrete patient records using generative adversarial networks. In Machine learning for healthcare conference (pp. 286-305). PMLR.
    '''
    def __init__(self,
        embedding_dim=128,
        random_dim=128,
        generator_dims=(128, 128),          # 128 -> 128 -> 128
        discriminator_dims=(256, 128, 1),   # datadim * 2 -> 256 -> 128 -> 1
        compress_dims=(),                   # datadim -> embedding_dim
        decompress_dims=(),                 # embedding_dim -> datadim
        bn_decay=0.99,
        l2scale=0.001,
        pretrain_epoch=200,
        batch_size=1000,
        epochs=2000,
        device='cpu',
        experiment_id='trial_simulation.tabular.medgan',
        verbose=False,
        ):
        super().__init__(experiment_id=experiment_id)
        self.config = {
            'embedding_dim': embedding_dim,
            'random_dim': random_dim,
            'generator_dims': generator_dims,
            'discriminator_dims': discriminator_dims,
            'compress_dims': compress_dims,
            'decompress_dims': decompress_dims,
            'bn_decay': bn_decay,
            'l2scale': l2scale,
            'pretrain_epoch': pretrain_epoch,
            'batch_size': batch_size,
            'epochs': epochs,
            'device': device,
            'verbose':verbose,
        }
        self._save_config(self.config)
    
    def fit(self, train_data):
        '''
        Train MedGAN model to generate synthetic tabular patient data.

        Parameters
        ----------
        train_data : TabularPatientBase
            Training data.
        '''
        self._input_data_check(train_data)
        self._build_model()
        self._fit_model(train_data)
    
    def predict(self, n):
        '''
        Generate synthetic tabular patient data.

        Parameters
        ----------
        n : int
            Number of samples to generate.
        
        Returns
        -------
        data : np.ndarray
            Generated synthetic data.
        '''
        return self.model.sample(n)
    
    def load_model(self, checkpoint):
        '''
        Load model from checkpoint.

        Parameters
        ----------
        checkpoint : str
            Path to checkpoint.
            If a directory is given, will load the latest checkpoint in the directory.
            If a filepath is given, will load the checkpoint from the filepath.
            If set None, will load from default directory `self.checkpoint_dir`.
        '''
        if checkpoint is None:
            checkpoint = self.checkpoint_dir
        
        checkpoint_filename = check_checkpoint_file(checkpoint, suffix='model')
        config_filename = check_model_config_file(checkpoint)
        self.model = joblib.load(checkpoint_filename)
        self.config = self._load_config(config_filename)

    def save_model(self, output_dir):
        '''
        Save model to checkpoint.

        Parameters
        ----------
        output_dir : str
            Output directory. If set None, will save to default directory `self.checkpoint_dir`.
        '''
        if output_dir is None:
            output_dir = self.checkpoint_dir

        make_dir_if_not_exist(output_dir)
        self._save_config(self.config, output_dir=output_dir)
        ckpt_path = os.path.join(output_dir, 'ctgan.model')
        joblib.dump(self.model, ckpt_path)

    def _build_model(self):
        self.model = BuildModel(self.config)

    def _fit_model(self, dataset):
        self.model.fit(dataset)
