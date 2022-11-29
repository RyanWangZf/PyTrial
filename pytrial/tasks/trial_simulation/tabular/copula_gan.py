'''
Implement CopulaGAN model for tabular simulation
prediction in clinical trials.
'''
import os
import pdb
import warnings
import joblib

import torch
from torch import optim
import numpy as np
import pandas as pd
from ctgan import CTGANSynthesizer as CTGANSynthesizerBase
from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer as DataTransformerBase
from ctgan.synthesizers.ctgan import Generator, Discriminator

from .base import TabularSimulationBase
from pytrial.data.patient_data import TabularPatientBase
from pytrial.utils.check import check_checkpoint_file, check_model_dir, check_model_config_file, make_dir_if_not_exist

warnings.filterwarnings('ignore')

class DataTransformer(DataTransformerBase):
    def transform(self, raw_data):
        """Take raw data and output a matrix data."""
        if not isinstance(raw_data, pd.DataFrame):
            column_names = [str(num) for num in range(raw_data.shape[1])]
            raw_data = pd.DataFrame(raw_data, columns=column_names)

        # Only use parallelization with larger data sizes.
        # Otherwise, the transformation will be slower.
        # if raw_data.shape[0] < 500:
        #     column_data_list = self._synchronous_transform(
        #         raw_data,
        #         self._column_transform_info_list
        #     )
        # else:
        #     column_data_list = self._parallel_transform(
        #         raw_data,
        #         self._column_transform_info_list
        #     )


        # do not use parallelization cuz bugs
        column_data_list = self._synchronous_transform(
            raw_data,
            self._column_transform_info_list
        )

        return np.concatenate(column_data_list, axis=1).astype(float)


class CTGANSynthesizer(CTGANSynthesizerBase):
    def fit(self, train_data, discrete_columns=(), epochs=None):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self._validate_discrete_columns(train_data, discrete_columns)
        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                ('`epochs` argument in `fit` method has been deprecated and will be removed '
                 'in a future version. Please pass `epochs` to the constructor instead'),
                DeprecationWarning
            )

        # skip transformation beacuse we have already transformed the data
        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)
        train_data = self._transformer.transform(train_data)

        self._data_sampler = DataSampler(
            train_data,
            self._transformer.output_info_list,
            self._log_frequency)

        data_dim = self._transformer.output_dimensions

        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(),
            self._generator_dim,
            data_dim
        ).to(self._device)

        discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim,
            pac=self.pac
        ).to(self._device)

        optimizerG = optim.Adam(
            self._generator.parameters(), lr=self._generator_lr, betas=(0.5, 0.9),
            weight_decay=self._generator_decay
        )

        optimizerD = optim.Adam(
            discriminator.parameters(), lr=self._discriminator_lr,
            betas=(0.5, 0.9), weight_decay=self._discriminator_decay
        )

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        for i in range(epochs):
            for id_ in range(steps_per_epoch):

                for n in range(self._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)

                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._data_sampler.sample_data(self._batch_size, col, opt)
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(
                            self._batch_size, col[perm], opt[perm])
                        c2 = c1[perm]

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    real = torch.from_numpy(real.astype('float32')).to(self._device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakeact

                    y_fake = discriminator(fake_cat)
                    y_real = discriminator(real_cat)

                    pen = discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac)
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    optimizerD.zero_grad()
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy

                optimizerG.zero_grad()
                loss_g.backward()
                optimizerG.step()

            if self._verbose:
                print(f'Epoch {i+1}, Loss G: {loss_g.detach().cpu(): .4f},'  # noqa: T001
                      f'Loss D: {loss_d.detach().cpu(): .4f}',
                      flush=True)


class BuildModel:
    def __new__(self, config) -> CTGANSynthesizer:
        model = CTGANSynthesizer(
            embedding_dim=config['embedding_dim'],
            generator_dim=config['generator_dim'],
            discriminator_dim=config['discriminator_dim'],
            generator_lr=config['generator_lr'],
            generator_decay=config['generator_decay'],
            discriminator_lr=config['discriminator_lr'],
            discriminator_decay=config['discriminator_decay'],
            batch_size=config['batch_size'],
            discriminator_steps=config['discriminator_steps'],
            log_frequency=config['log_frequency'],
            verbose=config['verbose'],
            epochs=config['epochs'],
            pac=config['pac'],
            cuda=config['cuda'],
            )
        return model


class CopulaGAN(TabularSimulationBase):
    '''
    Implement CopulaGAN model for tabular simulation
    prediction in clinical trials.

    Parameters
    ----------
    embedding_dim: int
        Size of the random sample passed to the Generator. Defaults to 128.

    generator_dim: tuple or list of int:
        Size of the output samples for each one of the Residuals. A Residual Layer
        will be created for each one of the values provided. Defaults to (256, 256).
    
    discriminator_dim: tuple or list of ints
        Size of the output samples for each one of the Discriminator Layers. A Linear Layer
        will be created for each one of the values provided. Defaults to (256, 256).
    
    generator_lr: float
        Learning rate for the generator. Defaults to 2e-4.
    
    generator_decay: float
        Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
    
    discriminator_lr: float
        Learning rate for the discriminator. Defaults to 2e-4.
    
    discriminator_decay: float
        Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
    
    batch_size: int
        Number of data samples to process in each step.
    
    discriminator_steps: int
        Number of discriminator updates to do for each generator update.
        From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
        default is 5. Default used is 1 to match original CTGAN implementation.
    
    log_frequency: boolean
        Whether to use log frequency of categorical levels in conditional
        sampling. Defaults to ``True``.
    
    verbose: boolean
        Whether to have print statements for progress results. Defaults to ``True``.
    
    epochs: int
        Number of training epochs. Defaults to 10.
    
    pac: int
        Number of samples to group together when applying the discriminator.
        Defaults to 10.
    
    cuda: bool or str
        - If ``True``, use CUDA. If a ``str``, use the indicated device.
        - If ``False``, do not use cuda at all.
        
    experiment_id: str, optional
        The name of current experiment. Decide the saved model checkpoint name.
    '''
    def __init__(
        self,
        embedding_dim=128,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        generator_lr=2e-4,
        generator_decay=1e-6,
        discriminator_lr=2e-4,
        discriminator_decay=1e-6,
        batch_size=500,
        discriminator_steps=1,
        log_frequency=True,
        verbose=True,
        epochs=50,
        pac=10,
        cuda=False, # can be set to "True" if applicable
        experiment_id='trial_simulation.tabular.copulagan',
    ) -> None:
        super().__init__(experiment_id=experiment_id)
        self.config = {
            'embedding_dim' : embedding_dim,
            'generator_dim' : generator_dim,
            'discriminator_dim' : discriminator_dim,
            'generator_lr' : generator_lr,
            'generator_decay' : generator_decay,
            'discriminator_lr' : discriminator_lr,
            'discriminator_decay' : discriminator_decay,
            'batch_size' : batch_size,
            'discriminator_steps' : discriminator_steps,
            'log_frequency' : log_frequency,
            'verbose' : verbose,
            'epochs' : epochs,
            'pac' : pac,
            'cuda' : cuda,
            'experiment_id': experiment_id,
            'model_name': 'copula_gan',
        }
        self._save_config(self.config)
        self._build_model()

    def fit(self, train_data):
        '''
        Train CopulaGAN model to simulate patient outcome
        with tabular input data.

        Parameters
        ----------
        train_data: TabularPatientBase
            The training data for the model.
        '''
        self._input_data_check(train_data)
        if isinstance(train_data, TabularPatientBase): # transform=True
            dataset = train_data.df
        if isinstance(train_data, dict): 
            dataset = TabularPatientBase(train_data, transform=True)
            dataset = dataset.df
        self._fit_model(dataset)
        self.metadata = train_data.metadata
        self.raw_dataset = train_data
        
    def predict(self, number_of_predictions=200):
        '''
        Simulate new tabular data with number_of_predictions.

        Parameters
        ----------
        number_of_predictions: int
            The number of synthetic data to generate.

        Returns
        -------
        ypred: TabularPatientBase
            A new tabular data simulated by the model
        '''
        ypred = self.model.sample(number_of_predictions)  # build df
        ypred = self.raw_dataset.reverse_transform(ypred) # transform back
        return ypred # output: dataset, same as the input dataset

    def save_model(self, output_dir=None):
        '''
        Save the learned CopulaGAN model to the disk.

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
        ckpt_path = os.path.join(output_dir, 'copulagan.model')
        joblib.dump(self.model, ckpt_path)

    def load_model(self, checkpoint=None):
        '''
        Save the learned CopulaGAN model to the disk.

        Parameters
        ----------
        checkpoint: str or None
            The path to the saved model.
            - If a directory, the only checkpoint file `.model` will be loaded.
            - If a filepath, will load from this file;
            - If None, will load from `self.checkout_dir`.
        '''
        if checkpoint is None:
            checkpoint = self.checkout_dir

        checkpoint_filename = check_checkpoint_file(checkpoint, suffix='model')
        config_filename = check_model_config_file(checkpoint)
        self.model = joblib.load(checkpoint_filename)
        self.config = self._load_config(config_filename)

    def _build_model(self):
        self.model = BuildModel(self.config)

    def _fit_model(self, data):
        self.model.fit(data)
