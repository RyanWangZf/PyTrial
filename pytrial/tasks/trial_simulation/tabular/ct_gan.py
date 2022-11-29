'''
Implement CTGAN model for tabular simulation
prediction in clinical trials.
'''
import os
import warnings
import joblib

from .base import TabularSimulationBase
from .copula_gan import CTGANSynthesizer

from pytrial.data.patient_data import TabularPatientBase
from pytrial.utils.check import check_checkpoint_file, check_model_dir, check_model_config_file, make_dir_if_not_exist

warnings.filterwarnings('ignore')


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


class CTGAN(TabularSimulationBase):
    '''
    Implement CTGAN model for tabular simulation
    prediction in clinical trials.

    Parameters
    ----------
    embedding_dim (int):
        Size of the random sample passed to the Generator. Defaults to 128.

    generator_dim (tuple or list of ints):
        Size of the output samples for each one of the Residuals. A Residual Layer
        will be created for each one of the values provided. Defaults to (256, 256).
    
    discriminator_dim (tuple or list of ints):
        Size of the output samples for each one of the Discriminator Layers. A Linear Layer
        will be created for each one of the values provided. Defaults to (256, 256).
    
    generator_lr (float):
        Learning rate for the generator. Defaults to 2e-4.
    
    generator_decay (float):
        Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
    
    discriminator_lr (float):
        Learning rate for the discriminator. Defaults to 2e-4.
    
    discriminator_decay (float):
        Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
    
    batch_size (int):
        Number of data samples to process in each step.
    
    discriminator_steps (int):
        Number of discriminator updates to do for each generator update.
        From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
        default is 5. Default used is 1 to match original CTGAN implementation.
    
    log_frequency (boolean):
        Whether to use log frequency of categorical levels in conditional
        sampling. Defaults to ``True``.
    
    verbose (boolean):
        Whether to have print statements for progress results. Defaults to ``True``.
    
    epochs (int):
        Number of training epochs. Defaults to 300.
    
    pac (int):
        Number of samples to group together when applying the discriminator.
        Defaults to 10.
    
    cuda (bool or str):
        If ``True``, use CUDA. If a ``str``, use the indicated device.
        If ``False``, do not use cuda at all.

    experiment_id: str, optional (default='trial_simulation.tabular.ctgan')
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
            experiment_id='trial_simulation.tabular.ctgan',
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
            'model_name': 'ct_gan',
        }
        self._save_config(self.config)

    def fit(self, train_data):
        '''
        Train CTGAN model to simulate patient outcome
        with tabular input data.
        
        Parameters
        ----------
        train_data: tabular data
        '''
        self._input_data_check(train_data)
        self._build_model()
        if isinstance(train_data, TabularPatientBase):  # transform=True
            self.metadata = train_data.metadata
            self.raw_dataset = train_data
        if isinstance(train_data, dict):
            tabular_patient_base_dataset = TabularPatientBase(train_data, transform=True)
            self.metadata = tabular_patient_base_dataset.metadata
            self.raw_dataset = tabular_patient_base_dataset
        dataset = self.raw_dataset.reverse_transform()  # transform back

        categoricals = []
        fields_names = self.metadata['sdtypes']
        for field in dataset.columns:
            field_name = field.replace('.value', '')
            if field_name in fields_names:
                meta = fields_names[field_name]
                if meta == 'categorical':
                    categoricals.append(field)
        self._fit_model(dataset, categoricals)

    def predict(self, number_of_predictions=200):
        '''
        simulate a new tabular data with number_of_predictions.

        Parameters
        ----------
        number_of_predictions: number of predictions

        Returns
        -------
        ypred: dataset, same as the input dataset
            A new tabular data simulated by the model
        '''
        ypred = self.model.sample(number_of_predictions)  # build df
        return ypred  # output: dataset, same as the input dataset, don't need to transform back

    def save_model(self, output_dir=None):
        '''
        Save the learned CTGAN model to the disk.

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
        ckpt_path = os.path.join(output_dir, 'ctgan.model')
        joblib.dump(self.model, ckpt_path)

    def load_model(self, checkpoint=None):
        '''
        Save the learned CTGAN model to the disk.

        Parameters
        ----------
        
        checkpoint: str or None
            If a directory, the only checkpoint file `.model` will be loaded.
            If a filepath, will load from this file;
            If None, will load from `self.checkout_dir`.
        '''
        if checkpoint is None:
            checkpoint = self.checkout_dir

        checkpoint_filename = check_checkpoint_file(checkpoint, suffix='model')
        config_filename = check_model_config_file(checkpoint)
        self.model = joblib.load(checkpoint_filename)
        self.config = self._load_config(config_filename)

    def _build_model(self):
        self.model = BuildModel(self.config)

    def _fit_model(self, data, discrete_columns):
        self.model.fit(data, discrete_columns=discrete_columns)
