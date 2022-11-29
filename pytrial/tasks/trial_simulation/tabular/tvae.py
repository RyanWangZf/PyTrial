'''
Implement TVAE model for tabular simulation
prediction in clinical trials.
'''
import os
import warnings
import joblib
import torch
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from ctgan import TVAESynthesizer as TVAESynthesizerBase
from ctgan.synthesizers.tvae import Encoder, Decoder

from .base import TabularSimulationBase
from .copula_gan import DataTransformer
from pytrial.data.patient_data import TabularPatientBase
from pytrial.utils.check import check_checkpoint_file, check_model_dir, check_model_config_file, make_dir_if_not_exist

warnings.filterwarnings('ignore')

def _loss_function(recon_x, x, sigmas, mu, logvar, output_info, factor):
    st = 0
    loss = []
    for column_info in output_info:
        for span_info in column_info:
            if span_info.activation_fn != 'softmax':
                ed = st + span_info.dim
                std = sigmas[st]
                eq = x[:, st] - torch.tanh(recon_x[:, st])
                loss.append((eq ** 2 / 2 / (std ** 2)).sum())
                loss.append(torch.log(std) * x.size()[0])
                st = ed

            else:
                ed = st + span_info.dim
                loss.append(cross_entropy(
                    recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'))
                st = ed

    assert st == recon_x.size()[1]
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    return sum(loss) * factor / x.size()[0], KLD / x.size()[0]

class TVAESynthesizer(TVAESynthesizerBase):
    def fit(self, train_data, discrete_columns=()):
        """Fit the TVAE Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self.transformer = DataTransformer()
        self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data)
        dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self._device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        data_dim = self.transformer.output_dimensions
        encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self._device)
        self.decoder = Decoder(self.embedding_dim, self.decompress_dims, data_dim).to(self._device)
        optimizerAE = Adam(
            list(encoder.parameters()) + list(self.decoder.parameters()),
            weight_decay=self.l2scale)

        for i in range(self.epochs):
            for id_, data in enumerate(loader):
                optimizerAE.zero_grad()
                real = data[0].to(self._device)
                mu, std, logvar = encoder(real)
                eps = torch.randn_like(std)
                emb = eps * std + mu
                rec, sigmas = self.decoder(emb)
                loss_1, loss_2 = _loss_function(
                    rec, real, sigmas, mu, logvar,
                    self.transformer.output_info_list, self.loss_factor
                )
                loss = loss_1 + loss_2
                loss.backward()
                optimizerAE.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)


class BuildModel:
    def __new__(self, config) -> TVAESynthesizer:
        model = TVAESynthesizer(
            embedding_dim=config['embedding_dim'],
            compress_dims=config['compress_dims'],
            decompress_dims=config['decompress_dims'],
            l2scale=config['l2scale'],
            batch_size=config['batch_size'],
            epochs=config['epochs'],
            loss_factor=config['loss_factor'],
            cuda=config['cuda'],
            )

        return model


class TVAE(TabularSimulationBase):
    '''
    Implement TVAE model for tabular simulation
    prediction in clinical trials.

    Parameters
    ----------
    embedding_dim: int
        Size of the random sample passed to the Generator. Defaults to 128.

    compress_dims: tuple or list[int]
        Size of each hidden layer in the encoder. Defaults to (128, 128).
    
    decompress_dims: tuple or list[int]
       Size of each hidden layer in the decoder. Defaults to (128, 128).
    
    l2scale: int
        Regularization term. Defaults to 1e-5.
    
    batch_size: int
        Number of data samples to process in each step.
    
    epochs: int
        Number of training epochs. Defaults to 300.
    
    loss_factor: int
        Multiplier for the reconstruction error. Defaults to 2.
    
    cuda: bool or str
        - If ``True``, use CUDA. If a ``str``, use the indicated device.
        - If ``False``, do not use cuda at all.

    experiment_id: str, optional
        The name of current experiment. Decide the saved model checkpoint name.
    '''
    def __init__(
        self,
        embedding_dim=128,
        compress_dims=(128, 128),
        decompress_dims=(128, 128),
        l2scale=1e-5,
        batch_size=500,
        epochs=50,
        loss_factor=2,
        cuda=False,
        experiment_id='trial_simulation.tabular.tvae',
    ) -> None:
        super().__init__(experiment_id=experiment_id)
        self.config = {
            'embedding_dim' : embedding_dim,
            'compress_dims' : compress_dims,
            'decompress_dims' : decompress_dims,
            'l2scale' : l2scale,
            'batch_size': batch_size,
            'epochs': epochs,
            'loss_factor' : loss_factor,
            'cuda' : cuda,
            'experiment_id': experiment_id,
            'model_name': 'tvae',
        }
        self._save_config(self.config)

    def fit(self, train_data):
        '''
        Train TVAE model to simulate patient data
        with tabular input data.

        Parameters
        ----------
        train_data: TabularPatientBase
            The training data for TVAE model.
        '''
        self._input_data_check(train_data)
        self._build_model()
        if isinstance(train_data, TabularPatientBase): # transform=True
            dataset = train_data.df
        if isinstance(train_data, dict): 
            dataset = TabularPatientBase(train_data, transform=True)
            dataset = dataset.df

        self.metadata = train_data.metadata
        self.raw_dataset = train_data

        dataset = self.raw_dataset.reverse_transform() # transform back

        categoricals = []
        fields_before_transform = self.metadata['sdtypes']
        for field in dataset.columns:
            field_name = field.replace('.value', '')
            if field_name in fields_before_transform:
                meta = fields_before_transform[field_name]
                if meta == 'categorical':
                    categoricals.append(field)

        self._fit_model(dataset, categoricals) 

    def predict(self, number_of_predictions=200):
        '''
        simulate a new tabular data with number_of_predictions.

        Parameters
        ----------
        number_of_predictions: int
            The number of new data to simulate.

        Returns
        -------
        ypred: TanularPatientBase
            A new tabular data simulated by the model
        '''
        ypred = self.model.sample(number_of_predictions) # build df
        return ypred # output: dataset, same as the input dataset not transform back

    def save_model(self, output_dir=None):
        '''
        Save the learned TVAE model to the disk.

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
        ckpt_path = os.path.join(output_dir, 'tvae.model')
        joblib.dump(self.model, ckpt_path)

    def load_model(self, checkpoint=None):
        '''
        Load the learned TVAE model from the disk.

        Parameters
        ----------
        checkpoint: str or None
            The path to the checkpoint file.
            
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

    def _fit_model(self, data, discrete_columns):
        self.model.fit(data, discrete_columns=discrete_columns)
