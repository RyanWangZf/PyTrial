from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pdb

import torch
from torch import nn
from torch.utils.data import DataLoader

from pytrial.utils.check import (
    check_checkpoint_file, check_model_dir, check_model_config_file, make_dir_if_not_exist
)
from pytrial.data.site_data import SiteBaseDemographics
from .base import SiteSelectionBase
from .data import TrialSiteSimple, SiteSelectionBaseCollator
from .losses import PolicyGradientLossEnrollment, PolicyGradientLossCombined
from .trainer import SiteSelectTrainer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      
class BuildModel(nn.Module):
    def __init__(self, 
        trial_dim,
        site_dim,
        embedding_dim
        ) -> None:
        super().__init__()
        
        self.site_encoder = nn.Linear(site_dim, embedding_dim)
        self.trial_encoder = nn.Linear(trial_dim, embedding_dim)
        self.site_fc = nn.Linear(embedding_dim, embedding_dim)
        self.trial_fc = nn.Linear(embedding_dim, embedding_dim)
        self.score_encoder = nn.Linear(2*embedding_dim, 2*embedding_dim)
        self.fc = nn.Linear(2*embedding_dim, embedding_dim)
        self.output = nn.Linear(embedding_dim, 1)
        
    def forward(self, inputs):
        trial = inputs['trial']
        investigators = inputs['site']
        num_inv = investigators.size(1)
        site_encoding = torch.relu(self.site_fc(torch.relu(self.site_encoder(investigators))))
        trial_encoding = torch.relu(self.trial_fc(torch.relu(self.trial_encoder(trial))))
        trial_encoding = trial_encoding.unsqueeze(1).repeat(1, num_inv, 1)
        network_input = torch.cat((site_encoding, trial_encoding), dim=-1)
        network_input = torch.relu(self.score_encoder(network_input))
        score = self.output(torch.relu(self.fc(network_input))).squeeze(-1)
        return score

class PolicyGradientEntropy(SiteSelectionBase):
    '''
    Implement Policy Gradient Entropy model for selecting clinical trial sites based on possibly missing multi-model site features. [1]_
    
    Parameters
    ----------
    
    trial_dim: list[int]
        Size of the trial representation

    site_dim: int
        Size of the site representation

    embedding_dim: int
        Size of all of the modality and other intermediate embeddings
    
    Notes
    -----
    .. [1] Srinivasa, R. S., Qian, C., Theodorou, B., Spaeder, J., Xiao, C., Glass, L., & Sun, J. (2022). Clinical trial site matching with improved diversity using fair policy learning. arXiv preprint arXiv:2204.06501.

    '''
    def __init__(self, 
        trial_dim=211, 
        site_dim=124, 
        embedding_dim=64, 
        enrollment_only=True,
        K=10,
        lam=1,
        learning_rate=1e-4,
        weight_decay=1e-4,
        batch_size=64,
        epochs=10,
        num_worker=0,
        device='cuda:0',
        experiment_id='test',
        ) -> None:
        super().__init__(experiment_id)
        self.config = {
            'trial_dim':trial_dim,
            'site_dim':site_dim,
            'embedding_dim':embedding_dim,
            'enrollment_only':enrollment_only,
            'K':K,
            'lambda':lam,
            'learning_rate':learning_rate,
            'weight_decay':weight_decay,
            'batch_size':batch_size,
            'epochs':epochs,
            'num_worker':num_worker,
            'device':device,
            'experiment_id':experiment_id,
            }
        self.device = device
        self._build_model()

    def fit(self, train_data):
        '''
        Train model with historical trial-site enrollments.

        Parameters
        ----------
        train_data: TrialSiteSimple
            A `TrialSiteSimple` contains trials, sites, and enrollments.
        '''
        self._input_data_check(train_data)
        self._fit_model(train_data)

    def predict(self, test_data):
        '''
        Make prediction for site selection.
        '''
        selections = []
        self._input_data_check(test_data)
        dataloader = DataLoader(test_data,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_worker'],
            pin_memory=True,
            shuffle=False,
            collate_fn=SiteSelectionBaseCollator(
                config={
                    'has_demographics':isinstance(test_data.sites, SiteBaseDemographics)
                    }
                ),
            )
        for data in dataloader:
            inputs = self._prepare_input(data)
            scores = self.model(inputs)
            selections += [l[:self.config['K']] for l in scores.argsort(dim=1, descending=True).tolist()]
        return selections

    def save_model(self, output_dir):
        '''
        Save the learned patient-match model to the disk.

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
            If a directory, the only checkpoint file `*.pth.tar` will be loaded.
            If a filepath, will load from this file.
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
            collate_fn=SiteSelectionBaseCollator(
                config={
                    'has_demographics': isinstance(train_data.sites, SiteBaseDemographics)
                    }
                ),
            )
        return dataloader

    def _build_model(self):
        self.model = BuildModel(
            trial_dim=self.config['trial_dim'],
            site_dim=self.config['site_dim'],
            embedding_dim=self.config['embedding_dim']
            )
    
    def _build_loss_model(self):
        if self.config['enrollment_only']:
            return PolicyGradientLossEnrollment(self.model, self.config['K'])
        else:
            return PolicyGradientLossCombined(self.model, self.config['K'], self.config['lambda'])     

    def _fit_model(self, train_data):
        train_dataloader = self.get_train_dataloader(train_data)
        loss_model = self._build_loss_model()
        train_objectives = [(train_dataloader, loss_model)]
        trainer = SiteSelectTrainer(
            model=self,
            train_objectives=train_objectives
            )
        trainer.train(**self.config)
        
    def _prepare_input(self, data):
        '''
        Prepare inputs to model.

        Parameters
        ----------
        data: dict[list]
            A batch of trials with their corresponding sites.
        '''
        inputs = {
            'trial': data['trial'].to(self.device),
            'site': data['site'].to(self.device),
            'label': data['label'].to(self.device),
            'eth_label': None if data['eth_label'] is None else data['eth_label'].to(self.device)
            }

        return inputs

    def _input_data_check(self, inputs):
        assert isinstance(inputs, TrialSiteSimple), f'`site_selection` models require input training data in `TrialSiteSimple`, find {type(inputs)} instead.'
