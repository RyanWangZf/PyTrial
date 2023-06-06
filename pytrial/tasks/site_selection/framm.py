from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pdb

import torch
from torch import nn
from torch.utils.data import DataLoader

from pytrial.utils.check import (
    check_checkpoint_file, check_model_dir, check_model_config_file, make_dir_if_not_exist
)
from .base import SiteSelectionBase
from .data import TrialSiteModalities, SiteSelectionModalitiesCollator
from .losses import PolicyGradientLossEnrollment, PolicyGradientLossCombined
from .trainer import SiteSelectTrainer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#######
### FRAMM Implementations
#######

class DiagnosisEncoder(nn.Module):
    '''
    Build a site's diagnosis history encoder.
    '''
    def __init__(self, 
        claim_dim, 
        lstm_dim, 
        embedding_dim, 
        num_layers):
        super(DiagnosisEncoder, self).__init__()
        
        self.claim_dim = claim_dim
        self.lstm_dim = lstm_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.biLSTM = nn.LSTM(claim_dim, lstm_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2*lstm_dim, embedding_dim)
        self.firstInput = nn.Parameter(torch.rand(1,1,1,claim_dim, dtype=torch.float))

    def forward(self, input, lengths, num_inv):
        bs = input.size(0)
        input = torch.cat((self.firstInput.repeat(bs, num_inv, 1, 1), input), -2)
        input = torch.reshape(input, (bs * num_inv, -1, self.claim_dim))
        lengths = torch.reshape(lengths, (bs * num_inv,))
        packed_input = pack_padded_sequence(input, lengths+1, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.biLSTM(packed_input)
        out, _ = pad_packed_sequence(packed_output, batch_first=True)
        combined_out = torch.cat((out[:, 0, self.lstm_dim:], out[range(len(out)), lengths, :self.lstm_dim]), 1)
        combined_out = torch.reshape(combined_out, (bs, num_inv, 2 * self.lstm_dim))
        return(self.fc(combined_out))

class PrescriptionEncoder(nn.Module):
    '''
    Build a site's prescription history encoder.
    '''
    def __init__(self, 
        claim_dim, 
        lstm_dim, 
        embedding_dim, 
        num_layers):
        super(PrescriptionEncoder, self).__init__()
        
        self.claim_dim = claim_dim
        self.lstm_dim = lstm_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.biLSTM = nn.LSTM(claim_dim, lstm_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2*lstm_dim, embedding_dim)
        self.firstInput = nn.Parameter(torch.rand(1,1,1,claim_dim, dtype=torch.float))

    def forward(self, input, lengths, num_inv):
        bs = input.size(0)
        input = torch.cat((self.firstInput.repeat(bs, num_inv, 1, 1), input), -2)
        input = torch.reshape(input, (bs * num_inv, -1, self.claim_dim))
        lengths = torch.reshape(lengths, (bs * num_inv,))
        packed_input = pack_padded_sequence(input, lengths+1, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.biLSTM(packed_input)
        out, _ = pad_packed_sequence(packed_output, batch_first=True)
        combined_out = torch.cat((out[:, 0, self.lstm_dim:], out[range(len(out)), lengths, :self.lstm_dim]), 1)
        combined_out = torch.reshape(combined_out, (bs, num_inv, 2 * self.lstm_dim))
        return(self.fc(combined_out))

class PastTrialEncoder(nn.Module):
    '''
    Build a site's trial history encoder.
    '''
    def __init__(self, 
        trial_dim, 
        lstm_dim, 
        embedding_dim, 
        num_layers):
        super(PastTrialEncoder, self).__init__()
        
        self.trial_dim = trial_dim
        self.lstm_dim = lstm_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.biLSTM = nn.LSTM(trial_dim+1-(2*768), lstm_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2*lstm_dim, embedding_dim)
        self.firstInput = nn.Parameter(torch.rand(1,1,1,trial_dim+1-(2*768), dtype=torch.float))

    def forward(self, input, lengths, num_inv):
        bs = input.size(0)
        input = torch.cat((self.firstInput.repeat(bs, num_inv, 1, 1), input), -2)
        input = torch.reshape(input, (bs * num_inv, -1, self.trial_dim+1-(2*768)))
        lengths = torch.reshape(lengths, (bs * num_inv,))
        packed_input = pack_padded_sequence(input, lengths+1, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.biLSTM(packed_input)
        out, _ = pad_packed_sequence(packed_output, batch_first=True)
        combined_out = torch.cat((out[:, 0, self.lstm_dim:], out[range(len(out)), lengths, :self.lstm_dim]), -1)
        combined_out = torch.reshape(combined_out, (bs, num_inv, 2 * self.lstm_dim))
        return(self.fc(combined_out))
  
class AttentionEncoder(nn.Module):
    '''
    Build a site's representation encoder which combines each of the possibly missing modalities.
    '''
    def __init__(self, 
        embed_dim, 
        n_keys, 
        n_heads):
        super(AttentionEncoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.attention = nn.MultiheadAttention(embed_dim, n_heads)#, batch_first=True)

    def forward(self, query, values, attention_mask):
        values = torch.stack(values, dim=2)
        bs = values.size(0)
        num_inv = values.size(1)
        query = torch.reshape(query, (bs * num_inv, 1, self.embed_dim)).transpose(0,1)
        values = torch.reshape(values, (bs * num_inv, -1, self.embed_dim)).transpose(0,1)
        attention_mask = torch.reshape(attention_mask, (bs * num_inv, -1)).bool()
        embeddings, _ = self.attention(query, values, values, attention_mask, need_weights=False)
        embeddings = torch.reshape(embeddings.transpose(0,1), (bs, num_inv, self.embed_dim))
        return embeddings
    
class ModalityDropoutEncoder(nn.Module):
    '''
    Build a site's representation encoder which combines each of the possibly missing modalities.
    '''
    def __init__(self, 
        embed_dim, 
        n_modalities, 
        n_heads):
        super(AttentionEncoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.n_modalities = n_modalities
        self.encoder = nn.Linear(n_modalities*embed_dim, embed_dim)

    def forward(self, query, values, attention_mask):
        for i in range(self.n_modalities):
            values[i] = values[i] * attention_mask[:,:,i]
        values = torch.cat(values, dim=-1)
        embeddings = self.encoder(values)
        return embeddings
    
class TransformerScoringNetwork(nn.Module):
    '''
    Generate score representations using a Transformer encoder layer
    '''
    def __init__(self, embedding_dim, n_heads, hidden_dim):
        encoder_layer = nn.TransformerEncoderLayer(2*embedding_dim, n_heads, hidden_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, 2)
    
    def forward(self, inv_site_representation):
        return self.encoder(inv_site_representation.transpose(0,1)).transpose(0,1)
  
class BuildModel(nn.Module):
    def __init__(self, 
        trial_dim, 
        static_dim, 
        dx_dim, 
        rx_dim, 
        lstm_dim, 
        embedding_dim, 
        num_layers, 
        hidden_dim, 
        n_heads,
        missing_type='MCAT',
        scoring_type='Transformer'):
        super(BuildModel).__init__()
        
        self.static_encoder = nn.Linear(static_dim, embedding_dim)
        self.diagnosis_encoder = DiagnosisEncoder(dx_dim, lstm_dim, embedding_dim, num_layers)
        self.prescription_encoder = PrescriptionEncoder(rx_dim, lstm_dim, embedding_dim, num_layers)
        self.history_encoder = PastTrialEncoder(trial_dim, lstm_dim, embedding_dim, num_layers)
        self.trial_encoder = nn.Linear(trial_dim, embedding_dim)
        self.stat_fc = nn.Linear(embedding_dim, embedding_dim)
        self.dx_fc = nn.Linear(embedding_dim, embedding_dim)
        self.rx_fc = nn.Linear(embedding_dim, embedding_dim)
        self.hist_fc = nn.Linear(embedding_dim, embedding_dim)
        self.trial_fc = nn.Linear(embedding_dim, embedding_dim)
        if missing_type == 'MCAT':
            self.missing_modality_encoder = AttentionEncoder(embedding_dim, 4, n_heads)
        elif missing_type == 'MD':
            self.missing_modality_encoder = ModalityDropoutEncoder(embedding_dim, 4)
        else:
            raise ValueError(f'Invalid missing_type: {missing_type}')
        if scoring_type == 'Transformer':
            self.score_encoder = TransformerScoringNetwork(embedding_dim, n_heads, hidden_dim)
        elif scoring_type == 'Fully Connected':
            self.score_encoder = nn.Linear(2*embedding_dim, 2*embedding_dim)
        else:
            raise ValueError(f'Invalid scoring_type: {scoring_type}')
        self.fc = nn.Linear(2*embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        
    def forward(self, inputs):
        trial = inputs['trial']
        investigators = inputs['inv_static']
        inv_dx = inputs['inv_dx']
        inv_dx_lens = inputs['inv_dx_len']
        inv_rx = inputs['inv_rx']
        inv_rx_lens = inputs['inv_rx_len']
        past_trials = inputs['inv_enroll']
        past_trials_lengths = inputs['inv_enroll_len']
        inv_mask = inputs['inv_mask']
        num_inv = investigators.size(1)
        # Trial is (bs, trial_dim)
        # All other inputs are (bs, M, *) where * is either a sequence or single input
        investigator_encoding = self.stat_fc(torch.relu(self.static_encoder(investigators)))
        dx_encoding = self.dx_fc(torch.relu(self.diagnosis_encoder(inv_dx, inv_dx_lens, num_inv)))
        rx_encoding = self.rx_fc(torch.relu(self.prescription_encoder(inv_rx, inv_rx_lens, num_inv)))
        history_encoding = self.hist_fc(torch.relu(self.history_encoder(past_trials, past_trials_lengths, num_inv)))
        trial_encoding = self.trial_fc(torch.relu(self.trial_encoder(trial)))
        trial_encoding = trial_encoding.unsqueeze(1).repeat(1, num_inv, 1)
        inv_representation = self.missing_modality_encoder(trial_encoding, [investigator_encoding, dx_encoding, rx_encoding, history_encoding], inv_mask)
        inv_site_representation = torch.cat((inv_representation, trial_encoding), dim=-1)
        network_input = self.score_encoder(inv_site_representation)
        score = self.output(torch.relu(self.fc(network_input)))
        return score

class FRAMM(SiteSelectionBase):
    '''
    Implement FRAMM for selecting clinical trial sites based on possibly missing multi-model site features [1]_.
    
    Parameters
    ----------
    
    trial_dim: list[int]
        Size of the trial representation

    static_dim: int
        Size of the static site features modality

    dx_dim: int
        Size of the diagnosis code vocabulary

    rx_dim: int
        Size of the prescription code vocabulary

    lstm_dim: int
        Size of the lstm embedding

    embedding_dim: int
        Size of all of the modality and other intermediate embeddings

    num_layers: int
        Number of lstm layers
    
    hidden_dim: int
        Size of intermediate representation within output layer

    n_heads: int
        Number of heads for attention encoder
        
    missing_type: string
        Type of missing data mechanism to use
        
    scoring_type: string
        Type of scoring network to use
    
    Notes
    -----
    .. [1] Theodorou, B., Glass, L., Xiao, C., & Sun, J. (2023). FRAMM: Fair Ranking with Missing Modalities for Clinical Trial Site Selection. arXiv preprint arXiv:2305.19407.
        
    '''
    def __init__(self, 
        trial_dim=211, 
        static_dim=124, 
        dx_dim=157, 
        rx_dim=79, 
        lstm_dim=64, 
        embedding_dim=64, 
        num_layers=2, 
        hidden_dim=32, 
        n_heads=4,
        missing_type='MCAT',
        scoring_type='Transformer',
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
        ):
        super(SiteSelectionBase, self).__init__(experiment_id)
        self.config = {
            'trial_dim':trial_dim,
            'static_dim':static_dim,
            'dx_dim':dx_dim,
            'rx_dim':rx_dim,
            'lstm_dim':lstm_dim,
            'embedding_dim':embedding_dim,
            'num_layers':num_layers,
            'hidden_dim':hidden_dim,
            'n_heads':n_heads,
            'missing_type':missing_type,
            'scoring_type':scoring_type,
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
        self._build_model()

    def fit(self, train_data):
        '''
        Train model with historical trial-site enrollments.

        Parameters
        ----------
        train_data: TrialSiteModalities
            A `TrialSiteModalities` contains trials, sites, and enrollments.
        '''
        self._input_data_check(train_data)
        self._fit_model(train_data)

    def predict(self, test_data):
        self.model(**test_data) # TODO

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
            collate_fn=SiteSelectionModalitiesCollator(
                config={
                    'visit_mode':train_data.sites.metadata['visit']['mode'],
                    'trial_mode':train_data.sites.metadata['hist']['mode'],
                    'has_demographics':train_data.sites.get_label() is not None
                    }
                ),
            )
        return dataloader

    def _build_model(self):
        self.model = BuildModel(
            trial_dim=self.config['trial_dim'],
            static_dim=self.config['static_dim'],
            dx_dim=self.config['dx_dim'],
            rx_dim=self.config['rx_dim'],
            lstm_dim=self.config['lstm_dim'],
            embedding_dim=self.config['embedding_dim'],
            num_layers=self.config['num_layers'],
            hidden_dim=self.config['hidden_dim'],
            n_heads=self.config['n_heads'],
            missing_type=self.config['missing_type'],
            scoring_type=self.config['scoring_type']
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
            'label': data['label'].to(self.device),
            'eth_label': None if data['eth_label'] is None else data['eth_label'].to(self.device),
            'inv_static': data['inv_static'].to(self.device),
            'inv_dx': data['inv_dx'].to(self.device),
            'inv_dx_len': data['inv_dx_len'].to('cpu'),
            'inv_rx': data['inv_rx'].to(self.device),
            'inv_rx_len': data['inv_rx_len'].to('cpu'),
            'inv_enroll': data['inv_enroll'].to(self.device),
            'inv_enroll_len': data['inv_enroll_len'].to('cpu'),
            'inv_mask': data['inv_static'].to(self.device),
            }

        return inputs

    def _input_data_check(self, inputs):
        assert isinstance(inputs, TrialSiteModalities), f'`site_selection` models require input training data in `SiteSelectionBase`, find {type(inputs)} instead.'