'''
Implement SPOT model for clinical trial outcome prediction.
'''
import pdb
import os
import pickle
import math
import json
import wget
from collections import defaultdict

import numpy as np
import torch
from torch import nn
import torch.nn.init as nn_init
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from pytrial.utils.trainer import Trainer
from .base import TrialOutcomeBase
from .model_utils.topic_discovery import TrialTopicDiscovery
from .model_utils.molecule_encode import MPNN, ADMET
from .model_utils.icdcode_encode import GRAM
from .model_utils.module import Highway
from .model_utils.data_utils import collect_trials_from_topic
from .evaluation import evaluate



__all__ = ['SPOT']

class constants:
    YEAR_RANGE = [1950, 2050]
    ICDCODE_ANCESTOR_FILENAME = './demo_data/icdcode_data/icdcode2ancestor_dict.pkl'
    ICDCODE_LIST_FILENAME = './demo_data/icdcode_data/raw_data.csv'
    RAW_DATA_FILENAME = './demo_data/icdcode_data/raw_data.csv'
    INC_EMB_FILENAME = './demo_data/criteria_data/nct2inc_embedding.pkl'
    EXC_EMB_FILENAME = './demo_data/criteria_data/nct2exc_embedding.pkl'
    WEIGHT_NAME = 'pytorch_model.bin'
    CONFIG_NAME = 'config.json'
    TOPIC_DISCOVER_NAME = 'topic_discover.pkl'
    MODEL_OBJECT_NAME = 'spot_model.pkl'
    SEQDATA_NAME = 'seqtrial_data.pkl'
    TRAINING_PARAMETERS = ['batch_size','learning_rate','epochs','weight_decay','warmup_ratio','evaluation_steps','seed']
    # used for downloading pretrained embeddings / preprocessed data
    BENCHMARK_DATA_URL = 'https://storage.googleapis.com/pytrial/HINT-benchmark-data/hint_benchmark_dataset_w_date.zip'
    ICDCODE_ANCESTOR_URL = 'https://storage.googleapis.com/pytrial/HINT-benchmark-data/icdcode2ancestor_dict.pkl'
    INC_EMB_URL = 'https://storage.googleapis.com/pytrial/HINT-benchmark-data/nct2inc_embedding.pkl'
    EXC_EMB_URL = 'https://storage.googleapis.com/pytrial/HINT-benchmark-data/nct2exc_embedding.pkl'
    ADMET_MODEL_URL = 'https://storage.googleapis.com/pytrial/HINT-benchmark-data/admet_model_state_dict.ckpt'

def check_nan(inputs):
    # check nan for debugging
    if torch.isnan(inputs).any():
        pdb.set_trace()

class SPOTCollator:
    def __call__(self, inputs):
        '''
        Parameters
        ----------
        inputs: List[Topic]
            A list of `Topic` objects.
        '''
        return_data = defaultdict(list)
        return_data['topic'] = inputs
        # create mask list, (bs, n_ts, max_n_trial)
        for input in inputs:
            mask = []
            for trial_step in input.trials:
                mask.append(torch.tensor([1] * len(trial_step)))
            # pad mask
            mask = pad_sequence(mask, batch_first=True, padding_value=0)
            return_data['mask'].append(mask)
        return return_data

class MoleculeEncoder(nn.Module):
    '''
    Load pretrained MPNN and ADMET models.
    '''
    def __init__(self, input_dir=None, device='cuda:0'):
        super(MoleculeEncoder, self).__init__()
        if input_dir is None:
            input_dir = './pretrained_model/molecule_encoder'
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)
            # download pretrained model if not exist from the url ADMET_MODEL_URL
            self._download_pretrained(input_dir)
        
        # load pretrained admet model and mpnn model
        state_dict = torch.load(os.path.join(input_dir, 'admet_model_state_dict.ckpt'))
        self.mpnn = MPNN(mpnn_hidden_size=50, mpnn_depth=3, device=device)
        self.admet = ADMET(molecule_encoder=self.mpnn,
            highway_num=2,
            device=device,
            epoch=None,
            lr=None,
            weight_decay=None,
            save_name=None,
            )
        self.admet.load_state_dict(state_dict)

        # freeze admet model
        # for param in self.admet.parameters():
        #     param.requires_grad = False
             
        self.pk_fc = nn.Linear(50 * 5, 50)
        self.pk_highway = Highway(50, 2)
    
    def forward(self, smiles):
        # get molecule features
        res = self.mpnn.forward_smiles_lst(smiles) # (n_molecule, 50)
        # if w/o admet, return mpnn features
        return res

    def _download_pretrained(self, input_dir):
        wget.download(constants.ADMET_MODEL_URL, input_dir)

class DiseaseEncoder(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        self.gram = GRAM(
            embedding_dim=50, 
            icdcode2ancestor=None, 
            device=device,
        )
        self.device = device
    
    def forward(self, icdcode):
        # in one sample, icdcode is a list of list of icdcode belonging to the same trial
        res = self.gram.forward_code_lst2(icdcode)
        return res

class ProtocolEncoder(nn.Module):
    def __init__(self, output_dim=50) -> None:
        super().__init__()
        self.nct2incemb = nn.ParameterDict({})
        self.nct2excemb = nn.ParameterDict({})
        self.fc = nn.Linear(768, output_dim)
        self.activation = nn.ReLU()

    def forward(self, nctid):
        '''
        Get the pre-encoded protocol embeddings.
        '''
        device = self.fc.weight.device
        # say if nctid in the dict, return the embedding
        # else return an all zero embedding
        if nctid in self.nct2incemb:
            inc_emb = self.nct2incemb[nctid][None].to(device)
        else:
            inc_emb = torch.zeros(1, 768).to(device)

        if nctid in self.nct2excemb:
            exc_emb = self.nct2excemb[nctid][None].to(device)
        else:
            exc_emb = torch.zeros(1, 768).to(device)

        emb = torch.cat([inc_emb, exc_emb], dim=0) # [2, 768]
        emb = self.fc(emb)
        emb = self.activation(emb) # [2, 50]
        return emb
    
    def update_criteria_emb(self, inc_emb_dict=None, exc_emb_dict=None):
        '''
        Update the nct2incemb and nct2excemb dict.
        '''
        if inc_emb_dict is not None:
            self.nct2incemb.update(inc_emb_dict)
            self._disable_grad(self.nct2incemb)
                
        if exc_emb_dict is not None:
            self.nct2excemb.update(exc_emb_dict)
            self._disable_grad(self.nct2excemb)

    def _disable_grad(self, nct2incemb):
        for k,v in nct2incemb.items():
            nct2incemb[k] = v.requires_grad_(False)
        return nct2incemb

class FeatureEncoder(nn.Module):
    def __init__(self, input_dim, output_dim=50):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        nn_init.uniform_(self.fc.weight, a=-1/math.sqrt(output_dim),b=1/math.sqrt(output_dim))

        # assign each year an embedding, drawback is that some years are not available in the training data
        self.embed = nn.Embedding(150, output_dim)
        nn_init.uniform_(self.embed.weight, a=-1/math.sqrt(output_dim),b=1/math.sqrt(output_dim))

    def forward(self, x):
        '''
        Currently only encode year.
        Args:
            x: int
        '''
        # assign each year an embedding, drawback is that some years are not available in the training data
        # x = torch.tensor(x-1900).long().to(self.embed.weight.device)
        # x = self.embed(x)[None]

        # x maps to [0, 1], assume x is in [1950, 2050]
        start, end = constants.YEAR_RANGE
        x = torch.tensor((x - start)/(end-start), device=self.fc.weight.device).unsqueeze(0)
        x = self.fc(x)[None]
        return x

class TopicEncoder(nn.Module):
    def __init__(self, num_topics, output_dim=50):
        super().__init__()
        self.embed = nn.Embedding(num_topics, output_dim)
        nn_init.uniform_(self.embed.weight, a=-1/math.sqrt(output_dim),b=1/math.sqrt(output_dim))

    def forward(self, x):
        '''
        Currently only encode a single topic id in str or int.
        '''
        if isinstance(x, str):
            x = int(x)
        x = torch.tensor(x, device=self.embed.weight.device)
        x = self.embed(x)[None]
        return x


class SPOTTrainer(Trainer):
    '''
    Make supervised sequential modeling (no meta learning).
    '''
    def prepare_input(self, data):
        return self.model.prepare_input(data, return_label=True)
    
    def evaluate(self):
        res = self.model._predict_on_dataloader(self.test_dataloader)
        pred, label, nctid = res['pred'], res['label'], res['nctid']
        bool_idx = np.isin(nctid, self.test_data.new_trials[-1]) # 0: valids, 1: testids

        eval_label = label[bool_idx]
        eval_pred = pred[bool_idx]

        eval_res = evaluate(eval_pred, eval_label, find_best_threshold=True)
        return eval_res
    
    def get_test_dataloader(self, test_data):
        '''
        Build a dataloader for testing a spot model.
        Parameters
        ----------
        test_data: SequenceTrial
            A dataset consisting of multiple topics of clinical trials. Each topic is a list of clinical trials under this topic ordered with their timestamps.
        '''
        self.test_data = test_data
        loader = self.model.get_test_dataloader(test_data)
        return loader

    def train_one_iteration(self, 
        max_grad_norm=None,
        warmup_steps=None,
        use_amp=None, 
        scaler=None,
        train_loss_dict=None):
        '''
        Default training one iteration steps, can be subclass can reimplemented.
        '''        
        skip_scheduler = False
        num_train_objectives = len(self.train_dataloader)
        for train_idx in range(num_train_objectives):
            data_iterator = self.data_iterators[train_idx]
            loss_model = self.loss_models[train_idx]
            loss_model.zero_grad()
            loss_model.train()
            optimizer = self.optimizers[train_idx]
            scheduler = self.schedulers[train_idx]

            # get a batch of data from the target data_iterator
            data = self._get_a_train_batch(data_iterator=data_iterator, train_idx=train_idx)
            data = self.prepare_input(data)

            # update model by backpropagation
            if use_amp:
                loss_value, skip_scheduler, scale_before_step = self._update_one_iteration_amp(loss_model=loss_model, data=data, optimizer=optimizer, scaler=scaler, max_grad_norm=max_grad_norm)
            else:
                loss_value = self._update_one_iteration(loss_model=loss_model, data=data, optimizer=optimizer, max_grad_norm=max_grad_norm)                

            train_loss_dict[train_idx].append(loss_value.item())

            if use_amp:
                skip_scheduler = scaler.get_scale() != scale_before_step

            if not skip_scheduler and warmup_steps > 0:
                scheduler.step()

    def _update_one_iteration(self, loss_model, data, optimizer, max_grad_norm):
        loss_model_return = loss_model(data)
        loss_value = loss_model_return['loss_value']
        loss_value.backward()

        print('loss_value', loss_value.item())

        # check if nan in the gradients of all modules in the loss_model
        # for debugging
        # for i, (name, param) in enumerate(loss_model.named_parameters()):
        #     if param.grad is not None:
        #         if torch.isnan(param.grad).any():
        #             pdb.set_trace()
        #             raise ValueError(f'Gradient of {name} is nan.')

        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
        optimizer.step()
        return loss_value

    def _update_one_iteration_amp(self, loss_model, data, optimizer, scaler, max_grad_norm):
        with autocast():
            loss_return = loss_model(data)
        loss_value = loss_return['loss_value']
        scale_before_step = scaler.get_scale()
        scaler.scale(loss_value).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        skip_scheduler = scaler.get_scale() != scale_before_step
        return loss_value, skip_scheduler, scale_before_step



class SPOTModel(nn.Module):
    '''
    SPOT model that encodes the input data and predicts the trial outcome.
    
    Parameters
    ----------
    checkpoint: str
        The path to the checkpoint of the pretrained SPOT model.
    '''
    def __init__(self, 
        num_topics,
        n_trial_projector=2,
        n_timestemp_projector=2,
        n_rnn_layer=1,
        n_trial_per_batch=None,
        checkpoint='./pretrained_model', 
        device='cuda:0',
        **kwargs) -> None:
        super().__init__()
        self.num_topics = num_topics
        self.n_trial_per_batch = n_trial_per_batch

        self.molecule_encoder = MoleculeEncoder(input_dir=os.path.join(checkpoint, 'molecule_encoder'),
                                                    device=device)
        self.disease_encoder = DiseaseEncoder(device=device)
        self.protocol_encoder = ProtocolEncoder()
        self.feature_encoder = FeatureEncoder(1, 50)
        self.topic_encoder = TopicEncoder(num_topics, 50)

        self.rnn = nn.RNN(50, 50, num_layers=n_rnn_layer, batch_first=True, bidirectional=False)

        self.spatial_interaction = nn.MultiheadAttention(50, 1, dropout=0, bias=False, batch_first=True)
        
        self.trial_projector = Highway(50, n_trial_projector)
        self.timestep_projector = Highway(50, n_timestemp_projector)

        self.predictor = nn.Sequential(nn.ReLU(), nn.Linear(100, 50), nn.Linear(50, 1))
        self.activation = nn.ReLU()

        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')


        self.to(device)

    def forward(self, inputs, return_loss=True):
        '''
        inputs: dict
            {
                'topic': List[List[torch.Tensor]], # [batch_size, n_timestep, n_trials, emd_dim]
                'label':  List[torch.Tensor], # [batch_size, n_timestep, n_trials]
                'mask': List[torch.Tensor], # [batch_size, n_timestep, n_trials]
            }
        '''
        outputs = defaultdict(list)
        outputs['loss_value'] = 0
        for i in range(len(inputs['topic'])):
            topic_embs = pad_sequence(inputs['topic'][i], batch_first=True, padding_value=0)
            mask = inputs['mask'][i]
            mask = mask.to(topic_embs.device)
            
            # propogate historical information
            hidden_embs = topic_embs.sum(1) / mask.sum(1)[:,None] # [num_timestep, hidden_dim]
            hidden_embs = hidden_embs[None]
            hidden_embs = self.rnn(hidden_embs)[0]  # [1, num_timestep, hidden_dim]
            hidden_embs = hidden_embs.permute(1,0,2) # [num_timestep, 1, hidden_dim]

            # interaction across trials at the same timestep with the temporal embedding
            interact_embs = torch.cat([hidden_embs, topic_embs], dim=1)
            iteract_mask = torch.cat([torch.ones(mask.shape[0],1).to(mask.device), mask], 1)
            sp_hidden_embs = self.spatial_interaction(interact_embs, interact_embs, interact_embs, key_padding_mask=iteract_mask.float())[0] # [num_timestep, max_num_trials+1, hidden_dim]
            hidden_embs = sp_hidden_embs[:,1:,...] # remove the hidden state of the cls token (the first token is the temporal embedding)

            # concatenate
            hidden_embs = torch.cat([hidden_embs, topic_embs], dim=-1) # [num_timestep, max_num_trials, hidden_dim*2]

            # make predictions
            pred_logits = self.predictor(hidden_embs) # [num_timestep, max_num_trials, 1]
            outputs['pred_logits'].append(pred_logits[mask==1]) # get corresponding logits
            if 'label' in inputs and return_loss:
                label = inputs['label'][i]
                label = label.to(pred_logits.device)
                mask = mask.to(pred_logits.device)
                loss = self.compute_loss(pred_logits, label, mask)
                outputs['loss'].append(loss)
                outputs['label'].append(label[mask==1])

        if return_loss:
            all_losses = outputs.pop('loss')
            all_losses = torch.cat(all_losses, dim=0)
            if self.training and self.n_trial_per_batch is not None:
                # sampling trials for training
                n_trial = min(all_losses.shape[0], self.n_trial_per_batch)
                idx = torch.randperm(all_losses.shape[0])[:n_trial]
                outputs['loss_value'] = all_losses[idx].mean()
            else:
                outputs['loss_value'] = all_losses.mean()

        return outputs

    def prepare_input(self, inputs, return_label=True):
        '''
        transform the input data into the format that can be fed into the SPOT model
        generate the embeddings of molecule, disease, protocol serving for the sequential predictive model
        return labels for each topic
        return nctids for each topic

        Parameters
        ----------
        inputs: dict
            {
                'topic': List[List[Topic]],
                'mask': List[List[torch.Tensor]],
            }
        '''
        outputs = defaultdict(list)
        for i in range(len(inputs['topic'])):
            topic = inputs['topic'][i]
            topic_emb = self.embed_topic(topic) # [num_timestep, num_trials, emb_dim]
            outputs['topic'].append(topic_emb)
            outputs['topic_id'].append(topic.topic_id)
            if return_label:
                label = [torch.tensor([t_.attr_dict['label'] for t_ in t]) for t in inputs['topic'][i].trials]
                label = pad_sequence(label, batch_first=True, padding_value=0)
                outputs['label'].append(label)
        outputs['mask'] = inputs['mask']
        return outputs
    
    def embed_topic(self, topic):
        # embed the topic into the topic embedding
        # topic: [num_timestep, num_trials, emb_dim]
        topic_embs = []
        for trials in topic.trials: 
            ts_embs = []
            for trial in trials:
                # embed the molecule, disease, protocol
                trial_emb = self.embed_trial(trial) # [1, 50]
                # check if isnan in the embedding
                if torch.isnan(trial_emb).any():
                    pdb.set_trace()

                topic_emb = self.topic_encoder(topic.topic_id) # [1, 50]
                ts_embs.append(trial_emb+topic_emb)
            ts_embs = torch.cat(ts_embs, dim=0) # [num_trials, 50]
            ts_embs = self.timestep_projector(ts_embs) # [num_trials, 50]
            topic_embs.append(ts_embs)

        return topic_embs # [num_timestep, num_trials, 50]
    
    def embed_trial(self, trial):
        # embed the trial into the trial embedding
        # trial: Trial object
        mol_emb = self.molecule_encoder(trial.attr_dict['smiless']) # [1, 50] molecule + [1, 50] pharamakinetics
        # check if isnan in the embedding
        if torch.isnan(mol_emb).any():
            pdb.set_trace()        
        protocol_emb = self.protocol_encoder(trial.attr_dict['nctid']) # [2, 50]
        if torch.isnan(protocol_emb).any():
            pdb.set_trace()
        disease_emb = self.disease_encoder(trial.attr_dict['icdcodes']) # [1, 50]
        if torch.isnan(disease_emb).any():
            pdb.set_trace()
        year_emb = self.feature_encoder(trial.attr_dict['year']) # [1, 50]

        trial_emb = torch.cat([mol_emb, protocol_emb, disease_emb, year_emb], dim=0) # [4, 50]

        # add CLS token embedding
        trial_emb = torch.unsqueeze(trial_emb, dim=0) # [1, 4, 50]
        # trial_emb = self.cls_encoder(trial_emb)['embedding'] # [1, 5, 50]

        # highway network
        trial_emb = self.trial_projector(trial_emb) # [1, 5, 50]
        # return trial_emb[:,0,:] # get CLS embedding [1, 50]
        return trial_emb.mean(dim=1) # get mean embedding [1, 50]
    
    def compute_loss(self, pred_logits, label, mask):
        # compute the loss
        # pred_logits: [num_timestep, max_num_trials, 1]
        # label: [num_timestep, max_num_trials]
        # mask: [num_timestep, max_num_trials]
        loss = self.loss_fn(pred_logits, label.float().unsqueeze(-1))
        loss = loss * mask.unsqueeze(-1)

        # flatten the tensor of the loss
        return loss[loss!=0]

    def feed_lst_of_module(self, input_feature, lst_of_module):
        x = input_feature
        for single_module in lst_of_module:
            x = self.activation(single_module(x))
        return x

    def init_criteria_embedding(
        self, 
        input_data,
        criteria_column="criteria"
        ):
        # check if the criteria column in the input data
        assert criteria_column in input_data.df.columns, f"{criteria_column} is not in the input data."

        # initialize the criteria embedding
        _ = input_data.get_ec_sentence_embedding(criteria_column)
        # build nctid2criteria_emb assigning to self.protocol_encoder
        nct2emb = input_data.get_nct_to_ec_emb()
        self.protocol_encoder.update_criteria_emb(
            inc_emb_dict=nct2emb['nct2incemb'],
            exc_emb_dict=nct2emb['nct2excemb'],
            )

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class SPOT(TrialOutcomeBase):
    """
    Implement sequential predictive modeling of clinical trial outcome with meta-learning (SPOT) [1]_.

    Parameters
    ----------
    num_topics: int
        Number of topics to discover.
    
    n_trial_projector: int
        Number of layers in the trial projector.
    
    n_timestemp_projector: int
        Number of layers in the timestamp projector.
    
    n_rnn_layer: int
        Number of layers in the RNN for encoding ontology & smiles strings.
        
    criteria_column: str
        The column name of the criteria in the input data.

    batch_size: int
        Batch size for training.
    
    n_trial_per_batch: int
        Number of trials in each batch. If None, use all trials in the topic in each batch.
    
    learning_rate: float
        Learning rate for training.
    
    weight_decay: float
        Regularization strength for l2 norm; must be a positive float.
    
    epochs: int
        Number of training epochs.
    
    evaluation_steps: int
        Number of steps to evaluate the model.
    
    warmup_ratio: float
        Warmup ratio for learning rate scheduler.
    
    device: str
        Device to use for training and inference.

    seed: int
        Random seed.

    Notes
    -----
    .. [1] Wang, Z., Xiao, C., & Sun, J. (2023). SPOT: Sequential Predictive Modeling of Clinical Trial Outcome with Meta-Learning. arXiv preprint arXiv:2304.05352.
    """
    def __init__(self,
        num_topics=50,
        n_trial_projector=2,
        n_timestemp_projector=2,
        n_rnn_layer=1,
        criteria_column='criteria',
        batch_size=1,
        n_trial_per_batch=None,
        learning_rate=1e-4,
        weight_decay=1e-4,
        epochs=10,
        evaluation_steps=50,
        warmup_ratio=0,
        device="cuda:0",
        seed=42,
        output_dir="./checkpoints/spot",
        ):
        self.config = {
            'num_topics': num_topics,
            'n_trial_projector': n_trial_projector,
            'n_timestemp_projector': n_timestemp_projector,
            'n_rnn_layer': n_rnn_layer,
            'criteria_column': criteria_column,
            'batch_size': batch_size,
            'n_trial_per_batch':n_trial_per_batch,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'weight_decay': weight_decay,
            'evaluation_steps':evaluation_steps,
            'warmup_ratio':warmup_ratio,
            'seed': seed,
            'output_dir': output_dir,
            'device': device,
            }
        self.model = SPOTModel(**self.config)
        self.topic_discover = TrialTopicDiscovery(num_topics=num_topics, random_seed=seed)
        self.training = False
        print(self.config)

    def fit_topic(self, df_train):
        '''
        Discover topics from the training data.

        Parameters
        ----------
        df_train: pandas.DataFrame
            A dataframe containing the training data.
        '''
        self.topic_discover.fit(df_train)

    def transform_topic(self, df, df_val=None, df_test=None):
        '''
        Transform the data into a SequenceTrial object.
        
        Parameters
        ----------
        df: pandas.DataFrame
            A dataframe containing the training data.
        
        df_val: pandas.DataFrame
            A dataframe containing the validation data.
        
        df_test: pandas.DataFrame
            A dataframe containing the test data.
        '''

        seqtrain = self.topic_discover.transform(df)
        seqval = None
        seqtest = None
        if df_val is not None:
            seqval = self.topic_discover.add_trials(seqtrain, df_val)
            if df_test is not None:
                seqtest = self.topic_discover.add_trials(seqval, df_test)        
        return {'seqtrain':seqtrain, 'seqval':seqval, 'seqtest':seqtest}

    def add_trials(self, seqtrain, df):
        '''Add new trials to the SequenceTrial object. It is used for the validation and test data.

        Parameters
        ----------
        seqtrain: model_utils.data_structure.SequenceTrial
            The sequence of clinical trials represented by SequenceTrial object.

        df: pd.DataFrame
            The dataframe of new trials that should be added to the input seqtrain.
        '''
        return self.topic_discover.add_trials(seqtrain, df)

    def fit(self, train_data, valid_data=None):
        '''Train the SPOT model. It has two/three steps:
        1. Encode the criteria using pretrained BERT to get the criteria embedding (optional).
        2. Discover topics from the training data and transform the data into a SequenceTrial object.
        3. Train the SPOT model.

        Parameters
        ----------
        train_data: TrialOutcomeDataset
            Training data, should be a `TrialOutcomeDataset` object.

        valid_data: TrialOutcomeDataset
            Validation data, should be a `TrialOutcomeDataset` object.
        '''
        # initialize the criteria embedding using pretrained language models
        self.model.init_criteria_embedding(train_data, self.config['criteria_column'])
        if valid_data is not None:
            # will add the criteria embedding for the validation data
            self.model.init_criteria_embedding(valid_data, self.config['criteria_column'])

        # initialize the topics using the topic discovery model
        self.fit_topic(train_data.df)

        seqdatas = self.transform_topic(train_data.df, valid_data.df if valid_data is not None else None)
        train_seq_data = seqdatas['seqtrain']
        valid_seq_data = seqdatas['seqval'] if valid_data is not None else None
        self._fit_model(train_seq_data, valid_seq_data)

        # save the seqdata
        self.seqdata = train_seq_data if valid_seq_data is None else valid_seq_data

    def predict(self, data, target_trials=None):
        '''
        Predict the outcome of a clinical trial.

        Parameters
        ----------
        data: TrialOutcomeDataset
            A `TrialOutcomeDataset` object containing the data to predict.
        
        target_trials: list
            A list of trial ids to predict. If None, all trials in `data` will be predicted.
        '''
        self.eval()
        if target_trials is not None:
            target_trials = data.df['nctid'].tolist()

        # need to initialize the criteria embedding for the test data
        self.model.init_criteria_embedding(data, self.config['criteria_column'])

        # add test data to the seqtrain stored in the model
        seqtest = self.add_trials(self.seqdata, data.df)

        loader = self.get_test_dataloader(seqtest)
        result = self._predict_on_dataloader(loader)
        pred, label, nctid = result['pred'], result['label'], result['nctid']
        if target_trials is not None:
            bool_idx = np.isin(nctid, target_trials)
            nctid = np.array(nctid)
            result = {'pred': pred[bool_idx], 'label': label[bool_idx], 'nctid': nctid[bool_idx]}
        
        return result

    def save_model(self, output_dir=None):
        '''
        Save the model to a directory.

        Parameters
        ----------
        output_dir: str or None
            The directory to save the model.
            If None, use the default directory './checkpoints'.
        '''
        if output_dir is None:
            output_dir = './checkpoints'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # save the model object into pickle
        with open(os.path.join(output_dir, constants.MODEL_OBJECT_NAME), 'wb') as f:
            pickle.dump(self.model, f)

        # save config
        with open(os.path.join(output_dir, constants.CONFIG_NAME), 'w') as f:
            f.write(json.dumps(self.config))

        # save topic discover
        with open(os.path.join(output_dir, constants.TOPIC_DISCOVER_NAME), 'wb') as f:
            pickle.dump(self.topic_discover, f)

        # save the seqdata object
        with open(os.path.join(output_dir, constants.SEQDATA_NAME), 'wb') as f:
            pickle.dump(self.seqdata, f)

        print('save model to {}'.format(output_dir))

    def load_model(self, input_dir=None):
        '''
        Load the model from a directory.

        Parameters
        ----------
        input_dir: str or None
            The directory to load the model.
        
        '''
        if input_dir is None:
            raise ValueError('Please specify the input directory.')

        self._load_config(input_dir)

        # load model object
        with open(os.path.join(input_dir, constants.MODEL_OBJECT_NAME), 'rb') as f:
            self.model = pickle.load(f)

        # load topic discover
        with open(os.path.join(input_dir, constants.TOPIC_DISCOVER_NAME), 'rb') as f:
            self.topic_discover = pickle.load(f)
        
        print('load model from {}'.format(input_dir))

    def get_train_dataloader(self, train_data):
        '''
        Build a dataloader for training a spot model.
        Parameters
        ----------
        train_data: SequenceTrial
            A dataset consisting of multiple topics of clinical trials. Each topic is a list of clinical trials under this topic ordered with their timestamps.
        '''
        loader = DataLoader(train_data, batch_size=self.config['batch_size'], shuffle=True, collate_fn=SPOTCollator())
        return loader
    
    def get_test_dataloader(self, test_data):
        '''
        Build a dataloader for testing a spot model.
        Parameters
        ----------
        test_data: SequenceTrial
            A dataset consisting of multiple topics of clinical trials. Each topic is a list of clinical trials under this topic ordered with their timestamps.
        '''
        loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=SPOTCollator())
        return loader
    
    def train(self, mode=True):
        self.training = mode
        self.model.train(mode)
        return self
    
    def eval(self, mode=False):
        self.training = mode
        self.model.eval()
        return self
    
    def prepare_input(self, data, return_label=False):
        return self.model.prepare_input(data, return_label=return_label)
    
    def _fit_model(self, train_data, valid_data):
        trainloader = self.get_train_dataloader(train_data)
        train_objectives = [(trainloader, self.model)]
        trainer = SPOTTrainer(
            model = self,
            train_objectives = train_objectives,
            test_data=valid_data,
            test_metric='pr_auc',
            output_dir=self.config['output_dir'],
            )
        trainer.train(**self.config)
    
    @torch.no_grad()
    def _predict_on_dataloader(self, test_dataloader):
        self.model.eval()
        pred_list, label_list, nctid_list = [], [], []
        for batch in test_dataloader:
            nctids = [collect_trials_from_topic(x) for x in batch['topic']]
            nctid_list.extend(nctids)
            inputs = self.prepare_input(batch, return_label=True)
            outputs = self.model(inputs)
            pred_list.append(torch.cat(outputs['pred_logits'], dim=0).cpu())
            label_list.append(torch.cat(outputs['label'], dim=0).cpu())

        pred = torch.cat(pred_list, dim=0)
        pred = torch.sigmoid(pred).numpy()
        label = torch.cat(label_list, dim=0).numpy()
        nctid_list = sum(nctid_list, [])
        return {'pred':pred, 'label':label, 'nctid':nctid_list}
    
    def _load_config(self, input_dir):
        with open(os.path.join(input_dir, constants.CONFIG_NAME), 'r') as f:
            config = json.loads(f.read())
        for k in constants.TRAINING_PARAMETERS:
            if k in config: config.pop(k)
        self.config.update(config)


