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

import torch
from torch import nn
import torch.nn.init as nn_init
from torch.nn.utils.rnn import pad_sequence

from .base import TrialOutcomeBase
from .model_utils.topic_discovery import TrialTopicDiscovery
from .model_utils.molecule_encode import MPNN, ADMET
from .model_utils.icdcode_encode import GRAM
from .model_utils.module import Highway
from .model_utils.protocol_encode import save_trial_criteria_bert_dict_pkl


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
    TRAINING_PARAMETERS = ['batch_size','learning_rate','epochs','weight_decay','warmup_ratio','evaluation_steps','seed']
    # used for downloading pretrained embeddings / preprocessed data
    BENCHMARK_DATA_URL = 'https://storage.googleapis.com/pytrial/HINT-benchmark-data/hint_benchmark_dataset_w_date.zip'
    ICDCODE_ANCESTOR_URL = 'https://storage.googleapis.com/pytrial/HINT-benchmark-data/icdcode2ancestor_dict.pkl'
    INC_EMB_URL = 'https://storage.googleapis.com/pytrial/HINT-benchmark-data/nct2inc_embedding.pkl'
    EXC_EMB_URL = 'https://storage.googleapis.com/pytrial/HINT-benchmark-data/nct2exc_embedding.pkl'
    ADMET_MODEL_URL = 'https://storage.googleapis.com/pytrial/HINT-benchmark-data/admet_model_state_dict.ckpt'

class MoleculeEncoder(nn.Module):
    '''
    Load pretrained MPNN and ADMET models.
    '''
    def __init__(self, input_dir=None, device='cuda:0'):
        super(MoleculeEncoder, self).__init__()
        if input_dir is None:
            input_dir = './pretrained_models/molecule_encoder'
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
            embedding_dim=50, icdcode2ancestor=None, device=device,
        )
        self.device = device
    
    def forward(self, icdcode):
        # in one sample, icdcode is a list of list of icdcode belonging to the same trial
        res = self.gram.forward_code_lst2(icdcode)
        return res

class ProtocolEncoder(nn.Module):
    def __init__(self, output_dim=50) -> None:
        super().__init__()
        self.nct2incemb = None
        self.nct2excemb = None
        self.fc = nn.Linear(768, output_dim)
        self.activation = nn.ReLU()

    def forward(self, nctid):
        '''
        Get the pre-encoded protocol embeddings.
        '''
        device = self.fc.weight.device
        inc_emb = self.nct2incemb[nctid][None].to(device)
        exc_emb = self.nct2excemb[nctid][None].to(device)
        emb = torch.cat([inc_emb, exc_emb], dim=0) # [2, 768]
        emb = self.fc(emb)
        emb = self.activation(emb) # [2, 50]
        return emb

    def init_criteria_emb(self, input_dir): 
        '''need to be called before fit the main model.
        '''

        pdb.set_trace()

        if not os.path.exists(input_dir):
            os.makedirs(input_dir)
            # get the nct2criteria embeddings dict
            save_trial_criteria_bert_dict_pkl()
        
        # load the nct2inc and nct2exc embeddings dict
        nct2incemb = pickle.load(open(constants.INC_EMB_FILENAME, 'rb'))
        nct2excemb = pickle.load(open(constants.EXC_EMB_FILENAME, 'rb'))
        
        for k,v in nct2excemb.items(): nct2excemb[k] = torch.tensor(v)
        for k,v in nct2incemb.items(): nct2incemb[k] = torch.tensor(v)

        self.nct2incemb = self._disable_grad(nn.ParameterDict(nct2incemb))
        self.nct2excemb = self._disable_grad(nn.ParameterDict(nct2excemb))

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

class CLSToken(nn.Module):
    '''
    Prepend CLS token embedding to the input embedding.
    '''
    def __init__(self, hidden_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim))
        nn_init.uniform_(self.weight, a=-1/math.sqrt(hidden_dim),b=1/math.sqrt(hidden_dim))

    def expand(self, *leading_dimensions):
        new_dims = (1,) * (len(leading_dimensions)-1)
        return self.weight.view(*new_dims, -1).expand(*leading_dimensions, -1)

    def forward(self, embedding, attention_mask=None, **kwargs):
        '''
        embeddings: [batch_size, seq_len, hidden_dim]
        '''
        embedding = torch.cat([self.expand(len(embedding), 1), embedding], dim=1)
        outputs = {'embedding': embedding}

        if attention_mask is not None:
            attention_mask = torch.cat([torch.ones(attention_mask.shape[0],1).to(attention_mask.device), attention_mask], 1)
        outputs['attention_mask'] = attention_mask
        return outputs

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
        checkpoint='./pretrained_models', 
        device='cuda:0',
        **kwargs) -> None:
        super().__init__()
        self.num_topics = num_topics
        self.n_trial_per_batch = n_trial_per_batch

        self.molecule_encoder = MoleculeEncoder(input_dir=os.path.join(checkpoint, 'molecule_encoder'))
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
        protocol_emb = self.protocol_encoder(trial.attr_dict['nctid']) # [2, 50]
        disease_emb = self.disease_encoder(trial.attr_dict['icdcodes']) # [1, 50]
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

    def init_criteria_embedding(self, train_data):
        # initialize the criteria embedding
        
        pdb.set_trace()

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
        
    pretrained_criteria_emb: bool
        Whether to use pretrained BERT to encode criteria to get the criteria embedding. It takes some time to encode the criteria.
        Set to False if you want to train the criteria embedding from scratch.
    
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
        pretrained_criteria_emb=False,
        batch_size=1,
        n_trial_per_batch=None,
        learning_rate=1e-4,
        weight_decay=1e-4,
        epochs=10,
        evaluation_steps=50,
        warmup_ratio=0,
        seed=42,
        output_dir=None,
        ):
        self.config = {
            'num_topics': num_topics,
            'n_trial_projector': n_trial_projector,
            'n_timestemp_projector': n_timestemp_projector,
            'n_rnn_layer': n_rnn_layer,
            'pretrained_criteria_emb': pretrained_criteria_emb,
            'batch_size': batch_size,
            'n_trial_per_batch':n_trial_per_batch,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'weight_decay': weight_decay,
            'evaluation_steps':evaluation_steps,
            'warmup_ratio':warmup_ratio,
            'seed': seed,
            'output_dir': output_dir,
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

    def fit(self, train_data, valid_data=None):
        '''Train the SPOT model. It has two/three steps:
        1. Encode the criteria using pretrained BERT to get the criteria embedding (optional).
        2. Discover topics from the training data and transform the data into a SequenceTrial object.
        3. Train the SPOT model.

        Parameters
        ----------
        train_data: TrialOutcomeDatasetBase
            Training data, should be a `TrialOutcomeDatasetBase` object.

        valid_data: TrialOutcomeDatasetBase
            Validation data, should be a `TrialOutcomeDatasetBase` object.
        '''

        if self.config['pretrained_criteria_emb']:
            # initialize the criteria embedding using pretrained language models
            self.model.init_criteria_embedding(train_data)

        # initialize the topics using the topic discovery model
        pdb.set_trace()

        self._fit_model(train_data, valid_data)

    def predict(self, data, target_trials=None):
        '''
        Predict the outcome of a clinical trial.
        Parameters
        ----------
        data: SequenceTrial
            A dataset consisting of multiple topics of clinical trials. Each topic is a list of clinical trials under this topic ordered with their timestamps.
        
        target_trials: list
            A list of trial ids to predict. If None, all trials will be predicted.
        '''
        self.eval()
        loader = self.get_test_dataloader(data)
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
        trainer = SuperviseTrainer(
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


