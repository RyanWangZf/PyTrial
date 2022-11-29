import pdb

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import numpy as np


from pytrial.utils.check import (
    check_checkpoint_file, check_model_dir, check_model_config_file, make_dir_if_not_exist
)
from .base import PatientTrialMatchBase
from ..data import TrialCollator, PatientCollator
from ..losses import COMPOSELoss
from ..trainer import PatientTrialTrainer


class COMPOSE(PatientTrialMatchBase):
    '''
    Leverage COMPOSE model for patient-trial matching [1]_.

    One patient's label = [[0,1,2,3], [2,3,4,5]] where
    the first is a list of indices of inclusion criteria that the patient satisfies.
    the second is a list of indices of exclusion criteria that the patient does not satisfy.
    for the other criteria we dont know if the patient satisfied or not.
    In this regard, we need to:
    (1) predict "match" for all inclusion criteria of a trial for recruiting the patient
    (2) predict "unmatch" for all exclusion criteria of a trial for recruiting the patient

    # prediction logits
    # 0 is unmatch
    # 1 is match
    # 2 is unknown

    Parameters
    ----------
    order: list[str]
        The order of events within each visit in the patient's EHR data, e.g., orders=['diag','med','prod'].

    vocab_size: int
        The vocabular size of each patient's EHR event types, e.g., diag, med, prod.

    max_visit: int
        Maximum number of visits to load for EHRs inputs.

    word_dim: int
        The dimension of input word embeddings for encoding eligibility criteria.

    conv_dim: int
        The dimension of convolutional layers for processing eligibility criteria embeddings.

    mem_dim: int
        The hidden dimension of the EHR memory network (encode patient EHRs).

    mlp_dim: int
        The hidden dimension of the MLP layers in the Query Network.

    demo_dim: int
        The input dimensions of patient demographic information, e.g., age, gender.

    margin: float
        The margin when compute nn.CosineEmbeddingLoss on patient embedding and inclusion/exclusion criteria embeddings.
        Refer to Eq. (12) of the reference paper.
    
    epochs: int, optional (default=50)
        Number of iterations (epochs) over the training data.

    batch_size: int, optional (default=512)
        Number of samples in each training batch.

    learning_rate: float, optional (default=1e-3)
        The learning rate.
    
    weight_decay: float (default=0)
        The weight decay during training.

    num_worker: int
        Number of workers used to do dataloading during training.

    device: str
        The model device.
    
    Notes
    -----
    .. [1] Gao, J., et al. (2020, August). COMPOSE: cross-modal pseudo-siamese network for patient trial matching. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (pp. 803-812).
    '''
    def __init__(self,
        order,
        vocab_size,
        max_visit=20,
        word_dim=768,
        conv_dim=128,
        mem_dim=320,
        mlp_dim=512,
        demo_dim=3,
        margin=0.3,
        batch_size=512,
        epochs=50,
        learning_rate=1e-3,
        weight_decay=0,
        num_worker=0,
        device='cuda:0',
        experiment_id='test',
        ) -> None:
        super().__init__(experiment_id)
        self.config = {
            'order':order,
            'max_visit':max_visit,
            'vocab_size':vocab_size,
            'word_dim':word_dim,
            'conv_dim':conv_dim,
            'mem_dim':mem_dim,
            'mlp_dim':mlp_dim,
            'demo_dim':demo_dim,
            'margin':margin,
            'batch_size':batch_size,
            'learning_rate':learning_rate,
            'weight_decay':weight_decay,
            'epochs':epochs,
            'num_worker':num_worker,
            'device':device,
            }
        self.device = device
        self.config['total_vocab_size'] = sum(vocab_size)
        self._build_model()
    
    def predict(self, test_data, return_dict=False):
        '''
        Predict the matching of patient and ECs of each target trial.
        Will make predictions for each patient, go through all included trials.
        For example, given 1000 patients and 10 trials, will generate predictions
        as the shape of ``1000 x 10 x num_eligibility_criteria``.

        Parameters
        ----------
        test_data: dict
            A dict contains patient and trial data, respecitvely.
            
            {

            'patient': `pytrial.tasks.trial_patient_match.data.PatientData`,

            'trial': `pytrial.tasks.trial_patient_match.data.TrialData`

            }
        
        return_dict: bool
            If return results stored in dictionary, otherwise return tuples of results

        '''
        self._input_data_check(test_data)
        patient_dataloader, trial_dataloader = self.get_test_dataloader(test_data)
        pred_res = self._predict_on_dataloader(patient_dataloader, trial_dataloader)

        if return_dict:
            pred_res['pred_trial'] = self._tuple_result_to_dict(pred_res['pred_trial'])

        return pred_res

    def fit(self, train_data, valid_data=None):
        '''
        Fit patient-trial matching model.

        Parameters
        ----------
        train_data: dict
            A dict contains patient and trial data, respectively.

            {

            'patient': `pytrial.tasks.trial_patient_match.data.PatientData`,

            'trial': `pytrial.tasks.trial_patient_match.data.TrialData`

            }

        valid_data: dict
            A dict contains patient and trial data for evaluation. Same format as
            `train_data`.

        '''
        self._input_data_check(train_data)
        if valid_data is not None: self._input_data_check(valid_data)
        self._fit_model(train_data, valid_data)

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
        trial_data = train_data['trial']
        patient_data = train_data['patient']

        patient_dataloader = DataLoader(patient_data,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_worker'],
            pin_memory=True,
            shuffle=True,
            collate_fn=PatientCollator(
                config={
                    'visit_mode':train_data['patient'].metadata['visit']['mode'],
                    'label_mode':train_data['patient'].metadata['label']['mode'],
                    }
                ),
            )
        
        trial_dataloader = DataLoader(trial_data,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_worker'],
            pin_memory=True,
            shuffle=True,
            collate_fn=TrialCollator(),
        )

        return patient_dataloader, trial_dataloader


    def get_test_dataloader(self, test_data):
        trial_data = test_data['trial']
        patient_data = test_data['patient']

        patient_dataloader = DataLoader(patient_data,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_worker'],
            pin_memory=False,
            shuffle=False,
            collate_fn=PatientCollator(
                config={
                    'visit_mode':patient_data.metadata['visit']['mode'],
                    'label_mode':patient_data.metadata['label']['mode'],
                    }
                ),
            )
        
        trial_dataloader = DataLoader(trial_data,
            batch_size=1, # get one trial once when making predictions
            num_workers=self.config['num_worker'],
            pin_memory=False,
            shuffle=False,
            collate_fn=TrialCollator(),
        )

        return patient_dataloader, trial_dataloader

    def _build_model(self):
        self.model = BuildModel(
            total_vocab_size=self.config['total_vocab_size'],
            word_dim=self.config['word_dim'],
            conv_dim=self.config['conv_dim'],
            mem_dim=self.config['mem_dim'],
            demo_dim=self.config['demo_dim'],
            mlp_dim=self.config['mlp_dim'],
        )
        if 'cuda' in self.device:
            self.model.cuda()

    def _fit_model(self, train_data, valid_data):
        patient_dataloader, trial_dataloader = self.get_train_dataloader(train_data)
        loss_models = self._build_loss_model()
        # need to customize the training process in trainer
        train_objectives = [[[patient_dataloader,trial_dataloader], loss_model] for loss_model in loss_models]
        trainer = PatientTrialTrainer(model=self,
            train_objectives=train_objectives,
            test_data=valid_data,        
        )
        trainer.train(**self.config)
    
    @torch.no_grad()
    def _predict_on_dataloader(self, patient_dataloader, trial_dataloader, return_label=False):
        '''Return trial level matching predictions.
        Return label for evaluation if required.
        '''
        self.eval()

        # record predictions
        pred_all, label_all = [], []
        pred_inc, pred_exc = [], []

        # record labels
        label_trial_inc, label_trial_exc = [], []
        label_patient_inc, label_patient_exc = [], []

        for i, batch_trial in enumerate(trial_dataloader):
            
            pred_patient_inc, pred_patient_exc = [], []

            label_trial = []
            label_trial_inc_, label_trial_exc_ = [], []

            for batch_ehr in patient_dataloader:
                inputs = {}
                inputs.update(batch_ehr)
                inputs.update(batch_trial)
                inputs = self._prepare_input(inputs)
                pred = self.model(inputs)
                pred_patient_inc.append(pred['logit_inc'])
                pred_patient_exc.append(pred['logit_exc'])

                if return_label:
                    # match trial ec idnex with patient index
                    match_inc, match_inc_ec = self._match_patient_with_trial_ec(batch_trial=batch_trial, batch_ehr=batch_ehr, ec='inc', ec_idx=0)
                    match_exc, match_exc_ec = self._match_patient_with_trial_ec(batch_trial=batch_trial, batch_ehr=batch_ehr, ec='exc', ec_idx=1)
                    match_label = match_inc * ~match_exc
                    label_trial.append(match_label.astype(int))
                    label_trial_inc_.append(match_inc_ec) # bs_patient, num_ec
                    label_trial_exc_.append(match_exc_ec)

                    if i == 0:
                        # get patient label
                        [label_patient_inc.append(y[0]) for y in batch_ehr['y']]
                        [label_patient_exc.append(y[1]) for y in batch_ehr['y']]


            if return_label:
                label_trial_inc_ = np.concatenate(label_trial_inc_)
                label_trial_exc_ = np.concatenate(label_trial_exc_)
                label_trial_inc.append(label_trial_inc_)
                label_trial_exc.append(label_trial_exc_)
                label_all.append(np.concatenate(label_trial))

            # get a batch of trial to all EHRs matching
            pred_patient_inc = torch.cat(pred_patient_inc, 1) # 1, num_patients, num_inc_ec, output_class
            pred_patient_exc = torch.cat(pred_patient_exc, 1) # 1, num_patients, num_exc_ec, output_class

            pred_trial_inc = self._match_trial_for_patients(pred_patient_inc, 'inc')
            pred_trial_exc = self._match_trial_for_patients(pred_patient_exc, 'exc')
            pred_trial = pred_trial_exc * pred_trial_inc # iff both inc and exc are matched, shape=(bs_trial, bs_patient)

            pred_all.append((batch_trial['nct_id'], pred_trial.detach().cpu().numpy()))
            pred_inc.append(pred_patient_inc.detach().cpu().numpy())
            pred_exc.append(pred_patient_exc.detach().cpu().numpy())        
        
        return_res = {
            'pred_trial':pred_all, # trial-level prediction
            'pred_inc': pred_inc, # inclusion-level prediction
            'pred_exc': pred_exc, # exclusion-level prediction
        }

        if return_label:
            return_res['label_trial'] = np.stack(label_all) # trial-level labels
            return_res['label_patient_inc'] = label_patient_inc # patient inclusion criteria label
            return_res['label_patient_exc'] = label_patient_exc # patient exclusion criteria label
            return_res['label_trial_inc'] = label_trial_inc # trial inclusion criteria index
            return_res['label_trial_exc'] = label_trial_exc # trial exclusion criteria index

        return return_res

    def _build_loss_model(self):
        return [COMPOSELoss(self.model)]

    def _prepare_input(self, data):
        '''
        Prepare inputs for compose model encoding.
        {
            'v': patient visit events,
            'x': patient features,
            'y': which ECs patient are matched,
            'inc_ec_index': sampled batch of inclusion criteria index,
            'exc_ec_index': sampled batch of exclusion criteria index,
            'inc_ec_index': sampled batch of inclusion criteria embedding,
            'exc_ec_index': sampled batch of exclusion criteria embedding,
        }
        '''
        inputs = self._prepare_ehr_input(data=data)
        trial_inputs = self._prepare_trial_patient_match_input(data=data)
        inputs.update(trial_inputs)
        return inputs

    def _prepare_ehr_input(self, data):
        inputs = {}
        visits = data['v']
        feature = data['x']

        if not isinstance(feature, torch.Tensor): feature = torch.tensor(feature)
        feature = feature.to(self.device)
        v_lengths = [len(visits[self.config['order'][0]][idx][:self.config['max_visit']]) for idx in range(len(visits[self.config['order'][0]]))]
        mask = torch.zeros(len(v_lengths), max(v_lengths))

        v = torch.zeros(len(v_lengths), max(v_lengths), self.config['total_vocab_size'])
        for idx in range(len(v_lengths)):
            v[idx,:v_lengths[idx]] = torch.tensor(self._translate_dense_visits_to_sparse({k: visits[k][idx][:self.config['max_visit']] for k in visits}))
            mask[idx,:v_lengths[idx]] = 1

        if 'cuda' in self.device:
            v = v.cuda()
            mask = mask.cuda()
            feature = feature.cuda()

        inputs['x'] = feature.float()
        inputs['mask'] = mask
        inputs['v'] = v
        return inputs

    def _prepare_trial_patient_match_input(self, data):
        outputs = {}
        inc_inputs = self._prepare_ec_input(data, 'inc', 0) # patient label 0 corresponds to inclusion
        exc_inputs = self._prepare_ec_input(data, 'exc', 1) # patient label 1 corresponds to exclusion
        outputs.update(inc_inputs)
        outputs.update(exc_inputs)
        return outputs

    def _prepare_ec_input(self, data, ec, label_idx):
        patient_label = data['y']
        inc_label = []
        inc_idx = data[f'{ec}_ec_index']
        inc_emb = data[f'{ec}_ec_emb']
        max_n_inc = [len(l) for l in inc_idx]
        inc_mask = torch.zeros(len(max_n_inc), max(max_n_inc))

        num_patient = len(patient_label)
        num_sample = min(len(max_n_inc), num_patient)

        inc_emb_ts = []
        for idx in range(num_sample): 
            # go through all trials
            # if len(patients) > len(trials)
            # only take the top len(trials) samples
            ec_embs = torch.zeros(max(max_n_inc), inc_emb[0].shape[-1])
            ec_embs[:max_n_inc[idx]] = inc_emb[idx]
            inc_emb_ts.append(ec_embs)
            inc_mask[idx][:max_n_inc[idx]] = 1
            p_label = patient_label[idx][label_idx]
            
            # each trial match each patient
            # trial1 -> patient1
            # trial2 -> patient2
            # ...
            # trial10 -> patient10
            # the remaining of patients will be ignored if len(trial) == 10
            inc_label_ = np.in1d(inc_idx[idx], p_label)

            if ec == 'inc': # inclusion
                inc_label_ = torch.tensor(inc_label_.astype(int)) # match is 1
                inc_label_[inc_label_==0]  = 2 # for the other inclusion it is unknown
            else: # exclusion
                inc_label_ = torch.tensor((~inc_label_).astype(int)) # unmatch is 0
                inc_label_[inc_label_!=0] = 2 # for the other exclusion it is unknown

            if 'cuda' in self.device: inc_label_ = inc_label_.cuda()
            inc_label.append(inc_label_)

        inc_emb_ts = torch.stack(inc_emb_ts)

        if 'cuda' in self.device:
            inc_emb_ts = inc_emb_ts.cuda()
            inc_mask = inc_mask.cuda()

        inc_label = pad_sequence(inc_label, batch_first=True)

        outputs = {
            f'{ec}_emb': inc_emb_ts,
            f'{ec}_emb_mask': inc_mask,
            f'{ec}_label': inc_label,
            }
        return outputs
    
    def _match_patient_with_trial_ec(self, batch_trial, batch_ehr, ec, ec_idx):
        trial_inc_idx = batch_trial[f'{ec}_ec_index']# assume test dataloader only has 1 sample each batch
        patient_label = batch_ehr['y']
        match_label, match_ec_label = [], []
        for p_label in patient_label:
            matched = np.in1d(trial_inc_idx, p_label[ec_idx]).astype(int)
            
            if ec_idx == 0: # inclusion
                label = sum(matched) == len(trial_inc_idx)
                matched[matched==0] = 2 # unk
            else: # exclusion
                label = sum(matched) == 0
                matched[matched==0] = 2 # unk
                matched[matched==1] = 0 # unmatch
            
            match_ec_label.append(matched)
            match_label.append(label)

        return np.array(match_label), np.stack(match_ec_label)

class BuildModel(nn.Module):
    '''
    Build COMPOSE model.
    '''
    def __init__(self, 
        total_vocab_size,
        word_dim, 
        conv_dim,
        mem_dim,
        demo_dim,
        mlp_dim,
        ) -> None:
        super().__init__()
        self.embedding_network = ECEmbedding(word_dim=word_dim, conv_dim=conv_dim)
        self.memory_network = EHRMemoryNetwork(total_vocab_size=total_vocab_size, word_dim=word_dim, mem_dim=mem_dim, demo_dim=demo_dim)
        self.query_network = QueryNetwork(mem_dim=mem_dim, conv_dim=conv_dim, mlp_dim=mlp_dim)

    def forward(self, inputs):
        ehr = inputs['v'] # tensor bs, n_visit, n_event
        demo = inputs['x'] # tensor bs, n_feature
        ehr_mask = inputs['mask']
        memory = self.memory_network(ehr, demo, ehr_mask) # batch_size, 2, mem_dim

        inc_ec, inc_ec_mask = inputs['inc_emb'], inputs['inc_emb_mask']
        inc_emb, inc_ec_emb = self.embedding_network(inc_ec, inc_ec_mask) # bs, num_ec, 512

        exc_ec, exc_ec_mask = inputs['exc_emb'], inputs['exc_emb_mask']
        exc_emb, exc_ec_emb = self.embedding_network(exc_ec, exc_ec_mask) # bs, num_ec, 512

        ouptut_inc, response_inc, query_inc = self.query_network(memory, inc_ec_emb, inc_ec_mask) # bs_trial, bs_ehr, num_ec, 2
        output_exc, response_exc, query_exc = self.query_network(memory, exc_ec_emb, exc_ec_mask) # bs_trial, bs_ehr, num_ec, 2

        # pred_inc = torch.softmax(ouptut_inc, dim=-1)
        # pred_exc = torch.softmax(output_exc, dim=-1)
        return {'logit_inc':ouptut_inc, 
                'logit_exc':output_exc, 
                'response_inc':response_inc,
                'response_exc':response_exc,
                'query_inc':query_inc,
                'query_exc':query_exc,
                }


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)
        
        self.__padding = (kernel_size - 1) * dilation
        
    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))

class HighwayBlock(nn.Module):
    def __init__(self, input_dim, kernel_size):
        super(HighwayBlock, self).__init__()
        self.conv_t = CausalConv1d(input_dim, input_dim, kernel_size)
        self.conv_z = CausalConv1d(input_dim, input_dim, kernel_size)

    def forward(self, input):
        t = torch.sigmoid(self.conv_t(input))
        z = t * self.conv_z(input) + (1-t) * input
        return z
    
class ECEmbedding(nn.Module):
    def __init__(self, word_dim, conv_dim):
        super(ECEmbedding, self).__init__()
        # TODO
        # add trainable embeddings: ec_emb, ec_emb_fixed
        #Input: batch_size * sentence_len * embd_dim
        self.word_dim = word_dim
        self.conv_dim = conv_dim
        
        #High-way network for word features
        self.init_conv1 = CausalConv1d(word_dim, conv_dim, 1)
        self.init_conv2 = CausalConv1d(word_dim, conv_dim, 3)
        self.init_conv3 = CausalConv1d(word_dim, conv_dim, 5)
        self.init_conv4 = CausalConv1d(word_dim, conv_dim, 7)
        self.highway1 = HighwayBlock(4*conv_dim, 3)
        self.highway2 = HighwayBlock(4*conv_dim, 3)
        self.highway3 = HighwayBlock(4*conv_dim, 3)

        self.pool = nn.AdaptiveMaxPool1d(1)
    
    def forward(self, input, mask):
        #Input: B * L * embd_dim
        #Mask: B * L

        input = input.permute(0,2,1)
        conv1 = self.init_conv1(input)
        conv2 = self.init_conv2(input)
        conv3 = self.init_conv3(input)
        conv4 = self.init_conv4(input)
        concat = torch.cat((conv1,conv2,conv3,conv4), dim=1)

        highway_res = self.highway1(concat)
        highway_res = torch.relu(highway_res)

        highway_res = self.highway2(highway_res)
        highway_res = torch.relu(highway_res)
        highway_res = self.highway3(highway_res)
        highway_res = torch.relu(highway_res) # bs, 512, num_ec

        highway_res = highway_res * mask.unsqueeze(1) # bs, 512, num_ec
        pooled_res = self.pool(highway_res)
        pooled_res = pooled_res.squeeze(-1) # bs, 512, encoding for the whole trial?
        highway_res = highway_res.permute(0,2,1) # bs, num_ec, 512
        return pooled_res, highway_res
  
class EHRMemoryNetwork(nn.Module):
    def __init__(self, total_vocab_size, word_dim, mem_dim, demo_dim):
        super(EHRMemoryNetwork, self).__init__()
        self.mem_dim = mem_dim
        self.ehr_embedding_matrix = nn.Linear(total_vocab_size, word_dim, bias=False)
        self.erase_layer = nn.Linear(word_dim, mem_dim)
        self.add_layer = nn.Linear(word_dim, mem_dim)
        self.demo_embd = nn.Linear(demo_dim, mem_dim)

        self.init_memory = nn.Parameter(torch.randn(1, mem_dim))

    def forward(self, input, demo, mask):
        '''
        TODO: use hierarchy of codes to enhance code embeddings
        Now: we only make multihot embedding of the input codes.
        input is ehr concept embedding with (bs, num_visit, 12=3x4, word_dim),
        12 is 3 types of codes and 4 layers of code hierarchy,
        maxpooling is applied to input EHRs when each type has multiple codes
        '''
        batch_size = input.size(0)
        time_step = input.size(1)
        memory = self.init_memory
        demo_mem = torch.tanh(self.demo_embd(demo.float()))

        for i in range(time_step):
            cur_input = input[:, i, ...] # bs, vocab_size
            cur_input = self.ehr_embedding_matrix(cur_input) # bs, word_dim
            erase = torch.sigmoid(self.erase_layer(cur_input)) # bs, 320
            add = torch.tanh(self.add_layer(cur_input)) # bs, 320
            cur_mask = mask[:, i].reshape(batch_size, 1) # bs, 1
            erase = erase * cur_mask
            add = add * cur_mask
            update_ratio = (1 - erase) + add # bs, 320
            memory = memory * update_ratio # bs, 320
        
        memory = torch.stack([memory, demo_mem], dim=1) # bs, 2, mem_dim
        return memory

class QueryNetwork(nn.Module):
    def __init__(self, mem_dim, conv_dim, mlp_dim):
        super(QueryNetwork, self).__init__()
        self.word_trans = nn.Linear(4*conv_dim,mem_dim, bias=False)
        self.mlp = nn.Linear(2*mem_dim, mlp_dim)
        self.output = nn.Linear(mlp_dim, 3) # 0: match, 1: umatch, 2: unknown
  
    def forward(self, memory, query, ec_mask):
        #query: bs, num_ec, 4*conv_dim
        #memory: bs, 2, mem_dim

        trans_query = self.word_trans(query) 
        trans_query = torch.relu(trans_query) #bs, num_ec, mem_dim

        # bs_trial, bs, num_ec, mem_dim
        trans_query = trans_query[None].repeat(len(memory), 1, 1, 1).permute(1,0,2,3)
        
        pred_list = []
        response_list = []
        # go through all trials
        for idx, t_query in enumerate(trans_query):
            # mask is only useful when computing loss?
            mask = ec_mask[idx] # num_ec

            # t_query.shape = bs, num_ec, mem_dim
            attention = torch.bmm(t_query, memory.permute(0,2,1)) # bs, num_ec, 2
            attention = torch.softmax(attention, dim=-1)  # bs, num_ec, 2, attention over memory slots
            response = torch.bmm(attention, memory) # bs, num_ec, mem_dim

            output = torch.cat((response, t_query), dim=-1) # bs, num_ec, 2*mem_dim
            output = self.mlp(output) # bs, num_ec, 512
            output = torch.relu(output)
            output = self.output(output) # bs, num_ec, 2 or 3(if label has the class `neutral`)

            pred_list.append(output)
            response_list.append(response)
        
        pred_list = torch.stack(pred_list, 0) # bs_trial, bs_ehr, num_ec, output_class
        response_list = torch.stack(response_list, 0) # bs_trial, bs_ehr, num_ec, mem_dim
        return pred_list, response_list, trans_query