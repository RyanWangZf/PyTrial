import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
from collections import defaultdict
from pytrial.data.patient_data import SequencePatientBase, SeqPatientCollator
from pytrial.utils.check import (
    check_checkpoint_file, check_model_dir, check_model_config_file, make_dir_if_not_exist
)
import os
import pdb
import json
import copy
import torch.nn.functional as F
import random
from torch.utils.data import Dataset
from numpy import vstack
from torch.optim import Adam

from pytrial.tasks.trial_simulation.sequence.base import SequenceSimulationBase
from pytrial.tasks.trial_simulation.data import SequencePatient

class trial_data(Dataset):
    # load the dataset
    def __init__(self, X, y):
        self.X = X
        self.y = y

        self.X = self.X.astype('float32')
        self.y = self.y.astype('float32')
        #self.y = self.y.reshape((len(self.y), 1))

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X.iloc[idx, :].to_numpy(), self.y.iloc[idx, :].to_numpy()]

def prepare_data(X_train, y_train, batch_size =64):
    # load the dataset
    train = trial_data(X_train, y_train)
    # prepare data loaders. cannot shuffle during training
    train_dl = DataLoader(train, batch_size=batch_size, shuffle=False)#, drop_last=True)
    return train_dl

class Encoder(nn.Module):
    
    def __init__(self, vocab_size, event_order, freeze_order, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        input_dim = vocab_size[event_order]
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        torch.nn.init.xavier_uniform_(self.FC_input.weight)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim, latent_dim)
        torch.nn.init.xavier_uniform_(self.FC_mean.weight)
        torch.nn.init.xavier_uniform_(self.FC_var.weight)
        
        self.LeakyReLU = nn.ReLU()
        self.training = True
        
    def forward(self, x):
        h_       = self.LeakyReLU(self.FC_input(x))
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)
        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, vocab_size, event_order):
        super(Decoder, self).__init__()
        output_dim = vocab_size[event_order]
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        torch.nn.init.xavier_uniform_(self.FC_hidden.weight)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        torch.nn.init.xavier_uniform_(self.FC_output.weight)
        
        self.LeakyReLU = nn.ReLU()
        
    def forward(self, x):
        h     = self.LeakyReLU(self.FC_hidden(x))
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat
    

class DotProductAttention(nn.Module):
    """
    Compute the dot products of the query with all values and apply a softmax function to obtain the weights on the values
    """
    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, query, value):
        batch_size, input_size = query.size(0),  value.size(1)
        
        score = torch.bmm(query.unsqueeze(1), value.transpose(1, 2))
        attn = F.softmax(score, 2)  
     
        context = torch.bmm(attn, value)

        #print(context.shape)
        return context, attn

def loss_function(x, x_hat, mean, log_var, AE_out, AE_true):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    ae_loss = nn.functional.binary_cross_entropy(AE_out, AE_true, reduction='sum')
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return (reproduction_loss + KLD) + ae_loss

class Predictor(nn.Module):
    def __init__(self, latent_dim, hidden_dim, vocab_size, freeze_order, event_order):
        super(Predictor, self).__init__()
        freeze_dim = 0
        freeze_dim_range = [] # from where to where to freeze
        if freeze_order is not None:
            if isinstance(freeze_order, int):
                freeze_order = [freeze_order]
            for i in freeze_order:
                freeze_dim += vocab_size[i]
                start_idx = sum(vocab_size[:i])
                end_idx = sum(vocab_size[:i+1])
                freeze_dim_range.append([start_idx, end_idx])
        self.freeze_dim = freeze_dim
        self.freeze_dim_range = freeze_dim_range

        target_dim = sum(vocab_size) - vocab_size[event_order] - self.freeze_dim        
        self.FC_hidden = nn.Linear(latent_dim+freeze_dim, latent_dim+10)
        self.FC_hidden2 = nn.Linear(latent_dim +10, hidden_dim)
        self.FC_hidden3 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, target_dim)
        self.FC_output1 = nn.Linear(target_dim, target_dim)
        self.LeakyReLU= nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
    
    def forward(self, x):
        h     = self.LeakyReLU(self.FC_hidden(x))
        h     = self.bn1(self.LeakyReLU(self.FC_hidden2(h)))
        h     = self.LeakyReLU(self.FC_hidden3(h))
        h     = self.LeakyReLU(self.FC_output(h))
        next_time = torch.sigmoid(self.FC_output1(h))
        return next_time

class BuildModel(nn.Module):
    def __init__(self,
        hidden_dim,
        latent_dim,
        vocab_size,
        orders,
        event_type,
        freeze_type,
        device,
        epochs
        ) -> None:
        super().__init__()
        self.event_type = event_type # either medication or adverse event
        self.device = device
        self.epochs = epochs

        # find event_tpye's index in orders
        target_order = orders.index(event_type)
      
        if not isinstance(vocab_size, list): vocab_size = [vocab_size]

        freeze_order = None
        if freeze_type is not None:
            if isinstance(freeze_type, str):
                freeze_type = [freeze_type]
            freeze_order = [orders.index(i) for i in freeze_type]

        self.freeze_order = freeze_order

        self.Encoder = Encoder(vocab_size=vocab_size, event_order=target_order, freeze_order=freeze_order, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.Decoder = Decoder(latent_dim, hidden_dim, vocab_size, target_order)

        # predictor modules predicting all the other events except the freeze ones
        self.Predictor = Predictor(latent_dim, hidden_dim, vocab_size, freeze_order, target_order)
        self.freeze_dim = self.Predictor.freeze_dim
        self.freeze_dim_range = self.Predictor.freeze_dim_range

        #attention module
        self.Att = DotProductAttention()

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)        # sampling epsilon
        z = mean + var*epsilon                          # reparameterization trick
        return z

    def forward(self, x):
        #last columns are freezed
        if self.freeze_dim > 0:
            all_indexes = self._create_non_freeze_indexes(x, self.freeze_dim_range)
            x_input = x[:,:, all_indexes].contiguous()
        else:
            x_input = x

        query = x_input[:, 0, :]
        keys = x_input[:, 1:, :]
        context, _ = self.Att(query , keys )

        out_mean, out_log_var = self.Encoder(context[:,0, :])
        z = self.reparameterization(out_mean, torch.exp(0.5 * out_log_var)) # takes exponential function (log var -> var)

        z_1= torch.cat((z, x[:, 0, -self.freeze_dim:]), 1)
        pred_out = self.Predictor(z_1)
        x_hat = self.Decoder(z)
        
        return x_hat, out_mean, out_log_var , pred_out

    def _create_non_freeze_indexes(self, x, freeze_dim_range):
        all_indexes = list(range(x.shape[-1]))
        for freeze_dim_range_ in freeze_dim_range:
            all_indexes = list(set(all_indexes) - set(range(freeze_dim_range_[0], freeze_dim_range_[1])))
        return all_indexes


class TWIN(SequenceSimulationBase):
    '''
    Implement a VAE based model for clinical trial patient digital twin simulation [1]_.
    
    Parameters
    ----------
    vocab_size: list[int]
        A list of vocabulary size for different types of events, e.g., for diagnosis, procedure, medication.

    order: list[str]
        The order of event types in each visits, e.g., ``['treatment', 'medication', 'adverse event']``.
        Visit = [treatment_events, medication_events, adverse_events], each event is a list of codes.

    freeze_event: str or list[str]
        The type(s) of event to be frozen during training and generation, e.g., ``['treatment']``.

    max_visit: int
        Maximum number of visits.

    emb_size: int
        Embedding size for encoding input event codes.
        
    latent_dim: int
        Size of final latent dimension between the encoder and decoder

    learning_rate: float
        Learning rate for optimization based on SGD. Use torch.optim.Adam by default.

    batch_size: int
        Batch size when doing SGD optimization.

    epochs: int
        Maximum number of iterations taken for the solvers to converge.

    num_worker: int
        Number of workers used to do dataloading during training.

    device: str
        Device to use for training, e.g., ``'cpu'`` or ``'cuda:0'``.

    experiment_id: str
        A unique identifier for the experiment.

    verbose: bool
        If True, print out training progress.

    Notes
    -----
    .. [1] Trisha Das*, Zifeng Wang*, and Jimeng Sun. TWIN: Personalized Clinical Trial Digital Twin Generation. KDD'23.
    '''
    def __init__(self,
        vocab_size,
        order,
        freeze_event = None,
        max_visit=13,
        emb_size=64,
        hidden_dim = 36,
        latent_dim=32,
        learning_rate=5e-5,
        batch_size=64,
        epochs=20,
        num_worker=0,
        device='cpu',# 'cuda:0',
        experiment_id='trial_simulation.sequence.twin',
        verbose=False,
        ):
        super().__init__(experiment_id)

        if isinstance(freeze_event, str):
            assert freeze_event in order, f'The specified freeze_event {freeze_event} is not in order {order}!'
            freeze_event = [freeze_event]

        if freeze_event is not None:
            for et in freeze_event:
                assert et in order, f'The specified freeze_event {et} is not in order {order}!'
        
        # build perturbing events
        if len(freeze_event) == 0 or freeze_event is None:
            perturb_event = copy.deepcopy(order)
        else:
            perturb_event = [et for et in order if et not in freeze_event]

        self.config = {
            'vocab_size':vocab_size,
            'max_visit':max_visit,
            'emb_size':emb_size,
            'hidden_dim': hidden_dim,
            'latent_dim':latent_dim,
            'device':device,
            'learning_rate':learning_rate,
            'batch_size':batch_size,
            'epochs':epochs,
            'num_worker':num_worker,
            'orders':order,
            'output_dir': self.checkout_dir,
            'verbose': verbose,
            'freeze_event': freeze_event,
            'perturb_event': perturb_event,
            }
        self.config['total_vocab_size'] = sum(vocab_size)
        self.device = device
        self._build_model()
    
    def _build_model(self):
        # build unimodal TWIN for the given event types
        self.models = {}
        for et in self.config['perturb_event']:
            self.models[et] = TWIN_unimodal(
                vocab_size=self.config['vocab_size'],
                order=self.config['orders'],
                event_type=et,
                freeze_event=self.config['freeze_event'],
                max_visit=self.config['max_visit'],
                emb_size=self.config['emb_size'],
                hidden_dim=self.config['hidden_dim'],
                latent_dim=self.config['latent_dim'],
                device=self.config['device'],
                verbose=self.config['verbose'],
                )

    def _input_data_check(self, inputs):
        assert isinstance(inputs, SequencePatientBase), f'`trial_simulation.sequence` models require input training data in `SequencePatientBase`, find {type(inputs)} instead.'

    def fit(self, train_data):
        '''
        Fit the model with training data.

        Parameters
        ----------
        train_data: SequencePatientBase
            Training data.
        '''
        self._input_data_check(train_data)
        df_train_data = self._translate_sequence_to_df(train_data)

        # train the model for each event type
        for et in self.config['perturb_event']:
            model = self.models[et]
            model._fit_model(df_train_data)

        pdb.set_trace()

    def load_model(self, checkpoint):
        pass

    def save_model(self, checkpoint):
        pass

    def predict(self, number_of_predictions):
        pass

    def _translate_sequence_to_df(self, inputs):
        '''
        returns dataframe from SeqPatientBase
        '''
        inputs= inputs.visit
        column_names = ['People', 'Visit']
        for i in range(len(self.config['orders'])):
            for j in range(self.config['vocab_size'][i]):
              column_names.append(self.config['orders'][i]+'_'+str(j))

        visits = []
        for i in range(len(inputs)):#each patient
            if self.config['verbose'] and i % 100 == 0:
                print(f'Translating Data: Sample {i}/{len(inputs)}')

            for j in range(len(inputs[i])): #each visit
                binary_visit = [i, j]
                for k in range(len(self.config["orders"])): #orders indicate the order of events
                    event_binary= np.array([0]*self.config['vocab_size'][k])
                    event_binary[inputs[i][j][k]] = 1 #multihot from dense
                    binary_visit.extend(event_binary.tolist())

                visits.append(binary_visit)
        df = pd.DataFrame(visits, columns=column_names)
        return df




class TWIN_unimodal(SequenceSimulationBase):
    '''
    Implement a VAE based model for clinical trial patient digital twin simulation [1]_.
    
    Parameters
    ----------
    vocab_size: list[int]
        A list of vocabulary size for different types of events, e.g., for diagnosis, procedure, medication.

    order: list[str]
        The order of event types in each visits, e.g., ``['treatment', 'medication', 'adverse event']``.
        Visit = [treatment_events, medication_events, adverse_events], each event is a list of codes.

    event_type: str or list[str]
        The type(s) of event to be modeled, e.g., ``'medication'`` or ``'adverse event'``.
        If a list is provided, then the model will be trained to model all event types in the list.

    freeze_event: str or list[str]
        The event type(s) that will be frozen during training, e.g., ``'medication'`` or ``'adverse event'``.

    max_visit: int
        Maximum number of visits.

    emb_size: int
        Embedding size for encoding input event codes.
        
    latent_dim: int
        Size of final latent dimension between the encoder and decoder

    learning_rate: float
        Learning rate for optimization based on SGD. Use torch.optim.Adam by default.

    batch_size: int
        Batch size when doing SGD optimization.

    epochs: int
        Maximum number of iterations taken for the solvers to converge.

    num_worker: int
        Number of workers used to do dataloading during training.

    device: str
        Device to use for training, e.g., ``'cpu'`` or ``'cuda:0'``.

    experiment_id: str
        A unique identifier for the experiment.

    verbose: bool
        If True, print out training progress.

    Notes
    -----
    .. [1] Trisha Das*, Zifeng Wang*, and Jimeng Sun. TWIN: Personalized Clinical Trial Digital Twin Generation. KDD'23.
    '''
    def __init__(self,
        vocab_size,
        order,
        event_type= 'medication',
        freeze_event= None,
        max_visit=13,
        emb_size=64,
        hidden_dim = 36,
        latent_dim=32,
        learning_rate=1e-3,
        batch_size=64,
        epochs=20,
        num_worker=0,
        device='cpu',# 'cuda:0',
        experiment_id='trial_simulation.sequence.twin',
        verbose=False,
        ):
        super().__init__(experiment_id)

        assert isinstance(event_type, str), "TWIN_unimodal only supports one event type! Got {} instead.".format(event_type)

        if isinstance(freeze_event, str):
            freeze_event = [freeze_event]
        
        if freeze_event is not None:
            for et in freeze_event:
                assert et in order, "Event type {} not found in order {}.".format(et, order)

        self.config = {
            'vocab_size':vocab_size,
            'max_visit':max_visit,
            'emb_size':emb_size,
            'hidden_dim': hidden_dim,
            'latent_dim':latent_dim,
            'device':device,
            'learning_rate':learning_rate,
            'batch_size':batch_size,
            'epochs':epochs,
            'num_worker':num_worker,
            'orders':order,
            'event_type': event_type,
            'freeze_event': freeze_event,
            'output_dir': self.checkout_dir,
            'verbose': verbose,
            }
        self.config['total_vocab_size'] = sum(vocab_size)
        self.device = device
        self._build_model()
        
    def _build_model(self):
        self.model = BuildModel(
            hidden_dim = self.config['hidden_dim'],
            latent_dim=self.config['latent_dim'],
            vocab_size=self.config['vocab_size'],
            orders=self.config['orders'],
            event_type = self.config['event_type'],
            freeze_type=self.config['freeze_event'],
            device=self.device,
            epochs = self.config['epochs']
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
        raise NotImplementedError("TWIN_unimodal does not support fit method! Use `TWIN` instead.")

    def next_step_df(self, data):
        columns = pd.Series(data.columns)
        target_columns = columns[columns.str.startswith(self.config["event_type"])]
        if self.config["freeze_event"] is not None:
            freeze_type = self.config["freeze_event"]
            for t in freeze_type:
                add_target_columns = columns[columns.str.startswith(t)]
                target_columns = target_columns.append(add_target_columns)
        
        other_columns = columns[~columns.isin(target_columns)]
        other_columns = other_columns[~other_columns.isin(["Visit","People"])]

        def _create_new_col(x): 
            splits = x.split("_")
            num = int(splits[-1])
            return "###nxt###_{}_{}".format("_".join(splits[:-1]), num)

        nxt_target_columns = other_columns.apply(lambda x: _create_new_col(x))
        column_map = dict(zip(other_columns, nxt_target_columns))

        # create a new column by shifting up
        nxt_data = data[other_columns].shift(-1).rename(columns=column_map)
        data = pd.concat([data, nxt_data], axis=1)

        # remove NaN rows
        data['Visit_']= data['Visit'].shift(-1)
        data.iloc[len(data)-1,-1]=-1
        data = data[data['Visit_']-data['Visit']==1]
        data = data.drop(columns =['Visit_'])

        # build X and y
        y = data[nxt_target_columns]
        X = data[target_columns]
        return X, y

    def _train(self, train_dl, device, optimizer, vocab_size, batch_size, model, out_dir):
        if self.config["verbose"]:
            print("...Start training VAE...")
            print('--- event type: ', self.config['event_type'], '---')
            print('--- order: ', self.config['orders'], '---')
            print('--- freeze_event: ', self.config['freeze_event'], '---')
            print('--- vocab_size: ', vocab_size, '---')

        for epoch in range(self.config['epochs']):
            overall_loss = 0
            for batch_idx, (x, y) in enumerate(train_dl):
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                n_cross_n = torch.matmul(x, x.T)
                top_5_index = torch.topk(n_cross_n, 4)
                ext_x=[]

                for i in range(x.shape[0]):
                    x_=[]
                    x_.append(x[i].tolist())
                    x_.extend(x[top_5_index.indices[i]].tolist())
                    ext_x.append(x_)

                ext_x= torch.as_tensor(ext_x)
                ext_x = ext_x.to(self.config["device"])
                x_hat, out_mean, log_var , out = model(ext_x)

                # x_hat should be the event_type reconstruction
                # y should be the other events to predict

                if self.config["freeze_event"] is not None:
                    x_indexes = model._create_non_freeze_indexes(x, model.freeze_dim_range)
                    x_tgt = x[:, x_indexes]
                else:
                    x_tgt = x

                loss = loss_function(x_tgt, x_hat, out_mean, log_var, out, y)
                overall_loss += loss.item()
                loss.backward()
                optimizer.step()

            if self.config["verbose"]:
                print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ".format(self.config["event_type"]), overall_loss / (batch_idx*batch_size))

        if self.config["verbose"]:
            print("Finish!!")

    def _fit_model(self, df, out_dir=None):
        X, y = self.next_step_df(df)
        train_dl= prepare_data(X, y, self.config['batch_size'])
        optimizer = Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self._train(train_dl, self.device, optimizer, self.config['vocab_size'], self.config['batch_size'], self.model, out_dir)

    def _input_data_check(self, inputs):
        assert isinstance(inputs, SequencePatientBase), f'`trial_simulation.sequence` models require input training data in `SequencePatientBase`, find {type(inputs)} instead.'

    def SequencePatientBase_to_df(self, inputs):
        '''
        returns dataframe from SeqPatientBase
        '''
        inputs= inputs.visit
        column_names = ['People', 'Visit']
        for i in range(len(self.config['orders'])):
            for j in range(self.config['vocab_size'][i]):
              column_names.append(self.config['orders'][i]+'_'+str(j))

        df = pd.DataFrame(columns=column_names)
        for i in range(len(inputs)):#each patient
            for j in range(len(inputs[i])): #each visit
                binary_visit = [i, j]
                for k in range(len(self.config["orders"])): #orders indicate the order of events
                    event_binary= [0]*self.config['vocab_size'][k]
                    for l in inputs[i][j][k]: #multihot from dense
                        event_binary[l]=1
                    binary_visit.extend(event_binary)
                df.loc[len(df)] = binary_visit
        return df

    def df_to_SequencePatientBase(self, df, inputs):
        '''
        returns SeqPatientBase from df
        '''
        visits = []
        columns=[]
        for k in self.config['orders']:
            columns.extend([col for col in df if col.startswith('k')])
        for i in df.People.unique():
            sample=[]
            temp = df[df['People']==i]
            for index, row in temp.iterrows():
                visit=[]
                visit.append(np.nonzero(row[columns].to_list())[0].tolist())
                sample.append(visit)
            visits.append(sample)
            seqdata = SequencePatient(
                data={'v':visits, 'y': inputs.label, 'x':None},
                metadata={
                    'visit':seqdata.metadata['visit'],
                    'label':{'mode':'tensor'},
                    'voc':seqdata.metadata['voc'],
                    'max_visit':seqdata.metadata['max_visit'],
                }
            )
        return seqdata

    def predict(self, data ):
        self._input_data_check(data)
        df_data = self.SequencePatientBase_to_df(data)
        X, y = self.next_step_df(df_data)
        dl= prepare_data(X, y, self.config['batch_size'])
        self.model.eval()
        x_hats, ins= list(), list()
        for i, (x, y) in enumerate(dl):
            # evaluate the model on the test set
            n_cross_n = torch.matmul(x, x.T)
    
            #print(x.shape)
            top_5_index = torch.topk(n_cross_n, 4)
    
            ext_x=[]
            for j in range(x.shape[0]):
                x_=[]
                x_.append(x[j].tolist())
                x_.extend(x[top_5_index.indices[j]].tolist())
                ext_x.append(x_)
    
            ext_x= torch.as_tensor(ext_x)
    
            x_hat, mean, log_var, yhat= self.model(ext_x)
    
            inp = x.detach().numpy()
            x_hat = x_hat.detach().numpy()
            x_hat = x_hat.round()
            x_hats.append(x_hat)
            ins.append(inp)

        x_hats, ins=  vstack(x_hats), vstack(ins)
        return x_hats

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
        state_dict = torch.load(checkpoint_filename, map_location=self.config['device'])
        if config_filename is not None:
            config = self._load_config(config_filename)
            self.config = config
        if self.config['event_type']=='medication':
            self.model.Encoder_med.load_state_dict(state_dict['encoder'])
            self.model.Decoder_med.load_state_dict(state_dict['decoder'])
            self.model.AE_pred.load_state_dict(state_dict['predictor'])
        if self.config['event_type']=='adverse events':
            self.model.Encoder_ae.load_state_dict(state_dict['encoder'])
            self.model.Decoder_ae.load_state_dict(state_dict['decoder'])
            self.model.Med_pred.load_state_dict(state_dict['predictor'])

    def _save_config(self, config, output_dir=None):
        temp_path = os.path.join(output_dir, self.config['event_type']+'_twin_config.json')
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(
                json.dumps(config, indent=4)
            )

    def save_model(self, output_dir):
        '''
        Save the learned simulation model to the disk.
        Parameters
        ----------
        output_dir: str
            The dir to save the learned model.
        '''
        make_dir_if_not_exist(output_dir)
        self._save_config(config=self.config, output_dir=output_dir)
        if (self.config['event_type']=='medication'):
            self._save_checkpoint({
                    'encoder': self.model.Encoder_med.state_dict(),
                    'decoder': self.model.Decoder_med.state_dict(),
                    'predictor': self.model.AE_pred.state_dict()
                },output_dir=output_dir, filename='checkpoint_med.pth.tar')
        if (self.config['event_type']=='adverse events'):
            self._save_checkpoint({
                    'encoder': self.model.Encoder_ae.state_dict(),
                    'decoder': self.model.Decoder_ae.state_dict(),
                    'predictor': self.model.Med_pred.state_dict()
                },output_dir=output_dir, filename='checkpoint_ae.pth.tar')
        print('Save the trained model to:', output_dir)