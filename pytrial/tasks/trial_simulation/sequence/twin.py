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
from pytrial.tasks.trial_simulation.sequence.base import SequenceSimulationBase
from pytrial.tasks.trial_simulation.data import SequencePatient
import os
import pdb
import json
import torch.nn.functional as F
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import vstack
from torch.optim import Adam

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
    #test = trial_data(X_test, y_test)
    
    batch_size =batch_size
    # prepare data loaders. cannot shuffle during training
    train_dl = DataLoader(train, batch_size=batch_size, shuffle=False)#, drop_last=True)
    #test_dl = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_dl#, test_dl

class Encoder(nn.Module):
    
    def __init__(self, vocab_size, event_order, hidden_dim, latent_dim):
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

class AE_pred_module(nn.Module):
  def __init__(self, latent_dim, hidden_dim, vocab_size, treatment_order, event_order):
     super(AE_pred_module, self).__init__()
     treatment_dim = vocab_size[treatment_order]
     ae_dim = vocab_size[event_order]
     self.FC_hidden = nn.Linear(latent_dim+treatment_dim, latent_dim+10)
     self.FC_hidden2 = nn.Linear(latent_dim +10, hidden_dim)
     self.FC_hidden3 = nn.Linear(hidden_dim, hidden_dim)
     self.FC_output = nn.Linear(hidden_dim, ae_dim)
     self.FC_output1 = nn.Linear(ae_dim, ae_dim)
     self.LeakyReLU= nn.ReLU()
     self.bn1 = nn.BatchNorm1d(hidden_dim)
    
  def forward(self, x):
      h     = self.LeakyReLU(self.FC_hidden(x))
      h     = self.bn1(self.LeakyReLU(self.FC_hidden2(h)))
      h     = self.LeakyReLU(self.FC_hidden3(h))
      h     = self.LeakyReLU(self.FC_output(h))
      next_time_ae = torch.sigmoid(self.FC_output1(h))
      return next_time_ae

class Med_pred_module(nn.Module):
  def __init__(self, latent_dim, hidden_dim, vocab_size, treatment_order, event_order):
     super(Med_pred_module, self).__init__()
     treatment_dim = vocab_size[treatment_order]
     med_dim = vocab_size[event_order]
     self.FC_hidden = nn.Linear(latent_dim+treatment_dim, latent_dim+10)
     self.FC_hidden2 = nn.Linear(latent_dim +10, hidden_dim)
     self.FC_hidden3 = nn.Linear(hidden_dim, hidden_dim)
     self.FC_output = nn.Linear(hidden_dim, med_dim)
     self.FC_output1 = nn.Linear(med_dim, med_dim)
     self.LeakyReLU= nn.ReLU()
     self.bn1 = nn.BatchNorm1d(hidden_dim)
    
  def forward(self, x):
      h     = self.LeakyReLU(self.FC_hidden(x))
      h     = self.bn1(self.LeakyReLU(self.FC_hidden2(h)))
      h     = self.LeakyReLU(self.FC_hidden3(h))
      h     = self.LeakyReLU(self.FC_output(h))
      next_time_med = torch.sigmoid(self.FC_output1(h))
      return next_time_med

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
class BuildModel(nn.Module):
    def __init__(self,
        hidden_dim,
        latent_dim,
        vocab_size,
        orders,
        event_type,
        device,
        epochs
        ) -> None:
        super().__init__()
        self.event_type = event_type # either medication or adverse event
        self.device = device
        self.ae_order=2
        self.med_order=1
        self. treatment_order = 0
        self.epochs = epochs

        if not isinstance(vocab_size, list): vocab_size = [vocab_size]

        if( event_type == 'medication'):
        #encoders for medications and adverse events
          self.Encoder_med = Encoder(vocab_size, self.med_order, hidden_dim, latent_dim)
          #decoders for medications and adverse events
          self.Decoder_med = Decoder(latent_dim, hidden_dim, vocab_size, self.med_order)
          #predictive modules for next time steps
          self.AE_pred = AE_pred_module(latent_dim, hidden_dim, vocab_size, self.treatment_order, self.ae_order)

        if(event_type == 'adverse events'):
          self.Encoder_ae = Encoder(vocab_size, self.ae_order, hidden_dim, latent_dim)
          self.Decoder_ae = Decoder(latent_dim, hidden_dim, vocab_size, self.ae_order)
          self.Med_pred = Med_pred_module(latent_dim, hidden_dim, vocab_size, self.treatment_order, self.med_order)

        #attention module
        self.Att = DotProductAttention()

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)        # sampling epsilon
        z = mean + var*epsilon                          # reparameterization trick
        return z

    def forward_med(self, med_x):#forward for medication reconstruction
        x = med_x[:,:, :-5]
        
        query = x[:, 0, :]
        keys = x[:, 1:, :]

        context, attn = self.Att(query, keys)

        med_mean, med_log_var = self.Encoder_med(context[:,0, :])
        z = self.reparameterization(med_mean, torch.exp(0.5 * med_log_var)) # takes exponential function (log var -> var)

        z_1= torch.cat((z, med_x[:,0, -5:]), 1)

        AE_out = self.AE_pred(z_1)

        med_x_hat  = self.Decoder_med(z)
        
        return med_x_hat, med_mean, med_log_var , AE_out

    def forward_ae(self, ae_x): #forward for adverse event reconstruction
        x = ae_x[:,:, :-5]
        
        
        query = x[:, 0, :]
        keys = x[:, 1:, :]
        context, _ = self.Att(query , keys )

        #print(context.shape, x.shape, query.shape, keys.shape)
        ae_mean, ae_log_var = self.Encoder_ae(context[:,0, :])
        z = self.reparameterization(ae_mean, torch.exp(0.5 * ae_log_var)) # takes exponential function (log var -> var)


        z_1= torch.cat((z, ae_x[:,0, -5:]), 1)
        Med_out = self.Med_pred(z_1)
        ae_x_hat = self.Decoder_ae(z)
        
        return ae_x_hat, ae_mean, ae_log_var , Med_out

    def forward(self, x): #last five columns are treatments
        if( self.event_type == 'medication'):
          x_hat, mean, log_var , out = self.forward_med(x)
        if (self.event_type == 'adverse events'):
          x_hat, mean, log_var , out = self.forward_ae( x)

        return x_hat, mean, log_var , out


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

    Notes
    -----
    .. [1] Trisha Das*, Zifeng Wang*, and Jimeng Sun. TWIN: Personalized Clinical Trial Digital Twin Generation. KDD'23.
    '''
    def __init__(self,
        vocab_size,
        order,
        event_type= 'medication',
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
        ):
        super().__init__(experiment_id)
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
            device=self.device,
            epochs = self.config['epochs']
            )
        self.model = self.model.to(self.device)

    def fit(self, train_data, out_dir):
      '''
      Train model with sequential patient records.
      Parameters
      ----------
      train_data: SequencePatientBase
          A `SequencePatientBase` contains patient records where 'v' corresponds to 
          visit sequence of different events.
      '''
      self._input_data_check(train_data)
      df_train_data = self.SequencePatientBase_to_df(train_data)
      self._fit_model(df_train_data, out_dir)

    def next_step_df(self, data):
      if self.config['event_type']== 'medication':
        ae_columns = [col for col in data if col.startswith('adverse events_')]
        for a in ae_columns:
          data[a[0:15]+'nxt_'+ a[15:]]= data[a].shift(-1)
        #create a new column by shifting up
        data['Visit_']= data['Visit'].shift(-1)
        data.iloc[len(data)-1,-1]=-1
        data = data[data['Visit_']-data['Visit']==1]
        data = data.drop(columns =['Visit_'])

        label_cols=[col for col in data if col.startswith('adverse events_nxt_')]
        y = data[label_cols]

        med_cols=[col for col in data if col.startswith('medication_')]
        treat_cols=[col for col in data if col.startswith('treatment_')]
        cols=med_cols+treat_cols
        X = data[cols]
        
      if self.config['event_type']== 'adverse events':
        med_columns = [col for col in data if col.startswith('medication_')]
        for a in med_columns:
          data[a[0:11]+'nxt_'+ a[11:]]= data[a].shift(-1)
        #create a new column by shifting up
        data['Visit_']= data['Visit'].shift(-1)
        data.iloc[len(data)-1,-1]=-1
        data = data[data['Visit_']-data['Visit']==1]
        data = data.drop(columns =['Visit_'])

        label_cols=[col for col in data if col.startswith('medication_nxt_')]
        y = data[label_cols]

        ae_cols=[col for col in data if col.startswith('adverse events_')]
        treat_cols=[col for col in data if col.startswith('treatment_')]
        cols=ae_cols+treat_cols
        X = data[cols]
        #print(X.columns)
      return X, y

    def train(self, train_dl, device, optimizer, vocab_size, batch_size, model,out_dir):
      print("...Start training VAE...")
      print('--- event type: ', self.config['event_type'], '---')
      best_auc=0
      best_train_auc = 0

      for epoch in range(self.config['epochs']):

          #model.train()
          overall_loss = 0
          for batch_idx, (x, y) in enumerate(train_dl):
              
              x = x.to(device)
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
              #print(ext_x[:,:, :-5])

              x_hat, mean, log_var , out = model(ext_x)

              loss= loss_function(x[:, :-5], x_hat, mean, log_var, out, y)
              
              overall_loss += loss.item() 
              
              loss.backward()
              optimizer.step()
                
          print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))
      self.save_model(output_dir=out_dir)
      print("Finish!!")

    def _fit_model(self, df, out_dir):
        X, y = self.next_step_df(df)
        train_dl= prepare_data(X, y, self.config['batch_size'])
        optimizer = Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.train(train_dl, self.device, optimizer, self.config['vocab_size'], self.config['batch_size'], self.model, out_dir)

    def _input_data_check(self, inputs):
        assert isinstance(inputs, SequencePatientBase), f'`trial_simulation.sequence` models require input training data in `SequencePatientBase`, find {type(inputs)} instead.'


    def SequencePatientBase_to_df(self, inputs):
    #convert SequencePatientBase to dataframe for training twin
        '''
        returns dataframe from SeqPatientBase
        '''
        #voc1 = inputs.voc
        inputs= inputs.visit
        column_names = ['People', 'Visit']
        for i in range(len(self.config['orders'])):
            for j in range(self.config['vocab_size'][i]):
              column_names.append(self.config['orders'][i]+'_'+str(j))

        df = pd.DataFrame(columns=column_names)
        for i in range(len(inputs)):#each patient
          for j in range(len(inputs[i])): #each visit
            binary_visit = [i, j]
            #print(inputs[i])
            for k in range(len(inputs[i][j])): #k=3 types of events
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
