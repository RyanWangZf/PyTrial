'''
Implement Logistic Regression model for clinical trial outcome prediction
'''
import os
import json

from sklearn.metrics import roc_auc_score
import numpy as np 
import torch 
from torch import nn 
from torch.autograd import Variable
from rdkit import Chem 
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs

from .base import TrialOutcomeBase
from .model_utils.utils import trial_collate_fn

class BuildModel(nn.Module):
    def __init__(self,
        input_dim=512,
        output_dim=1,
        hidden_dim=128,
        num_layer=2,
        ) -> None:
        super().__init__()
        if num_layer == 1:
            self.mlp = nn.ModuleList([nn.Linear(input_dim, output_dim)])
        else:
            self.mlp = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])

        for _ in range(num_layer-2):
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Linear(hidden_dim, hidden_dim))
        
        if num_layer > 1:
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Linear(hidden_dim, output_dim))
    
    def forward(self, inputs):
        h = inputs
        h = h.float()
        for layer in self.mlp:
            h = layer(h)
        return h

class MLP(TrialOutcomeBase):
    '''
    Implement MLP model for clinical trial outcome prediction. 

    Parameters
    ----------
    epoch: int
        number of training epochs.
    
    lr: float
        learning rate.

    weight_decay: float
        Regularization strength for l2 norm; must be a positive float.
    '''
    def __init__(self, 
        epoch = 5,
        lr = 1e-3,
        batch_size = 32, 
        weight_decay=0.0
        ):

        super(MLP, self).__init__()
        
        self.model = BuildModel() 

        self.config = {
            'epoch': epoch,
            'lr': lr,
            'weight_decay': weight_decay,
            'batch_size': batch_size,
        }

        self.epoch = epoch 
        self.lr = lr 
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.loss = nn.BCEWithLogitsLoss()

    def feature(self, data_loader):
        labels = []
        features = []
        nctids = []
        for nctid_lst, label_vec, smiles_lst2, icdcode_lst3, criteria_lst in data_loader:
            label_vec = label_vec.to(self.device)
            nctids.extend(nctid_lst)
            for smiles_lst in smiles_lst2: 
                mol_lst = [Chem.MolFromSmiles(smiles) for smiles in smiles_lst]
                mol_lst = list(filter(lambda x:x is not None, mol_lst))
                fp_lst = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512) for mol in mol_lst]
                feature = np.zeros(512)
                for fp in fp_lst:
                    arr = np.zeros((0,), dtype=np.int8)
                    DataStructs.ConvertToNumpyArray(fp,arr)
                    feature += arr 
                feature = feature.reshape(1,-1)
                features.append(feature)
        features = np.concatenate(features, 0)
        labels = np.array(labels)	
        return nctids, features, labels 	


    def fit(self, train_data, valid_data=None):
        '''
        Train model to predict clinical trial outcomes.

        Parameters
        ----------
        train_data: TrialOutcomeDatasetBase
            Training data, should be a `TrialOutcomeDatasetBase` object.
        
        valid_data: TrialOutcomeDatasetBase
            Validation data, should be a `TrialOutcomeDatasetBase` object. 
        '''

        # build dataloader using train_data
        train_loader = self._build_dataloader_from_dataset(train_data, 
            num_workers=0, batch_size=self.batch_size, shuffle=True,
            collate_fn=trial_collate_fn)
        
        if valid_data is not None:
            valid_loader = self._build_dataloader_from_dataset(valid_data, 
                num_workers=0, batch_size=self.batch_size, shuffle=False,
                collate_fn=trial_collate_fn)

        opt = torch.optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = self.weight_decay)

        for epoch in range(self.epoch):
            for nctid_lst, label_vec, smiles_lst2, icdcode_lst3, criteria_lst in train_loader:
                features = []
                for smiles_lst in smiles_lst2: 
                    mol_lst = [Chem.MolFromSmiles(smiles) for smiles in smiles_lst]
                    mol_lst = list(filter(lambda x:x is not None, mol_lst))
                    fp_lst = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512) for mol in mol_lst]
                    feature = np.zeros(512)
                    for fp in fp_lst:
                        arr = np.zeros((0,), dtype=np.int8)
                        DataStructs.ConvertToNumpyArray(fp,arr)
                        feature += arr 
                    feature = feature.reshape(1,-1)
                    features.append(feature)
                features = np.concatenate(features, 0)
                features = Variable(torch.from_numpy(features))
                y = self.model(features) 
                y = y.view(-1)
                loss = self.loss(y, label_vec.float())
                opt.zero_grad() 
                loss.backward() 
                opt.step()

            if valid_data is not None:
                eval_res = self._evaluate(valid_loader)
                for k, v in eval_res.items():
                    print('Epoch %d, %s: %.4f' % (epoch, k, v))
                
    def predict(self, test_data):
        '''
        Make clinical trial outcome predictions.
        
        Parameters
        ----------
        test_data: TrialOutcomeDatasetBase
            Testing data, should be a `TrialOutcomeDatasetBase` object.
        '''

        test_loader = self._build_dataloader_from_dataset(test_data, 
            num_workers=0, batch_size=self.batch_size, shuffle=False,
            collate_fn=trial_collate_fn)
        res = self._predict_on_dataloader(test_loader)
        nctids, ypred = res['nctid'], res['ypred']
        prediction = list(zip(nctids, ypred))
        return prediction

    def save_model(self, output_dir = None):
        '''
        Save the learned model to the disk.

        Parameters
        ----------
        output_dir: str or None
            The output folder to save the learned model.
            If set None, will save model to `save_model/model.ckpt`.
        '''
        if output_dir is None:
            output_dir = 'save_model'
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        filename = os.path.join(output_dir, 'model.ckpt')
        torch.save(self.model, filename)

        config_filename = os.path.join(output_dir, 'config.json')
        with open(config_filename, 'w') as f:
            json.dump(self.config, f)

    def load_model(self, checkpoint=None):
        '''
        Load the learned MLP model from the disk.

        Parameters
        ----------
        checkpoint: str 
            The checkpoint folder to load the learned model.
            The checkpoint under this folder should be `model.ckpt`.
        '''
        if checkpoint is None:
            ckpt_dir = 'save_model'
            checkpoint = os.path.join(ckpt_dir, 'model.ckpt')
        else:
            checkpoint = os.path.join(checkpoint, 'model.ckpt')
        self.model = torch.load(checkpoint)

        ckpt_dir = os.path.dirname(checkpoint)
        config_filename = os.path.join(ckpt_dir, 'config.json')
        with open(config_filename, 'r') as f:
            self.config = json.load(f)

    def _evaluate(self, dataloader):
        res = self._predict_on_dataloader(dataloader)
        nctids, ypred, ytrue = res['nctid'], res['ypred'], res['label']
        return {'auc': roc_auc_score(ytrue, ypred)}

    def _predict_on_dataloader(self, test_loader):
        nctids = []
        ypred = []
        labels = []
        for nctid_lst, label_vec, smiles_lst2, icdcode_lst3, criteria_lst in test_loader:
            features = []
            for smiles_lst in smiles_lst2: 
                mol_lst = [Chem.MolFromSmiles(smiles) for smiles in smiles_lst]
                mol_lst = list(filter(lambda x:x is not None, mol_lst))
                fp_lst = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512) for mol in mol_lst]
                feature = np.zeros(512)
                for fp in fp_lst:
                    arr = np.zeros((0,), dtype=np.int8)
                    DataStructs.ConvertToNumpyArray(fp,arr)
                    feature += arr
                feature = feature.reshape(1,-1)
                features.append(feature)
            features = np.concatenate(features, 0)
            features = Variable(torch.from_numpy(features))
            y = self.model(features)
            y = torch.sigmoid(y)
            nctids.extend(nctid_lst)
            ypred.extend(y.tolist())
            if label_vec is not None:
                labels.extend(label_vec.tolist())
        res_dict = {'nctid': nctids, 'ypred': ypred, 'label': None}
        if len(labels) > 0:
            res_dict['label'] = labels
        return res_dict


