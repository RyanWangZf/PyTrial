import joblib
import os
import json

from rdkit import Chem 
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
import xgboost
import numpy as np 

from .base import TrialOutcomeBase 
from .model_utils.utils import trial_collate_fn

class XGBoost(TrialOutcomeBase):
    '''
    Implement XGBoost model for clinical trial outcome prediction. 

    Parameters
    ----------
    n_estimators: int
        number of trees in the forest
    
    max_depth: int
        maximum depth of the tree
    
    reg_lambda: float
        L2 regularization term on weights
    
    eval_metric: {'auc', 'logloss'}
        evaluation metric for validation data, default is AUC.
    '''
    def __init__(self,
        n_estimators=100,
        max_depth=3,
        reg_lambda=0,
        eval_metric='auc',
        ):
        super(XGBoost, self).__init__()
        self.model = xgboost.XGBClassifier(n_estimators=n_estimators, 
            max_depth=max_depth, reg_lambda=reg_lambda, eval_metric=eval_metric)
        self.config = {'n_estimators': n_estimators, 'max_depth': max_depth, 'reg_lambda': reg_lambda, 'eval_metric': eval_metric}

    def feature(self, data_loader):
        labels = []
        features = []
        nctids = []
        for nctid_lst, label_vec, smiles_lst2, icdcode_lst3, criteria_lst in data_loader:
            labels.extend(label_vec.tolist())
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
        eval_set = None

        # build dataloader using train_data
        train_loader = self._build_dataloader_from_dataset(train_data, 
            num_workers=0, batch_size=32, shuffle=True,
            collate_fn=trial_collate_fn)
        _, features, labels = self.feature(train_loader)

        if valid_data is not None:
            valid_loader = self._build_dataloader_from_dataset(valid_data, 
                num_workers=0, batch_size=32, shuffle=False,
                collate_fn=trial_collate_fn)
            _, valid_features, valid_labels = self.feature(valid_loader)
            eval_set = [(valid_features, valid_labels)]

        self.model.fit(features, labels, eval_set=eval_set, verbose=True)

    def predict(self, test_data):
        '''
        Make clinical trial outcome predictions.
        
        Parameters
        ----------
        test_data: TrialOutcomeDatasetBase
            Testing data, should be a `TrialOutcomeDatasetBase` object.
        '''
        # build dataloader using test_data
        testloader = self._build_dataloader_from_dataset(test_data, 
            num_workers=0, batch_size=32, shuffle=False,
            collate_fn=trial_collate_fn)

        nctids, features, labels = self.feature(testloader)
        ypred = self.model.predict_proba(features)
        ypred = ypred[:, 1].tolist() 
        prediction = list(zip(nctids, ypred))
        return prediction

    def save_model(self, output_dir=None):
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
        joblib.dump(self.model, filename)

        config_filename = os.path.join(output_dir, 'config.json')
        with open(config_filename, 'w') as f:
            json.dump(self.config, f)


    def load_model(self, checkpoint=None):
        '''
        Load the learned model from the disk.

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
        self.model = joblib.load(checkpoint)

        ckpt_dir = os.path.dirname(checkpoint)
        config_filename = os.path.join(ckpt_dir, 'config.json')
        with open(config_filename, 'r') as f:
            self.config = json.load(f)