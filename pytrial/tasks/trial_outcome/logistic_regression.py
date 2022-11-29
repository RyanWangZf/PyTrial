'''
Implement Logistic Regression model for clinical trial outcome prediction
'''
import os
import pdb

import numpy as np 
from sklearn.linear_model import LogisticRegression as lr_model
import joblib
from rdkit import Chem 
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs

from pytrial.data.trial_data import TrialOutcomeDatasetBase
from .base import TrialOutcomeBase
from .model_utils.utils import trial_collate_fn

class LogisticRegression(TrialOutcomeBase):
    '''
    Implement Logistic Regression model for clinical trial outcome prediction. 

    Parameters
    ----------
    C: float
        Regularization strength for l2 norm; must be a positive float. 
        Like in support vector machines, smaller values specify weaker regularization.

    dual: bool
        Dual or primal formulation. Dual formulation is only implemented for l2 penalty with liblinear solver. 
        Prefer `dual=False` when `n_samples > n_features`.
    
    solver: {'newton-cg','lbfgs','liblinear','sag','saga'}
        Algorithm to use in the optimization problem. default='lbfgs'.

    max_iter: int (default=100)
        Maximum number of iterations taken for the solvers to converge.
    '''
    def __init__(
        self, 
        C=1.0, 
        dual=False,
        solver='lbfgs', 
        max_iter=100
        ):

        super(LogisticRegression, self).__init__()
        self.model = lr_model(
            C = C,
            dual = dual,
            solver = solver,
            max_iter = max_iter,
        )

    def fit(self, train_data, valid_data):
        '''
        Train logistic regression model to predict clinical trial outcomes.

        Parameters
        ----------
        train_data: TrialOutcomeDatasetBase
            Training data, should be a `TrialOutcomeDatasetBase` object.
        
        valid_data: TrialOutcomeDatasetBase
            Validation data, should be a `TrialOutcomeDatasetBase` object. Ignored for logistic regression model.
            Keep this parameter for compatibility with other models.

        '''
        # build dataloader using train_data
        trainloader = self._build_dataloader_from_dataset(train_data, 
            num_workers=0, batch_size=32, shuffle=True,
            collate_fn=trial_collate_fn)

        _, features, labels = self.feature(trainloader)
        
        self.model.fit(features, labels)

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

    def save_model(self, output_dir = None):
        '''
        Save the learned logistic regression model to the disk.

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



