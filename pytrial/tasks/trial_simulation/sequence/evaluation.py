'''
Evaluation methods for the simulated sequence of patient records.
'''
import pdb
from copy import deepcopy

from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from pytrial.tasks.indiv_outcome.sequence import RNN
from pytrial.data.patient_data import SeqPatientCollator
from pytrial.data.utils import concat_dataset, split_dataset
from pytrial.tasks.indiv_outcome.trainer import IndivSeqTrainer
from ..data import SequencePatient

__all__ = ['RNNPrivacyDetection', 'RNNUtilityDetection']


class SequenceMetric:
    '''
    Base class for evaluation of the simulated sequence of patient records.
    '''
    @staticmethod
    def _build_dataset(real_data, syn_data):
        raise NotImplementedError

    @staticmethod
    def _build_model(train_data, test_data):
        '''
        Build a detection model.
        '''
        raise NotImplementedError
    
    @classmethod
    def compute(self, real_data, syn_data):
        raise NotImplementedError

class SequencePrivacyMetric(SequenceMetric):
    '''
    Base class for Machine Learning detection based metrics for sequential synthetic patient records.

    These metrics build a Machine Learning Classifier that learns to tell the synthetic
    data apart from the real data.

    The output is the binary classification AUC.
    '''
    @staticmethod
    def _build_dataset(real_data, syn_data):
        '''
        Build a dataset for training a ML model.

        Parameters
        ----------
        real_data: pytrial.tasks.trial_simulation.data.PatientSequence
            A PatientSequence object that contains real patient records.

        syn_data: pytrial.tasks.trial_simulation.data.PatientSequence
            A PatientSequence data structure containing the synthetic patient data.

        Returns
        -------
        train_set: torch.utils.data.Dataset
            A dataset for training a ML model.
        
        test_set: torch.utils.data.Dataset
            A dataset for testing a ML model.
        '''
        real_syn_data = concat_dataset(real_data, syn_data)
        train_data, test_data = split_dataset(real_syn_data, split=0.8)
        return train_data, test_data
    
    @staticmethod
    def _build_model(train_data, test_data):
        '''
        Build a detection model.
        '''      
        raise NotImplementedError


class RNNPrivacyDetection(SequencePrivacyMetric):
    '''
    Try to discriminate real and synthetic patient records using an RNN based neural network.
    The output score is AUC evaluated on the prediction.
    So the biggest score will be AUC=1.0, which means the model can correctly discriminate real and synthetic data.
    The lowest score will be AUC=0.5, which means the model cannot discriminate real and sytnethic data.

    There are multiple interpretations of the score. 
    A low score (low AUC) can indicates high synthetic data quality as well as low privacy. 
    A high score (high AUC) can indicate low synthetic data quality as well as high privacy. 
    '''

    @classmethod
    def compute(self, real_data, syn_data, device='cpu'):
        '''
        Compute the detection score.

        Parameters
        ----------
        real_data: pytrial.data.patient_sequence.PatientSequence
            A PatientSequence object that contains real patient records.

        syn_data: pytrial.data.patient_sequence.PatientSequence
            A PatientSequence data structure containing the synthetic patient data.

        device: str
            The device to run the model on. Default is 'cpu'.

        Returns
        -------
        float: The detection score.
        '''
        assert isinstance(real_data, SequencePatient), '`real_data` must be a `SequencePatient` object.'
        assert isinstance(syn_data, SequencePatient), '`syn_data` must be a `SequencePatient` object.'

        # assign labels to real and syn data
        real_data = deepcopy(real_data)
        syn_data = deepcopy(syn_data)
        real_data.label = [1] * len(real_data) # 1 is real data label
        syn_data.label = [0] * len(syn_data) # 0 is syn data label

        # build a dataset with labeled real and syndata
        train_data, test_data = self._build_dataset(real_data, syn_data)

        # train a RNN model
        model = self._build_model(train_data, test_data, device=device)

        # evaluate the model
        pred = model.predict(test_data)
        label = test_data.get_label()
        auc = roc_auc_score(label, pred)
        return auc

    @staticmethod
    def _build_model(train_data, test_data, device='cpu'):
        '''
        Build a RNN model for binary classification.

        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
            A dataloader for training a ML model.

        Returns
        -------
        A ML model for binary classification of synthetic and real patient records.
        '''
        model = RNN(
            vocab_size=[len(train_data.metadata['voc'][o]) for o in train_data.metadata['visit']['order']],
            orders=train_data.metadata['visit']['order'],
            mode='binary',
            bidirectional=False,
            epochs=10,
            batch_size=64,
            device=device,
            )
        model.fit(train_data=train_data, valid_data=test_data)
        return model


class SequenceUtilityMetric(SequenceMetric):
    '''
    Base class for Machine Learning based models to learn from the synthetic patient records.
    And predict the outcome of the real patient records.
    Compared with the model learning from the real patient records.
    The output is the AUC of the prediction from the real-data-model and the synthetic-data-model.
    '''
    @staticmethod
    def _build_dataset(real_data, syn_data):
        '''
        Build a dataset for training a ML model.

        Parameters
        ----------
        real_data: pytrial.tasks.trial_simulation.data.PatientSequence
            A PatientSequence object that contains real patient records.

        syn_data: pytrial.tasks.trial_simulation.data.PatientSequence
            A PatientSequence data structure containing the synthetic patient data.

        Returns
        -------
        train_set: torch.utils.data.Dataset
            A dataset for training a ML model.
        
        test_set: torch.utils.data.Dataset
            A dataset for testing a ML model.
        '''
        train_real_data, test_real_data = split_dataset(real_data, split=0.7)
        train_syn_data, test_syn_data = split_dataset(syn_data, split=0.7)
        return [(train_real_data, test_real_data), (train_syn_data, test_syn_data)]
    
    @staticmethod
    def _build_model(train_data, test_data):
        '''
        Build a detection model.
        '''
        raise NotImplementedError


class RNNUtilityDetection(SequenceUtilityMetric):
    @staticmethod
    def _build_model(train_data, test_data, device='cpu'):
        '''
        Build a detection model.
        '''
        model = RNN(
            vocab_size=[len(train_data.metadata['voc'][o]) for o in train_data.metadata['visit']['order']],
            orders=train_data.metadata['visit']['order'],
            mode='binary',
            bidirectional=False,
            epochs=50,
            batch_size=128,
            weight_decay=0,
            rnn_type='lstm',
            n_rnn_layer=1,
            learning_rate=5e-4,
            device=device,
            )
        model.fit(train_data=train_data, valid_data=test_data)
        return model

    @classmethod
    def compute(self, real_data, syn_data, device='cpu'):
        '''
        Compute the detection score. Report the performance of the model trained on real data (real-data-model) 
        and the model trained on synthetic data (syn-data-model).
        Performance is evaluated on the test set of the real data.

        If the syn-data-model performs equally well or better than the real-data-model, then the synthetic data has high utility.
        
        Parameters
        ----------
        real_data: pytrial.tasks.trial_simulation.data.PatientSequence
            A PatientSequence object that contains real patient records.

        syn_data: pytrial.tasks.trial_simulation.data.PatientSequence
            A PatientSequence data structure containing the synthetic patient data.

        device: str
            The device to run the model. Default is 'cpu'.

        Returns
        -------
        float: The detection score of real-data-model and synthetic-data-model.
        '''
        assert isinstance(real_data, SequencePatient), '`real_data` must be a `SequencePatient` object.'
        assert isinstance(syn_data, SequencePatient), '`syn_data` must be a `SequencePatient` object.'

        # build a dataset with labeled real and syndata
        real_data_traintest, syn_data_traintest = self._build_dataset(real_data, syn_data)

        # train a RNN model
        real_model = self._build_model(real_data_traintest[0], real_data_traintest[1])
        syn_model = self._build_model(syn_data_traintest[0], syn_data_traintest[1], device=device)

        # evaluate the model
        outputs = {'real-data-model-auc': None, 'syn-data-model-auc': None}
        test_data = real_data_traintest[1] 

        pred = real_model.predict(test_data)
        label = test_data.get_label()
        auc = roc_auc_score(label, pred)
        outputs['real-data-model-auc'] = auc

        pred = syn_model.predict(test_data)
        label = test_data.get_label()
        auc = roc_auc_score(label, pred)
        outputs['syn-data-model-auc'] = auc

        return outputs