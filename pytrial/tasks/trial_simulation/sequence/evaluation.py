'''
Evaluation methods for the simulated sequence of patient records.
'''
import pdb
from copy import deepcopy

from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np

from pytrial.tasks.indiv_outcome.sequence import RNN
from pytrial.data.patient_data import SeqPatientCollator
from pytrial.data.utils import concat_dataset, split_dataset
from pytrial.tasks.indiv_outcome.trainer import IndivSeqTrainer

from ..data import SequencePatient
from .base import transform_sequence_to_table
from .base import transform_table_to_sequence


__all__ = ['RNNPrivacyDetection', 'RNNUtilityDetection', 'DimWiseFidelity']


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


class SequenceFidelityMetric(SequenceMetric):
    '''
    Base class for evaluating the fidelity of synthetic patient records.
    '''
    @staticmethod
    def _build_dataset(real_data, syn_data):
        '''
        BUild a dataset for comparing the distribution of real and synthetic patient records.

        Parameters
        ----------
        real_data: pytrial.tasks.trial_simulation.data.PatientSequence
            A PatientSequence object that contains real patient records.

        syn_data: pytrial.tasks.trial_simulation.data.PatientSequence
            A PatientSequence data structure containing the synthetic patient data.
        '''
        order = real_data.metadata['visit']['order']
        voc = real_data.metadata['voc']
        # build tabular format out of real and syn data
        real_df = transform_sequence_to_table(real_data.visit, order, voc)
        syn_df = transform_sequence_to_table(syn_data.visit, order, voc)
        return real_df, syn_df


class DimWiseFidelity(SequenceFidelityMetric):
    '''
    Evaluate the fidelity of synthetic patient records by computing the dimension-wise probability distributions 
    of the real and synthetic patient records. Overall results shown as the r-value.
    '''
    @classmethod
    def compute(self, real_data, syn_data, event_type=None, plot=False, save_path=None):
        '''
        Compute the fidelity score. Report the r-value of the dimension-wise probability distributions of real and synthetic patient records.

        Parameters
        ----------
        real_data: pytrial.tasks.trial_simulation.data.PatientSequence
            A PatientSequence object that contains real patient records.
        
        syn_data: pytrial.tasks.trial_simulation.data.PatientSequence
            A PatientSequence data structure containing the synthetic patient data.
        
        event_type: str or list[str] or None
            The type of event to compute the fidelity score. Default is None, which means all events are considered.
            If `event_type` is not None, then the fidelity score is computed for the specified event type list.

        plot: bool
            Whether to plot the scatters distribution. Default is False.

        save_path: str
            The path to save the plot. Default is None. Only used when `plot` is True.

        Usage
        -----
        >>> from pytrial.tasks.trial_simulation.metrics import DimWiseFidelity
        >>> # real_data and syn_data are PatientSequence objects for all types of events
        >>> fidelity_score = DimWiseFidelity.compute(real_data, syn_data)
        >>> # real_data and syn_data are PatientSequence objects for medication events
        >>> fidelity_score = DimWiseFidelity.compute(real_data, syn_data, event_type='medication', plot=True)
        '''
        # build a dataset with labeled real and syndata
        real_df, syn_df = self._build_dataset(real_data, syn_data)
        
        if event_type is None or event_type == 'all':
            event_types = list(real_data.metadata['voc'].keys())
        else:
            event_types = event_type if isinstance(event_type, list) else [event_type]

        outputs = {}
        syn_prob_outputs = {}
        real_prob_outputs = {}
        for etype in event_types:
            # compute the real marginal probability distribution
            real_probs = self._compute_marginal_probability(real_df, event_type=etype)
            # compute the synthetic marginal probability distribution
            syn_probs = self._compute_marginal_probability(syn_df, event_type=etype)
            # compute the r-value
            r_value = np.corrcoef(real_probs, syn_probs)[0, 1]
            # save the r-value
            outputs[etype] = r_value
            # save the real and synthetic probability distributions
            syn_prob_outputs[etype] = syn_probs
            real_prob_outputs[etype] = real_probs

        # plot the distribution
        if plot:
            for etype in event_types:
                r_value = outputs[etype]
                real_probs = real_prob_outputs[etype]
                syn_probs = syn_prob_outputs[etype]
                self._plot_scatter(real_probs, syn_probs, r_value)
                if save_path is not None:
                    plt.savefig(save_path + f'/{etype}.png')
                plt.show()
        return outputs

    @staticmethod
    def _compute_marginal_probability(df, event_type=None):
        '''
        Compute the marginal probability distribution of a dataframe.
        '''
        binary_probability = []
        if event_type is not None:
            columns = [c for c in df.columns if event_type in c]
        else:
            columns = df.columns
        for c in columns:
            binary_probability.append(df[c].mean())
        return binary_probability
    
    @staticmethod
    def _plot_scatter(real_probs, syn_probs, r_value):
        '''
        Plot the scatter distribution of real and synthetic data.
        '''
        import matplotlib.pyplot as plt
        import matplotlib
        plt.rcParams["figure.figsize"] = (3,3)
        SMALL_SIZE = 14
        matplotlib.rc('font', size=SMALL_SIZE)
        matplotlib.rc('axes', titlesize=SMALL_SIZE)
        plt.scatter(real_probs, syn_probs, label=f'r = {np.round(r_value, 2)}')
        plt.legend(loc=1)