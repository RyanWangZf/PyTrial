import pdb
from collections import defaultdict
import joblib
import random
import os

from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import numpy as np

from pytrial.data.patient_data import SequencePatientBase, SeqPatientCollator
from pytrial.utils.check import check_checkpoint_file, check_model_dir, check_model_config_file, make_dir_if_not_exist
from ..data import SequencePatient
from .base import SequenceSimulationBase
from .base import transform_sequence_to_table
from .base import transform_table_to_sequence


def get_neighbors(x, knn, idx, k):
    # knn = NearestNeighbors(n_neighbors=k, algorithm='auto')
    vector = x[idx]
    return knn.kneighbors([vector], k, return_distance=False)

def generate_data(df, idx):
    neighbor_ids = get_neighbors(idx)
    new_sample = [df.iloc[idx, 0], df.iloc[idx, 1], df.iloc[idx, 2]]
    for j in range(3, len(df.columns)):
        new_sample.append(df.iloc[random.choice(neighbor_ids[0]), j])
    return new_sample


class BuildModel:
    def __new__(self, config):
        model = NearestNeighbors(n_neighbors=config['k'], algorithm='auto')
        return model

class KNNSampler(SequenceSimulationBase):
    '''
    Implement an RNN based GAN model for longitudinal patient records simulation. The GAN part was proposed by Beigi et al. [1]_ to generate synthetic tabular patient records.
    We adapt it to generate sequential records by finding the nearest neighbor on the visit-level.

    Parameters
    ----------
    order: list[str]
        The order of event types in each visits, e.g., ``['diag', 'prod', 'med']``.
        Visit = [diag_events, prod_events, med_events], each event is a list of codes.

    voc: dict
        A dictionary that maps the event type to the corresponding vocabulary.

    k: int
        Number of nearest neighbors to use for synthetic data generation.
    
    tsne_dims: int
        Dimension of t-SNE embedding. Used to encode the visit events into a vector.

    Notes
    -----
    .. [1] Beigi, M., Shafquat, A., Mezey, J., & Aptekar, J. W. (2022) Synthetic Clinical Trial Data while Preserving Subject-Level Privacy. In NeurIPS 2022 Workshop on Synthetic Data for Empowering ML Research.
    '''
    def __init__(self,
        order=None,
        voc=None,
        k=3,
        tsne_dims=2,
        experiment_id='trial_simulation.sequence.knn',
        ):
        super().__init__(experiment_id)
        self.config = {
            'k': k,
            'tsne_dims': tsne_dims,
            'order':order,
            'voc': voc,
        }
        self._build_model()
        assert order is not None, 'Please specify the order of event types in each visit.'
        assert voc is not None, 'Please specify the vocabulary event types.'


    def fit(self, train_data, val_data=None):
        '''
        Fit KNN model on the input training sequential patient records.

        Parameters
        ----------
        train_data: SequencePatient
            A `SequencePatient` contains patient records where 'v' corresponds to
            visit sequence of different events.

        val_data: None
            Not used in this model. Only for compatibility with other models.
        '''
        self._input_data_check(train_data)
        self._fit_model(train_data)

    def load_model(self, checkpoint=None):
        '''
        Load the learned model from the disk.

        Parameters
        ----------
        checkpoint: str
            - If a directory, the only checkpoint file `.model` will be loaded.
            - If a filepath, will load from this file;
            - If None, will load from `self.checkout_dir`.
        '''
        if checkpoint is None:
            checkpoint = self.checkout_dir
        checkpoint_filename = check_checkpoint_file(checkpoint, suffix='model')
        model = joblib.load(checkpoint_filename)
        self.__dict__.update(model.__dict__)

    def save_model(self, output_dir=None):
        '''
        Save the fitted model and the raw data to the disk for later use.

        Parameters
        ----------
        output_dir: str
            The directory to save the model and data. If `None`, save to the default directory.
            `self.checkout_dir`.
        '''
        if output_dir is None:
            output_dir = self.checkout_dir
        make_dir_if_not_exist(output_dir)
        ckpt_path = os.path.join(output_dir, 'knnsampler.model')
        joblib.dump(self, ckpt_path)
    
    def predict(self, n_per_sample=None, return_tensor=False):
        '''
        Generate synthetic patient records by sampling and perturbing the nearest neighbors of the input patient records.

        Parameters
        ----------        
        n_per_sample: int
            Number of synthetic records to generate for each patient. If `None`, generate one synthetic record for each patient.
        
        return_tensor: bool
            If `True`, return the synthetic records as a tensor format (n, n_visit, n_event), good for later predictive modeling.
            If `False`, the synthetic records are in the same format as the input patient records.
        '''
        if n_per_sample is None:
            n_per_sample = 1
        assert isinstance(n_per_sample, int), 'Please specify the number of synthetic records to generate for each patient. Should be an integer. Get {} instead.'.format(type(n_per_sample))
        output = self._generate(n_per_sample)

        if return_tensor:
            # transform the output into tensor format
            output = self._transform_to_tensor(output)
        else:
            # transform the output into SequencePatient format
            output = self._transform_to_sequence_patient(output)
        return output

    def _build_model(self):
        self.knn = BuildModel(self.config)
        self.tsne = TSNE(n_components=self.config['tsne_dims'], random_state=0)
        self.data = None

    def _fit_model(self, train_data):
        # transform visit sequence into tabular format
        visit_df = transform_sequence_to_table(train_data.visit, 
            self.config['order'], self.config['voc'])
        visit_df.set_index(['pid', 'vid'], inplace=True)

        # fit TSNE model
        self.tsne.fit(visit_df)

        # transform visit sequence into vector
        z = self.tsne.embedding_

        # fit KNN model
        self.knn.fit(z)

        # save the fitted data for later use of generating synthetic data
        self.data = visit_df

        # save the other information of the input data
        self._input_data = train_data

    def _generate(self, n_per_sample):
        # find the nearest neighbors
        z = self.tsne.embedding_
        df = self.data.copy().reset_index()
        syn_df = pd.DataFrame(columns= df.columns)

        for j in range(n_per_sample):
            for i in tqdm(range(len(z)), desc='Generating synthetic data {}/{} per patient'.format(j+1, n_per_sample)):
                neighbor_ids = get_neighbors(z, self.knn, i, self.config['k'])
                # generate new sample
                # randomly sample one of the neighbors for each column
                new_sample = [df.iloc[i, 0], df.iloc[i, 1]]
                df_neighbors = df.iloc[neighbor_ids[0]]
                sampled = df_neighbors.apply(lambda x: np.random.choice(x), axis=0)
                new_sample.extend(sampled[2:].values)
                syn_df.loc[len(syn_df)] = new_sample

        return syn_df

    def _transform_to_tensor(self, df):
        # transform the output into tensor format
        # [n, n_visit, n_event]
        df['vid'] = df['vid'].astype(int)
        df['pid'] = df['pid'].astype(int)
        outputs = np.zeros((df['pid'].nunique(), df['vid'].max()+1, len(df.columns)-2))
        df.set_index(['pid', 'vid'], inplace=True)
        for pid in df.index.get_level_values(0).unique():
            vals = df.loc[pid].values
            outputs[pid, :vals.shape[0], :] = vals
        return outputs

    def _transform_to_sequence_patient(self, df):
        # transform the output into SequencePatient format
        df['vid'] = df['vid'].astype(int)
        df['pid'] = df['pid'].astype(int)
        visits, pids = transform_table_to_sequence(df, self.config['order'], self.config['voc'])

        # like pid to self._input_data's features
        labels = self._input_data.__dict__.get('label', None)
        if labels is not None:
            labels = [labels[pid] for pid in pids]
        features = self._input_data.__dict__.get('feature', None)
        if features is not None:
            features = [features[pid] for pid in pids]
        
        output = SequencePatient(
            data={
                'v': visits,
                'y': labels,
                'x': features,
            },
            metadata={
                'visit': {'mode':'dense'},
                'label': {'mode':'tensor'},
                'voc': self.config['voc'],
                'n_num_features': self._input_data.metadata.get('n_num_features', None),
                'cat_cardinalities': self._input_data.metadata.get('cat_cardinalities', None),
            }
        )
        return output

    def _input_data_check(self, inputs):
        assert isinstance(inputs, SequencePatientBase), f'`trial_simulation.sequence` models require input training data in `SequencePatientBase`, find {type(inputs)} instead.'
