'''
Topic discovery for clinical trial data.
'''
import pdb
import copy

import pandas as pd
import numpy as np
from torch import nn
from sklearn.cluster import KMeans

import warnings
from trial2vec import Trial2Vec
from trial2vec import download_embedding
warnings.filterwarnings('ignore')

from .data_utils import icdcode_text_2_lst_of_lst, list_text_2_list, smiles_txt_to_lst
from .data_utils import flatten_stacked_list
from .data_utils import split_protocol
from .data_structure import Trial, Topic, SequenceTrial

class TrialTopicDiscovery:
    '''
    Input a csv of clinical trial data, output a dataset consisting of multiple sequences of 
    clinical trials. Each sequence is a list of clinical trials under this topic ordered with their timestamps.
    '''
    def __init__(self,
        num_topics=50,
        random_seed=42,
        ) -> None:
        super().__init__()
        self.num_topics = num_topics
        self.random_seed = random_seed
        self.km = None
        # trial2vec = Trial2Vec()
        # self.trial2vec.from_pretrained()
        self.trial2vec = download_embedding()
    
    def fit(self, trial_data):
        '''
        Fit the topic discovery model.

        Parameters
        ----------
        trial_data : pd.DataFrame
            A dataframe of clinical trial data.
        '''
        _ = self.topic_classification(trial_data)
    
    def transform(self, trial_data):
        '''
        Transform a list of dataframes into a `SequenceTrial` object.

        Parameters
        ----------
        trial_data : pd.DataFrame
            A dataframe of clinical trial data.

        Returns
        -------
        seqtrial : SequenceTrial
            A `SequenceTrial` object.
        '''
        trial_data.fillna('[PAD]', inplace=True)
        datalist = self.topic_classification(trial_data)
        seqtrial = self.create_sequence_trial(datalist)
        return seqtrial

    def fit_transform(self, trial_data):
        '''
        Fit the topic discovery model and transform the trial data into a dataset of sequences of trials.

        Parameters
        ----------
        trial_data : pd.DataFrame
            A dataframe of clinical trial data.

        Returns
        -------
        seqtrial : SequenceTrial
            A dataset consisting of multiple topics of clinical trials. Each topic is a list of clinical trials under this topic ordered with their timestamps.
        '''
        # at present we just use the trial2vec embeddings for topic discovery by clustering
        _ = self.fit(trial_data)

        # create a dataset of topics
        seqtrial = self.transform(trial_data)
        return seqtrial

    def topic_classification(self, trial_data):
        '''
        Create a list of dataframes, each of which is a dataframe of clinical trial data for a given topic id.

        Can be replaced by other topic classification methods. At present, clustering is used.
        '''
        datalist = self._split_by_clustering(trial_data)
        return datalist

    def create_sequence_trial(self, datalist):
        '''
        Create a `SequenceTrial` object from a list of dataframes.

        Parameters
        ----------
        data_list : List[[str, pd.DataFrame]]
            A list of dataframes, each of which is a dataframe of clinical trial data for a given topic id.

        Returns
        -------
        seqtrial : SequenceTrial
            A `SequenceTrial` object.
        '''
        topics = []
        for topic_id, df in datalist:
            df_list = self._preprocess_df(df)
            topic = self._build_topic(topic_id, df_list)
            topics.append(topic)
        seqtrial = SequenceTrial(topics)
        return seqtrial

    def add_trials(self, seqtrial, trial_data):
        '''
        Add new trials to a given `TrialSequence` object.
        Will classify trials to each topic and add them to the corresponding topic/timestamp.

        Parameters
        ----------
        seqtrial : TrialSequence
            A `TrialSequence` object.

        trial_data : pd.DataFrame
            A dataframe of clinical trial data.
        '''
        new_seqtrial = copy.deepcopy(seqtrial)
        data_list = self.topic_classification(trial_data)
        topic_list = []
        for topic_id, df in data_list:
            df_list = self._preprocess_df(df)
            topic = self._build_topic(topic_id, df_list)
            topic_list.append(topic)
        new_seqtrial.add(topic_list)
        return new_seqtrial

    def _split_by_clustering(self, df):
        '''
        Split the dataframe trials by clustering based on Trial2Vec pretrained embeddings.

        Parameters
        ----------
        df : pd.DataFrame
            A dataframe of clinical trial data.

        Returns
        -------
        df_list : List[[str, pd.DataFrame]]
            A list of dataframes, each of which is a dataframe of clinical trial data for a given year.
        '''
        # get the embeddings
        emblist = [self.trial2vec[nctid] for nctid in df['nctid']]

        # cluster the embeddings
        if self.km is None:
            self.km = KMeans(n_clusters=self.num_topics, random_state=self.random_seed)
            self.km.fit(emblist)
            topic_ids = self.km.predict(emblist)
        else:
            topic_ids = self.km.predict(emblist)

        # split the dataframe by topic ids
        df_list = []
        for topic_id in range(self.num_topics):
            df_list.append([str(topic_id), df[topic_ids == topic_id]])
        return df_list

    def _split_by_year(self, df):
        '''
        Split the dataframe by year.

        Parameters
        ----------
        df : pd.DataFrame
            A dataframe of clinical trial data.

        Returns
        -------
        df_list : List[pd.DataFrame]
            A list of dataframes, each of which is a dataframe of clinical trial data for a given year.
        '''
        df_list = []
        df = df.sort_values(by='year')
        for year, df_year in df.groupby('year'):
            df_list.append([str(year), df_year])
        return df_list

    def _preprocess_df(self, df):
        ecs = df['criteria'].apply(split_protocol)
        df['inclusion_criteria'] = ecs.apply(lambda x: x[0])
        df['exclusion_criteria'] = ecs.apply(lambda x: x[1] if len(x) > 1 else ['[PAD]'])
        df.drop(columns=['criteria'], inplace=True)
        df['icdcodes'] = df['icdcodes'].apply(icdcode_text_2_lst_of_lst)
        df['diseases'] = df['diseases'].apply(list_text_2_list)
        df['drugs'] = df['drugs'].apply(list_text_2_list)
        df['smiless'] = df['smiless'].apply(smiles_txt_to_lst)

        # get year and sort and split by year
        df['study_first_submitted_date'] = pd.to_datetime(df['study_first_submitted_date'])
        df['year'] = df['study_first_submitted_date'].dt.year
        df_list = self._split_by_year(df)
        return df_list
    
    def _build_topic(self, topic_id, df_list):
        '''
        Build a topic object from a list of dataframes.

        Parameters
        ----------
        topic_id : str
            The topic id.

        df_list : List[pd.DataFrame]
            A list of dataframes, each of which is a dataframe of clinical trial data for a given year.

        Returns
        -------
        topic : Topic
            A topic object.
        '''
        trials = []
        for year, df_year in df_list:
            trial_year = []
            for _, row in df_year.iterrows():
                trial = self._build_trial(row)
                trial_year.append(trial)
            trials.append(trial_year)
        topic = Topic(topic_id, trials)
        return topic
    
    def _build_trial(self, row):
        '''
        Build a trial object from a row of a dataframe.

        Parameters
        ----------
        row : pd.Series
            A row of a dataframe.

        Returns
        -------
        trial : Trial
            A trial object.
        '''
        row = row.to_dict()
        trial = Trial(
            nctid=row.pop('nctid'),
            title=row.pop('title'),
            label=row.pop('label'),
            status=row.pop('status'),
            year=row.pop('year'),
            **row,
            )
        return trial