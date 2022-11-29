'''
Implement Doc2Vec model for clinical trial similarity search.
'''
import pdb
import os
import warnings
import json

import pandas as pd
import numpy as np
import torch
from torch import nn
import gensim
from gensim.models.doc2vec import Doc2Vec as doc2vec_model
from gensim.models.doc2vec import TaggedDocument

from pytrial.utils.check import check_checkpoint_file, check_model_config_file
from .base import TrialSearchBase

warnings.filterwarnings('ignore')

class BuildModel(nn.Module):
    def __init__(self,
        train_corpus,
        emb_dim=128,
        epochs=10,
        window=5,
        min_count=None,
        max_vocab_size=None,
        num_workers=None,
        ) -> None:
        super().__init__()

        # build gensim.Doc2vec model
        self.model = doc2vec_model(
            documents=train_corpus,
            vector_size=emb_dim,
            min_count=min_count,
            window=window,
            max_vocab_size=max_vocab_size,
            epochs=epochs,
            workers=num_workers,
            )

    def forward(self, x):
        x_w = pd.Series(x).apply(lambda x_: gensim.utils.simple_preprocess(x_))
        embs = x_w.apply(lambda x: self.model.infer_vector(x)[None])
        embs = np.concatenate(embs.values, 0)
        return embs

class Doc2Vec(TrialSearchBase):
    '''
    Implement the Doc2Vec model for trial document similarity search.

    Parameters
    ----------
    emb_dim: int, optional (default=128)
        Dimensionality of the embedding vectors.

    epochs: int, optional (default=10)
        Number of iterations (epochs) over the corpus. Defaults to 10 for Doc2Vec.

    window: int, optional (default=5)
        The maximum distance between the current and predicted word within a sentence.

    min_count: int, optional (default=5)
        Ignores all words with total frequency lower than this.

    max_vocab_size: int, optional
        Limits the RAM during vocabulary building; if there are more unique words than this, then prune the infrequent ones.
        Every 10 million word types need about 1GB of RAM. Set to None for no limit.

    num_workers: int, optional (default=4)
        Use these many worker threads to train the model (=faster training with multicore machines).

    experiment_id: str, optional (default='test')
        The name of current experiment.
    '''
    def __init__(self,
        emb_dim=128,
        epochs=10,
        window=5,
        min_count=5,
        max_vocab_size=None,
        num_workers=4,
        experiment_id='test',
        ) -> None:
        super().__init__(experiment_id=experiment_id)
        self.emb_dim=emb_dim
        self.epochs=epochs
        self.window=window
        self.min_count=min_count
        self.max_vocab_size=max_vocab_size
        self.num_workers=num_workers

    def fit(self, train_data, valid_data=None):
        '''Train the doc2vec model to get document embeddings for trial search.

        Parameters
        ----------
        train_data: dict

            Training corpus for the model. 
            
            - `x`: a dataframe of trial documents.
            
            - `fields`: optional, the fields of documents to use for training. If not given, the model uses all fields in `x`.

            - `tag`: optional, the field in `x` that serves as unique identifiers. Typically it is the `nct_id` of each trial. If not given, the model takes integer tags.

            train_data =

            {
            
            'x': pd.DataFrame,

            'fields': list[str],
        
            'tag': str,

            }


        valid_data: Ignored.
            Not used, present here for API consistency by convention.
        '''
        self._input_data_check(train_data)
        self._build_model(train_data)
        self._save_checkpoint(state=self.model, epoch_id=self.epochs)

    def encode(self, inputs):
        '''
        Encode input documents and output the document embeddings.

        Parameters
        ----------
        inputs: dict
            The documents which are to be encoded.
            `x`: a dataframe of trial documents.
            `fields`: the list of columns to be used in `x`.

            inputs = 

            {

            'x': pd.DataFrame,

            'fields': list[str],

            }

        Return
        ------
        embs: np.ndarray
            The encoded trial document embeddings.
        '''
        self._input_data_check(inputs)
        df = inputs['x']
        if 'fields' in inputs: fields = inputs['fields']
        else: fields = None
        df_raw = self._process_dataframe(df, fields)
        embs = self.model(df_raw['text'].tolist())
        return embs

    def predict(self, test_data, top_k=10):
        '''
        Take the input document, find the most similar documents in the training corpus.

        Parameters
        ----------
        test_data: dict
            Trial docs to be predicted. `x`: a dataframe of trial documents.
            `fields`: optional, the fields of documents to use for training. If not given,
            the model uses all fields in `x`.

            test_data =

            {

            'x': pd.DataFrame,

            'fields': list[str],

            }

        top_k: int, optional (default=10)
            The number of top similar documents to be retrieved.

        Return
        ------
        pred: list[list[tuple[str, float]]]
            The sequence of `('key': similarities)` for the input test documents for each input trial.
        '''
        self._input_data_check(test_data)

        # encode doc into embs
        embs = self.encode(test_data)

        # rank for each test document
        rank_list = []
        for emb in embs:
            ranks = self.model.model.dv.most_similar([emb], topn=top_k)
            rank_list.append(ranks)
        return rank_list

    def load_model(self, checkpoint):
        '''
        Load the pretrained model from disk.

        Parameters
        ----------
        checkpoint: str
            The path to the pretrained model.

            - If a directory, the only checkpoint file `*.pth.tar` will be loaded.
            - If a filepath, will load from this file.
        '''
        checkpoint_filename = check_checkpoint_file(checkpoint)
        config_filename = check_model_config_file(checkpoint)
        self.model = torch.load(checkpoint_filename)
        if config_filename is not None:
            config = self._load_model_config(config_filename)
            self.config.update(config)

    def save_model(self, output_dir):
        '''
        Save the trained model.

        Parameters
        ----------
        output_dir: str
            The output directory to save. Checkpoint is saved with name `checkpoint.pth.tar` by default.
        '''
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self._save_checkpoint(self.model, output_dir=output_dir)
        self._save_model_config(self.config, output_dir=output_dir)

    def __getitem__(self, tag):
        '''
        Get the embeddings of documents by the trial tags.

        Parameters
        ----------
        tag: str, int, list[str], list[int]
            The tag (or tags) to be looked up in the model.

        Returns
        -------
        emb: np.ndarry
            The embeddings of each document.
        '''
        return self.model.model[tag]

    def _build_model(self, train_data):
        # process raw pd.DataFrame inputs to the doc2vec tagged document inputs
        df = train_data['x']
        if 'fields' in train_data:
            fields = train_data['fields']
        else:
            fields = None
        df_raw = self._process_dataframe(df, fields)

        df_raw['text'] = df_raw['text'].apply(lambda x: gensim.utils.simple_preprocess(x))

        if 'tag' in train_data:
            tag = df[train_data['tag']]
            df_raw = pd.concat([tag, df_raw], axis=1)
            df_tagged_doc = df_raw.apply(lambda x: TaggedDocument(x.values[1], [x.values[0]]),axis=1)
        else:
            df_tagged_doc = df_raw.reset_index().apply(lambda x: TaggedDocument(x.values[1], [x.name]),axis=1)

        train_corpus = df_tagged_doc.tolist()

        config = {
            'emb_dim': self.emb_dim,
            'epochs': self.epochs,
            'window': self.window,
            'min_count': self.min_count,
            'max_vocab_size': self.max_vocab_size,
            'num_workers': self.num_workers,
        }
        self.model = BuildModel(train_corpus=train_corpus, **config)
        self.config = config
        self._save_model_config(self.config)
