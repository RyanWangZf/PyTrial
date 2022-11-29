import pdb
import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import nn
from datasets import Dataset
from tqdm import tqdm

from pytrial.model_utils.bert import BERT
from pytrial.utils.check import make_dir_if_not_exist
from .base import TrialSearchBase
from .base import whitening_torch_final
from ..metrics import precision, recall, ndcg
from ..data import TrialSearchCollator, batch_to_device


class BuildModel:
    def __new__(self, bert_name):
        return BERT(bert_name)    

class WhitenBERT(TrialSearchBase):
    '''
    Implement a postprocessing method to improve BERT embeddings for similarity search [1]_.

    Parameters
    ----------
    layer_mode: {'last_first', 'last''}
        The mode of layer of embeddings to aggregate. 'last_first' means use the last layer and the first layer. 'last' means use the last layer only.

    bert_name: str, optional (default = 'emilyalsentzer/Bio_ClinicalBERT')
        The name of base BERT model used for encoding input texts.

    device: str, optional (default = 'cuda:0')
        The device of this model, typically be 'cpu' or 'cuda:0'.

    experiment_id: str, optional (default = 'test')
        The name of current experiment.
    
    Notes
    -----
    .. [1] Huang, J., Tang, D., Zhong, W., Lu, S., Shou, L., Gong, M., ... & Duan, N. (2021, November). WhiteningBERT: An Easy Unsupervised Sentence Embedding Approach. In Findings of the Association for Computational Linguistics: EMNLP 2021 (pp. 238-244).
    '''
    trial_embs = {}
    def __init__(self,
        layer_mode='last_first',
        bert_name='emilyalsentzer/Bio_ClinicalBERT',
        device='cuda:0',
        experiment_id='test'):
        super().__init__(experiment_id)
        self.layer_mode = layer_mode
        self.bert_name = bert_name
        self.device = device
        self._build_model()
        self.config = {
            'layer_model': self.layer_mode,
            'bert_name': self.bert_name,
            'device': self.device,
            'batch_size': 32, # default batch size for encoding
            'num_workers': 4, # default num_workers for encoding
            'max_seq_length': 512, # default max_seq_length for encoding
            'tag_field': 'nctid', # default tag_field for encoding
        }

    def __getitem__(self, tag):
        return self.embs[tag]

    def fit(self, train_data, valid_data=None):
        '''
        Go over all trials and encode them into embeddings.
        Note that this is a post-processing method based on a pretrained BERT model,
        so it does `NOT` need to be trained.

        Parameters
        ----------
        train_data: dict
            The data for encoding.

            - 'x' is the dataframe that contains multiple sections of a trial.
            - 'fields' is the list of fields to be encoded.
            - 'tag' is the unique index column name of each document, e.g., 'nctid'.

            train_data = 
            
            {
            
            'x': pd.DataFrame,

            'fields': list[str],

            'tag': str,

            }

        valid_data: Not used.
            This is a placeholder because this model does not need training.
        '''
        embs = self.encode(train_data, return_dict=True)
        self.trial_embs = embs
        return embs

    def predict(self,
        test_data,
        top_k=10,
        return_df=True):
        '''
        Predict the top-k relevant for input documents.

        Parameters
        ----------
        test_data: dict
            Share the same input format as the `train_data` in `fit` function.
            If `fields` and `tag` are not given, will reuse the ones used during training.

            test_data = 

            {
            
            'x': pd.DataFrame,

            'fields': list[str],

            'tag': str,
            
            }

        top_k: int
            Number of retrieved candidates.

        return_df: float
            - If set True, return dataframe for the computed similarity ranking.
            - else, return rank_list=[[(doc1,sim1),(doc2,sim2)], [(doc1,sim1),...]].

        Returns
        -------
        rank: pd.DataFrame
            A dataframe contains the top ranked NCT ids for each.

        sim: pd.DataFrame
            A dataframe contains the corresponding similarities.

        rank_list: list[list[tuple]]
            A list of tuples of top ranked docs and similarities.
        '''
        self._input_data_check(test_data)
        
        tag_field = self.config['tag_field']

        assert len(self.trial_embs) > 0, 'No embeddings found. Please run `fit` first. Or load embeddings from `load_model`.'

        all_embs = np.stack(self.trial_embs.values())
        all_tags = np.stack(self.trial_embs.keys())

        # TODO: skip encoding those already stored in model
        # to_encode_test_trial = test_trial[~test_trial[tag_field].isin(all_tags)]
        tags, embs = self.encode(test_data, return_dict=False)
        
        # rank for each test document
        sim = embs.dot(all_embs.T)
        rank = np.argsort(sim, 1)[:,::-1] # flip
        rank = rank[:,1:top_k+1]

        if return_df:
            tag_list, sim_list = [], []
            for i,row in enumerate(rank):
                tag_list.append(all_tags[row])
                sim_list.append(sim[i][row])
            tag_ar = np.array(tag_list)
            sim_ar = np.array(sim_list)
            rank_df = pd.DataFrame(tag_ar, columns=[f'rank_{i}' for i in range(top_k)], index=tags)
            sim_df = pd.DataFrame(sim_ar, columns=[f'rank_{i}' for i in range(top_k)], index=tags)
            return rank_df, sim_df
        else:
            rank_list = []
            for i,row in enumerate(rank):
                tag_ar = all_tags[row]
                sim_ar = sim[i][row]
                rank_list.append(
                    list(zip(tag_ar, sim_ar))
                )
            return rank_list

    def evaluate(self, test_data):
        '''
        Evaluate within the given trial and corresponding candidate trials.

        Parameters
        ----------
        test_data:  dict
            The provided labeled dataset for test trials. Follow the format listed below.

            test_data =

            {

            'x': pd.DataFrame,

            'y': pd.DataFrame

            }


        Returns
        -------
        results: dict[float]
            A dict of metrics and the values.

        Notes
        -----
        x =
        
        | target_trial | trial1 | trial2 | trial3 |


        | nct01        | nct02  | nct03  | nct04  |

        y =

        | label1 | label2 | label3 |


        | 0      | 0      | 1      |

        '''
        test_df = test_data['x']
        label_df = test_data['y']
        ranked_label_list = []
        for idx, row in test_df.iterrows():
            target_trial = row['target_trial']
            if target_trial in self.trial_embs:
                target_emb = self.__getitem__(target_trial)
            else:
                raise ValueError(f'The embeddings of trial {target_trial} are not found.')
            candidate_embs = np.stack([self.__getitem__(tag) for tag in row.values[1:]])
            sim = target_emb[None].dot(candidate_embs.T)[0]
            labels = label_df.iloc[idx].to_numpy()
            if labels.sum() == 0: continue
            ranked_label = labels[np.argsort(sim)[::-1]]
            ranked_label_list.append(ranked_label)
        ranked_label_list = np.array(ranked_label_list)

        return_dict = {}
        for k in [1,2,5]:
            return_dict[f'precision@{k}'] = precision(ranked_label_list, k)
            return_dict[f'recall@{k}'] = recall(ranked_label_list, k)
        return_dict[f'ndcg@{k}'] = ndcg(ranked_label_list, k)
        return return_dict

    def save_model(self, output_dir):
        '''
        Only save the embeddings. Do not save the model.
        
        Parameters
        ----------
        output_dir: str
            The output directory to save the model.
        '''
        if not os.path.exists(output_dir):
            make_dir_if_not_exist(output_dir)
        with open(os.path.join(output_dir, 'embeddings.pkl'), 'wb') as f:
            pickle.dump(self.trial_embs, f)
        print(f'Embeddings saved to {output_dir}')

    def load_model(self, input_dir):
        '''
        Only load the embeddings. Do not load the model.
        
        Parameters
        ----------
        input_dir: str
            The input directory to load the model.
        '''
        with open(os.path.join(input_dir, 'embeddings.pkl'), 'rb') as f:
            self.trial_embs = pickle.load(f)
        print(f'Embeddings loaded from {input_dir}')

    def encode(self,
        inputs,
        batch_size=None,
        num_workers=None,
        return_dict=True,
        verbose=True,
        ):
        '''
        Encode input documents and output the document embeddings.

        Parameters
        ----------
        inputs: dict
            The input documents to encode:

            - 'fields' is the list of fields to be encoded.

            - 'tag' is the unique index column name of each document, e.g., 'nctid'.

            inputs = 

            {
            
            'x': pd.DataFrame,

            'fields': list[str],

            'tag': str,

            }

        batch_size: int, optional
            The batch size when encoding trials.
        
        num_workers: int, optional
            The number of workers when building the val dataloader.

        return_dict: bool
            Whether to return a dict of results.

            - If set True, return dict[np.ndarray].
            - Else, return np.ndarray with the order same as the input documents.

        verbose: bool
            Whether plot progress bar or not.

        Returns
        -------
        embs: dict[np.ndarray]
            Encoded trial-level embeddings with key (tag) and value (embedding)..
        '''
        self._input_data_check(inputs)
        if batch_size is not None:
            self.config['batch_size'] = batch_size
        
        if num_workers is not None:
            self.config['num_worker'] = num_workers

        # build dataset and dataloader
        dataloader = self.get_val_dataloader(inputs)

        # encode all trials
        embs = self._encode_for_dataloader(dataloader, verbose=verbose, return_dict=return_dict)
        return embs

    def get_val_dataloader(self, inputs):
        df = inputs['x']
        self.config['fields'] = inputs['fields']
        self.config['tag_field'] = inputs['tag']
        dataset = self._build_val_dataset(df=df)
        collate_fn = self._build_val_collator()
        dataloader = self._build_val_dataloader(dataset=dataset, collate_fn=collate_fn)
        return dataloader

    def _build_model(self):
        self.model = BuildModel(self.bert_name)
        self.model.to(self.device)

    def _encode_for_dataloader(self, dataloader, verbose=True, return_dict=True):
        emb_list = []
        tag_list = []
        self.model.eval()
        with torch.no_grad():
            for data in tqdm(dataloader, desc='Encoding', disable=not verbose):
                if self.config['tag_field'] is not None:
                    tag = data.pop(self.config['tag_field'])
                    tag_list.append(tag)

                data = self._prepare_input(data)

                # get an average of embeddings of all fields
                embs = []
                for field in self.config['fields']:
                    emb = self._encode_for_field(data=data, field=field)
                    embs.append(emb)
                temp_emb = torch.stack(embs, dim=0).mean(dim=0)
                if len(temp_emb.shape) == 1: temp_emb = temp_emb[None]

                # whitening for each batch
                temp_emb = whitening_torch_final(temp_emb)
                emb_list.append(temp_emb.cpu().numpy())

        embs = np.concatenate(emb_list, 0)

        if len(tag_list) == 0:
            tag_list = list(range(len(embs)))
        else:
            tag_list = sum(tag_list, [])

        if return_dict:
            embs = dict(zip(tag_list, embs))
            return embs
        else:
            return (tag_list, embs)

    def _encode_for_field(self, data, field):
        inputs = {}
        for k, v in data.items():
            if field in k:
                tempkey = k.replace(field+'_','')
                inputs[tempkey] = v
        inputs = batch_to_device(inputs, self.device)
        if self.layer_mode == 'last_first':
            # take a mean of the last and first layer
            inputs['return_hidden_states'] = True
            outputs = self.model(**inputs)
            emb = (outputs[1].mean(1) + outputs[-1].mean(1))/2
        else:
            # only take the last layer embedding
            emb = self.model(**inputs)
        return emb

    def _prepare_input(self, inputs):
        fields = self.config['fields']
        output_dict = {}
        for fd in fields:
            if fd not in inputs:
                continue
            for k,v in inputs[fd].items():
                output_dict[f'{fd}_{k}'] = v
        return output_dict

    def _build_val_dataset(self, df):
        dataset = Dataset.from_pandas(df)
        dataset.set_transform(self._val_dataset_transform)
        return dataset

    def _build_val_collator(self):
        return TrialSearchCollator(
            bert_name=self.config['bert_name'],
            max_seq_length=self.config['max_seq_length'],
            fields=self.config['fields'],
            device=self.config['device'],
            tag_field=self.config['tag_field'],
        )

    def _build_val_dataloader(self, dataset, collate_fn):
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            collate_fn=collate_fn,
            pin_memory=True,
        )
        return dataloader

    def _dataset_transform(self, examples):
        return_dict = {}

        if self.config['fields'] is not None:
            for k in self.config['fields']: 
                return_dict[k] = examples[k]
        
        if self.config['tag_field'] is not None:
            return_dict[self.config['tag_field'] ] = examples[self.config['tag_field'] ]

        if len(return_dict) == 0:
            return_dict = examples

        return return_dict

    def _val_dataset_transform(self, examples):
        return self._dataset_transform(examples=examples)

    def _input_data_check(self, inputs):
        '''
        Check the training / testing data fits the formats.
        Target to (1) check if inputs valid,
                    if not, give tips about the data problem.

        Parameters
        ----------
        inputs: {
                'x': pd.DataFrame,
                'fields': list[str],
                'tag': str,
                }
        '''
        # check overall input format
        assert 'x' in inputs, 'No input trial doc dataframe found in inputs.'
        df = inputs['x']
        if 'fields' in inputs:
            try:
                _ = df[inputs['fields']]
            except:
                raise Exception('Cannot find the specified `fields` in inputs dataframe.')
        if 'tag' in inputs:
            try:
                _ = df[inputs['tag']]
            except:
                raise Exception('Cannot find the specified `tag` in inputs dataframe.')

        # check data type
        try:
            _ = df.applymap(str)
        except:
            raise Exception('Cannot transform the input dataframe to str type, please check the inputs.')