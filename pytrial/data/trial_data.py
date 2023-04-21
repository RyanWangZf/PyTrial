import warnings
from collections import defaultdict
import pdb
import os
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from ..utils.trial_utils import ClinicalTrials
from ..utils.tabular_utils import read_csv_to_df
from ..data.vocab_data import Vocab

from ..utils.tabular_utils import HyperTransformer

class TrialDatasetBase(Dataset):
    '''
    The basic trial datasets loader.

    Parameters
    ----------
    data: pd.DataFrame
        Contain the trial document in tabular format.

    criteria_column: str
        The column name of eligibility criteria in the dataframe.
    '''
    inc_ec_embedding = None # inclusion criteria embedding
    inc_vocab = None # inclusion criteria vocab
    exc_ec_embedding = None # exclusion criteria embedding
    exc_vocab = None # exclusion criteria vocab

    def __init__(self, data, criteria_column='criteria'):
        self.df = data
        self._process_ec(criteria_column=criteria_column)
        self._collect_cleaned_sentence_set()
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return self.df.iloc[index:index+1]
    
    def get_ec_sentence_embedding(self):
        '''
        Process the eligibility criteria of each trial,
        get the criterion-level emebddings stored in dict.

        Parameters
        ----------
        criteria_column: str
            The column name of eligibility criteria in the dataframe.
        '''
        if self.inc_ec_embedding is None or self.exc_ec_embedding is None:
            self._get_ec_emb()
        
        return self.inc_ec_embedding, self.exc_ec_embedding

    def _process_ec(self, criteria_column):
        res = self.df[criteria_column].apply(lambda x: self._split_protocol(x))
        self.df['inclusion_criteria'] = res.apply(lambda x: x[0])
        self.df['exclusion_criteria'] = res.apply(lambda x: x[1])

    def _get_ec_emb(self):
        # create EC embedding with indexed ECs
        from pytrial.model_utils.bert import BERT
        # check if cuda is available
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        bert_model = BERT(device=device)
        self.inc_ec_embedding = bert_model.encode(self.inc_vocab.words, batch_size=64)
        self.exc_ec_embedding = bert_model.encode(self.exc_vocab.words, batch_size=64)
        self.inc_ec_embedding = self.inc_ec_embedding.cpu()
        self.exc_ec_embedding = self.exc_ec_embedding.cpu()

    def _collect_cleaned_sentence_set(self):
        # create a vocab for ec sentences
        self.inc_vocab = Vocab()
        self.exc_vocab = Vocab()
        self.inc_vocab.add_sentence(['[PAD]']) # 0 belongs to the pad token
        self.exc_vocab.add_sentence(['[PAD]']) # 0 belongs to the pad token

        inc_index_set, exc_index_set = [], []
        for idx, row in self.df.iterrows():
            row_inc_set, row_exc_set = [], []
            inc = row['inclusion_criteria']
            exc = row['exclusion_criteria']
            for sent in inc:
                self.inc_vocab.add_sentence(sent)
                row_inc_set.append(self.inc_vocab.word2idx[sent])
            for sent in exc:
                self.exc_vocab.add_sentence(sent)
                row_exc_set.append(self.exc_vocab.word2idx[sent])
            inc_index_set.append(list(set(row_inc_set)))
            exc_index_set.append(list(set(row_exc_set)))
        self.df['inclusion_criteria_index'] = inc_index_set
        self.df['exclusion_criteria_index'] = exc_index_set

    def _clean_protocol(self, protocol):
        protocol = protocol.lower()
        protocol_split = protocol.split('\n')
        filter_out_empty_fn = lambda x: len(x.strip())>0
        strip_fn = lambda x: x.strip()
        protocol_split = list(filter(filter_out_empty_fn, protocol_split))	
        protocol_split = list(map(strip_fn, protocol_split))
        return protocol_split

    def _split_protocol(self, protocol):
        protocol_split = self._clean_protocol(protocol)
        inclusion_idx, exclusion_idx = len(protocol_split), len(protocol_split)	
        for idx, sentence in enumerate(protocol_split):
            if "inclusion" in sentence:
                inclusion_idx = idx
                break
        for idx, sentence in enumerate(protocol_split):
            if "exclusion" in sentence:
                exclusion_idx = idx 
                break 		
        if inclusion_idx + 1 < exclusion_idx + 1 < len(protocol_split):
            inclusion_criteria = protocol_split[inclusion_idx:exclusion_idx]
            exclusion_criteria = protocol_split[exclusion_idx:]
            if not (len(inclusion_criteria) > 0 and len(exclusion_criteria) > 0):
                print(len(inclusion_criteria), len(exclusion_criteria), len(protocol_split))
                exit()
            return inclusion_criteria, exclusion_criteria ## list, list 
        else:
            return protocol_split, []


class TrialDataset(Dataset):
    '''
    Basic trial datasets loader.

    Parameters
    ----------
    input_dir: str
        The path to the trial dataset in tabular form (.csv).
        If a directory is given, the code will automatically pick the only '.csv' file under this dir.
    '''
    def __init__(self, input_dir=None) -> None:
        if os.path.isfile(input_dir):
            self.df = read_csv_to_df(input_dir, index_col=0)

        if os.path.isdir(input_dir):
            csv_names = [name for name in os.listdir(input_dir) if name.endswith('.csv')]
            if len(csv_names) > 1:
                raise Exception(f'`input_dir` {input_dir} is given where more than one csv files are found under this path.')
            if len(csv_names) == 0:
                raise Exception(f'`input_dir` {input_dir} is given where no csv file is found under this path.')
            self.df = read_csv_to_df(os.path.join(input_dir, csv_names[0]), index_col=0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return self.df.iloc[index:index+1]

class TrialDataCollator:
    '''The basic trial data collator.
    Subclass it and override the `__init__` & `__call__` function if need operations inside this step.

    Returns
    -------
    batch_df: pd.DataFrame
        A dataframe contains multiple fields for each trial.
    '''
    def __init__(self) -> None:
        # subclass to add tokenizer
        # subclass to add feature preprocessor
        pass

    def __call__(self, examples):
        batch_df = pd.concat(examples, 0)
        batch_df.fillna('none',inplace=True)
        return batch_df


class TrialOutcomeDatasetBase(TrialDatasetBase):
    '''
    Basic trial outcome datasets loader.

    Parameters
    ----------
    data: pd.DataFrame
        Contain the trial document in tabular format.
    '''
    columns = ['nctid', 'label', 'smiless',  'icdcodes', 'criteria']
    def __init__(self, data, columns=None) -> None:
        self.data = data
        if columns is not None:
            self.columns = columns
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        return row[self.columns[0]], row[self.columns[1]], row[self.columns[2]], row[self.columns[3]], row[self.columns[4]]


class TrialDatasetStructured(Dataset):
    '''
    Dataset class for structured trial features. Subclass it if additional properties and functions
    are required to add for specific tasks. We make use `rdt`: https://docs.sdv.dev/rdt for transform
    and reverse transform of the tabular data.

    Parameters
    ----------
    df: pd.DataFrame
        The input trial tabular format records.

    metadata: dict
        Contains the meta setups of the input data. It should contain the following keys:

        (1) `sdtypes`: dict, the data types of each column in the input data. The keys are the column
            names and the values are the data types. The data types can be one of the following:
            'numerical', 'categorical', 'datetime', 'boolean'.

        (2) `transformers`: dict, the transformers to be used for each column. The keys are the column
            names and the values are the transformer names. The transformer names can be one in
            https://docs.sdv.dev/rdt/transformers-glossary/browse-transformers.

        metadata = {

        'sdtypes': {

        'column1': 'numerical',
        'column2': 'boolean',
        'column3': 'datetime',
        'column4': 'categorical',
        'column5': 'categorical',

        },

        transformers':{

        'column1': rdt.transformers.FloatFormatter(missing_value_replacement='mean'),
        'column2': rdt.transformers.BinaryEncoder(missing_value_replacement='mode'),
        'column3': rdt.transformers.UnixTimestampEncoder(missing_value_replacement='mean'),
        'column4': rdt.transformers.FrequencyEncoder(),
        'column5': None, # will not do any thing to this column if no transformer is specified.
        
        }
        
        } 
        
        It is recommend to provide the metadata of the input tabular data.

        - If no metdata is given, the dataset will automatically detect the `dtypes` of columns and build the corresponding `transformers`.
        - If only `sdtypes` are given, the dataset will detect if there are missing `sdtypes` given and build the `transformers` and dtype automatically.

    transform: bool(default=True)
        Whether or not transform raw self.df by hypertransformer.
        If set False, :code:`self.df` will keep as the same as the passed one.

    Examples
    --------
    >>> from pytrial.data.site_data import TabularSiteBase
    >>> df = pd.read_csv('tabular_trial.csv', index_col=0)
    >>> # set `transform=True` will replace dataset.df with dataset.df_transformed
    >>> dataset = TabularPatientBase(df, transform=True)
    >>> # transform raw dataframe to numerical tables
    >>> df_transformed = dataset.transform(df)
    >>> # make back transform to the original df
    >>> df_raw = dataset.reverse_transform(df_transformed)
    '''
    def __init__(self, df, metadata=None, transform=True):
        self.df = df
        self.metadata = metadata

        # initialize hypertransformer
        self.ht = HyperTransformer()

        if transform:
            if metadata is None:
                warnings.warn('No metadata provided. Metadata will be automatically '
                            'detected from your data. This process may not be accurate. '
                            'We recommend writing metadata to ensure correct data handling.')
                self.ht.detect_initial_config(df)
                self.metadata = self.ht.get_config()
                self.ht.fit(df)

            else:
                # parse the metadata and update hypertransformer's config
                self._parse_metadata()

            # replace data with the transformed one
            self.df = self.transform(df)

    def __getitem__(self, index):
        # TODO: support better indexing
        '''
        Indexing the dataframe stored in tabular patient dataset.

        Parameters
        ----------
        index: int or list[int]
            Retrieve the corresponding rows in the dataset.
        '''
        if isinstance(index, int):
            return self.df.iloc[index:index+1]
        
        elif isinstance(index, list):
            return self.df.iloc[index]

    def __len__(self):
        return len(self.df)

    def __repr__(self):
        return f'<pytrial.data.trial_data.TrialDatasetStructured object> Tabular trial data with {self.df.shape[0]} samples {self.df.shape[1]} features, call `.df` to yield the pd.DataFrame data: \n' + repr(self.df)

    def transform(self, df=None):
        '''
        Transform the input df or the self.df by hypertransformer.
        If transform=True in `__init__`, then you do not need to call this function
        to transform self.df because it was tranformed already.

        Parameters
        ----------
        df: pd.DataFrame
            The dataframe to be transformed by self.ht
        '''

        if df is None:
            return self.ht.transform(self.df)
        else:
            return self.ht.transform(df)

    def reverse_transform(self, df=None):
        '''
        Reverse the input dataframe back to the original format. Return the self.df in the original
        format if `df=None`.

        Parameters
        ----------
        df: pd.DataFrame
            The dataframe to be transformed back to the original format by self.ht.
        '''
        if df is None:
            return self.ht.reverse_transform(self.df)
        else:
            return self.ht.reverse_transform(df)

    def _parse_metadata(self):
        '''
        Parse the passed metadata, cope with the following scnearios:
        (1) only `sdtypes` are given;
        (2) only `transformers` are given;
        (3) only partial `sdtypes` are given;
        (4) only partial `transformers` are given.
        '''
        # parse metadata dict for building the hypertransformer
        metadata = self.metadata
        self.ht.detect_initial_config(self.df, verbose=False)

        if 'transformers' in metadata:
            self.ht.update_transformers(metadata['transformers'])
        
        if 'sdtypes' in metadata:
            self.ht.update_sdtypes(metadata['sdtypes'])

        self.ht.fit(self.df)
        self.metadata.update(self.ht.get_config())

def test():
    trialdata = TrialDataset('./datasets/AACT-ClinicalTrial/')
    trial_collate_fn = TrialDataCollator()
    trialoader = DataLoader(trialdata, batch_size=10, shuffle=False, collate_fn=trial_collate_fn)
    batch = next(iter(trialoader))
    print(batch)

if __name__ == '__main__':
    test()
