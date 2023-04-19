'''
Basic patient data functions.
'''
import warnings
import pdb
import json
import pickle
import dill
from collections import defaultdict, OrderedDict
import torch

from torch.utils.data import Dataset, DataLoader
import numpy as np

from ..utils.tabular_utils import HyperTransformer
from ..utils.tabular_utils import get_transformer

class TabularPatientBase(Dataset):
    '''
    Base dataset class for tabular patient records. Subclass it if additional properties and functions
    are required to add for specific tasks. We make use `rdt`: https://docs.sdv.dev/rdt for transform
    and reverse transform of the tabular data.

    Parameters
    ----------
    df: pd.DataFrame
        The input patient tabular format records.

    metadata: dict
        Contains the meta setups of the input data. It should contain the following keys:

        (1) `sdtypes`: dict, the data types of each column in the input data. The keys are the column
            names and the values are the data types. The data types can be one of the following:
            'numerical', 'categorical', 'datetime', 'boolean'.

        (2) `transformers`: dict, the transformers to be used for each column. The keys are the column
            names and the values are the transformer names. The transformer names can be one in
            https://docs.sdv.dev/rdt/transformers-glossary/browse-transformers.
            In addition, we also support inputting a transformer string name, e.g., {'column1': 'OneHotEncoder'}.


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
    >>> from pytrial.data.patient_data import TabularPatientBase
    >>> df = pd.read_csv('tabular_patient.csv', index_col=0)
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
            
            # create a mapping from column name before to column name after transformation
            self._create_transformed_col2col()

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
        return f'<pytrial.data.patient_data.TabularPatientBase object> Tabular patient data with {self.df.shape[0]} samples {self.df.shape[1]} features, call `.df` to yield the pd.DataFrame data: \n' + repr(self.df)

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

    def _create_transformed_col2col(self):
        # create transformed column id to the original columns
        transformed_col2col = OrderedDict()
        for idx, col in enumerate(self.df.columns):
            col_transformer = self.metadata['transformers'][col]
            transformed_col2col[col] = col_transformer.output_columns
        self.metadata['transformed_col2col'] = transformed_col2col


class SequencePatientBase(Dataset):
    '''
    Load sequential patient inputs for longitudinal patient records generation.

    Parameters
    ----------
    data: dict
        A dict contains patient data in sequence and/or in tabular.

        data = {

        'x': np.ndarray or pd.DataFrame
        
        Static patient features in tabular form, typically those baseline information.

        'v': list or np.ndarray
        
        Patient visit sequence in dense format or in tensor format (depends on the model input requirement.)
            
        - If in dense format, it is like [[c1,c2,c3],[c4,c5],...], with shape [n_patient, NA, NA];

        - If in tensor format, it is like [[0,1,1],[1,1,0],...] (multi-hot encoded), with shape [n_patient, max_num_visit, max_num_event].
        
        'y': np.ndarray or pd.Series

        - Target label for each patient if making risk detection, with shape [n_patient, n_class];

        - Target label for each visit if making some visit-level prediction with shape [n_patient, NA, n_class].

        }
    
    metadata: dict (optional)
        
        A dict contains configuration of input patient data.

        metadata = {

        'voc': dict[Voc]

        Vocabulary contains the event index to the exact event name, has three keys in general:
        'diag', 'med', 'prod', corresponding to diagnosis, medication, and procedure.
        ``Voc`` object should have two functions: `idx2word` and `word2idx`.
        
        'visit': dict[str]
        
        a dict contains the format of input data for processing input visit sequences.

        `visit`: {

        'mode': 'tensor' or 'dense',
        
        'order': list[str] (required when `mode='tensor'`)
        
        },

        'label': dict[str]
        
        a dict contains the format of input data for processing input labels.

        `label`: {

        'mode': 'tensor' or 'dense',

        }
    
        'max_visit': int

        the maximum number of visits considered when building tensor inputs, ignored
        when visit mode is dense.
        
        }

    '''
    visit = None
    feature = None
    label = None
    max_visit = None
    visit_voc_size = None
    visit_order = None

    metadata = {
        'voc': {},
        
        'visit':{
            'mode': 'dense',
            'order': ['diag', 'prod', 'med'],
            },

        'label':{
            'mode': 'tensor',
            },

        'max_visit': 20,
    }

    def __init__(self, data, metadata=None) -> None:
        # parse metadata
        self._parse_metadata(metadata=metadata)

        # get input data
        self._parse_inputdata(inputs=data)

    def __getitem__(self, index):
        return_data = {}
        if self.visit is not None:
            visits = self.visit[index]
            if self.metadata['visit']['mode'] == 'tensor':
                visit_ts = self._dense_visit_to_tensor(visits) # return a dict with keys corresponding to order
                return_data['v'] = visit_ts
            else:
                visit_dict = self._parse_dense_visit_with_order(visits) # return a dict with keys corresponding to otder
                return_data['v'] = visit_dict
        
        if self.feature is not None:
            return_data['x'] = self.feature[index]
            
        if self.label is not None:
            return_data['y'] = self.label[index]
        
        return return_data

    def __len__(self):
        return len(self.visit)

    def _get_voc_size(self):
        order = self.metadata['visit']['order']
        vocs = self.metadata['voc']
        voc_size = []
        for order_ in order:
            voc_size.append(
                len(vocs[order_])
            )
        self.visit_voc_size = voc_size

    def _read_pickle(self, file_loc):
        return dill.load(open(file_loc, 'rb'))
    
    def _parse_metadata(self, metadata):
        if metadata is not None: 
            for k,v in metadata.items():
                if isinstance(v, dict):
                    self.metadata[k].update(v)
                else:
                    self.metadata[k] = v
        metadata = self.metadata

        if 'voc' in metadata:
            voc = metadata['voc']

            if 'diag' in voc: self.diag_voc = voc['diag']
            if 'prod' in voc: self.prod_voc = voc['prod']
            if 'med' in voc: self.med_voc = voc['med']
        
        if metadata['visit']['mode'] == 'tensor':
            self._get_voc_size()
        
        if 'order' in metadata['visit']:
            self.visit_order = metadata['visit']['order']

        if 'max_visit' in metadata:
            self.max_visit = metadata['max_visit']

    def _parse_inputdata(self, inputs):
        if 'x' in inputs: self.feature = inputs['x']
        if 'v' in inputs: self.visit = inputs['v']
        if 'y' in inputs: self.label = inputs['y']

    def _dense_visit_to_tensor(self, visits):
        res = {}
        for i,o in enumerate(self.visit_order):
            res[o] = np.zeros((self.max_visit, self.visit_voc_size[i]), dtype=int)

        for i, visit in enumerate(visits):
            # clip if the max visit is larger than self.max_visit
            if i >= self.max_visit: break
            
            for j, o in enumerate(self.visit_order):
                res[o][i, visit[j]] = 1
        return res
    
    def _parse_dense_visit_with_order(self, visits):
        return_data = defaultdict(list)
        order_list = self.metadata['visit']['order']
        for visit in visits:
            for i, o in enumerate(order_list):
                return_data[o].append(visit[i])
        return return_data
    
    def get_label(self):
        return self.label

class SeqPatientCollator:
    '''
    support make collation of unequal sized list of batch for densely stored
    sequential visits data.

    Parameters
    ----------
    config: dict
        
        {

        'visit_mode': in 'dense' or 'tensor',

        'label_mode': in 'dense' or 'tensor',

        }
        
    '''
    is_tensor_visit = False
    is_tensor_label = False
    config = {'visit_mode':'dense', 'label_mode':'tensor'} 

    def __init__(self, config=None):
        if config is not None:
            self.config.update(config)
            
        if self.config['visit_mode'] == 'tensor': self.is_tensor_visit = True
        else: self.is_tensor_visit = False

        if self.config['label_mode'] == 'tensor': self.is_tensor_label = True
        else: self.is_tensor_label = False

    def __call__(self, inputs):
        '''
        Paramters
        ---------
        inputs = {
            'v': {
                'diag': list[np.ndarray],
                'prod': list[np.ndarray],
                'med': list[np.ndarray],
                },

            'x': list[np.ndarray],

            'y': list[np.ndarray]
        }

        Returns
        -------
        outputs = {
            'v':{ # visit event sequence
                'diag': tensor or list[tensor],
                'prod': tensor or list[tensor],
                'med': tensor or list[tensor],
            },
            'x': tensor, # static features
            'y': tensor or list, # patient-level label in tensor or visit-level label in list[tensor]
        }
        '''
        # init output dict
        return_data = defaultdict(list)
        return_data['v'] = defaultdict(list)

        for input in inputs:
            for k, v in input.items():
                if k == 'v': # visit seq
                    for key, value in v.items():
                        return_data['v'][key].append(value)
                else: # feature and label
                    return_data[k].append(v)

        # processing all
        if self.is_tensor_visit:
            self._parse_tensor_visit(return_data)
        
        if self.is_tensor_label:
            self._parse_tensor_label(return_data)

        if 'x' in input:
            self._parse_tensor_feature(return_data)

        return return_data
    
    def _parse_tensor_visit(self, return_data):
        for k, v in return_data['v'].items():
            if isinstance(v, list):
                v = np.array(v)
            return_data['v'][k] = torch.tensor(v, dtype=int)

    def _parse_tensor_label(self, return_data):
        y_ts = torch.tensor(np.array(return_data['y']))
        return_data['y'] = y_ts

    def _parse_tensor_feature(self, return_data):
        x_ts = torch.tensor(np.array(return_data['x']))
        return_data['x'] = x_ts