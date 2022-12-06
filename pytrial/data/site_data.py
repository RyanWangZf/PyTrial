'''
Basic site data functions.
'''
import warnings
import dill
from collections import defaultdict
import torch

from torch.utils.data import Dataset, DataLoader
import numpy as np

from ..utils.tabular_utils import HyperTransformer

class SiteBase(Dataset):
    '''
    Base dataset class for trial site features. Subclass it if additional properties and functions
    are required to add for specific tasks. We make use `rdt`: https://docs.sdv.dev/rdt for transform
    and reverse transform of the tabular data.

    Parameters
    ----------
    df: pd.DataFrame
        The input site tabular format records.

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
    >>> df = pd.read_csv('tabular_site.csv', index_col=0)
    >>> # set `transform=True` will replace dataset.df with dataset.df_transformed
    >>> dataset = SiteBase(df, transform=True)
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
        return f'<pytrial.data.site_data.SiteBase object> Tabular site data with {self.df.shape[0]} samples {self.df.shape[1]} features, call `.df` to yield the pd.DataFrame data: \n' + repr(self.df)

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
        
class SiteBaseDemographics(Dataset):
    '''
    Base dataset class for trial site features with demographics added in. Subclass it if additional properties and functions
    are required to add for specific tasks. We make use `rdt`: https://docs.sdv.dev/rdt for transform
    and reverse transform of the tabular data.

    Parameters
    ----------
    df: pd.DataFrame
        The input site tabular format records. Must have column 'demographics' which is a list or ndarray of the racial distribution of the site

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
    >>> df = pd.read_csv('tabular_site.csv', index_col=0)
    >>> # set `transform=True` will replace dataset.df with dataset.df_transformed
    >>> dataset = TabularPatientBase(df, transform=True)
    >>> # transform raw dataframe to numerical tables
    >>> df_transformed = dataset.transform(df)
    >>> # make back transform to the original df
    >>> df_raw = dataset.reverse_transform(df_transformed)
    '''
    def __init__(self, df, metadata=None, transform=True):
        self.eth_labels = np.array(df.pop('demographics'))
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
        
    def get_label(self, index):
        '''
        Indexing the demographic labels

        Parameters
        ----------
        index: int or list[int]
            Retrieve the corresponding rows in the data.
        '''
        return self.eth_labels[index]

    def __len__(self):
        return len(self.df)

    def __repr__(self):
        return f'<pytrial.data.site_data.SiteBase object> Tabular site data with {self.df.shape[0]} samples {self.df.shape[1]} features, call `.df` to yield the pd.DataFrame data: \n' + repr(self.df)

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


class ModalitySiteBase(Dataset):
    '''
    Load sequential site inputs for longitudinal records generation.

    Parameters
    ----------
    data: dict
        A dict contains site data in sequence and/or in tabular.

        data = {

        'x': np.ndarray or pd.DataFrame
        
        Static site features in tabular form, typically those specialty information.

        'dx': list or np.ndarray
        
        Diagnosis history sequence in dense format or in tensor format (depends on the model input requirement.)
            
        - If in dense format, it is like [[c1,c2,c3],[c4,c5],...], with shape [n_patient, NA, NA];

        - If in tensor format, it is like [[0,1,1],[1,1,0],...] (multi-hot encoded), with shape [n_patient, max_num_visit, max_num_event].
        
        'rx': list or np.ndarray
        
        Prescription history sequence in dense format or in tensor format (depends on the model input requirement.)
            
        - If in dense format, it is like [[c1,c2,c3],[c4,c5],...], with shape [n_patient, NA, NA];

        - If in tensor format, it is like [[0,1,1],[1,1,0],...] (multi-hot encoded), with shape [n_patient, max_num_visit, max_num_event].
        
        'hist': list or np.ndarray
        
        Past trial enrollment sequence in dense format or in tensor format (depends on the model input requirement.)
            
        - If in dense format, it is like [(t1, e1),(t2, e2),...], with shape [n_past_trial, NA, NA];

        - If in tensor format, it is like [[0,1,1 ... e1],[1,1,0 ... e2],...] (multi-hot encoded), with shape [n_patient, max_num_trial, trial_dim + 1].
        
        'eth_label': np.ndarray

        - Distribution of racial demographics of patients in the site (array should sum to 1 or 100)
        
        }
    
    metadata: dict (optional)
        
        A dict contains configuration of input site data.

        metadata = {

        'dx_visit': dict[str]
        
        a dict contains the format of input data for processing input diagnosis visit sequences.

        `dx_visit`: {

        'mode': 'tensor' or 'dense',
        
        'voc': int,
        
        },
        
        'rx_visit': dict[str]
        
        a dict contains the format of input data for processing input prescription visit sequences.

        `rx_visit`: {

        'mode': 'tensor' or 'dense',
        
        'voc': int,
        
        },
        
        `hist`: {

        'mode': 'tensor' or 'dense',
        
        },
    
        'max_visit': int

        the maximum number of visits for the diagnosis and prescription history modalities considered when building tensor inputs, ignored
        when visit mode is dense.
        
        'max_trial': int

        the maximum number of past trials for the enrollment history modality considered when building tensor inputs, ignored
        when visit mode is dense.
        
        }

    '''
    feature = None
    dx_hist = None
    rx_hist = None
    enroll_hist = None
    eth_label = None
    max_visit = None
    max_trial = None
    visit_voc_size = None

    metadata = {
        'voc': {},
        
        'feature':{
            'mode': 'tensor',
            },
        
        'visit':{
            'mode': 'tensor',
            },
        
        'hist':{
            'mode': 'tensor',
            },

        'max_visit': 20,
        'max_trial': 10,
    }

    def __init__(self, data, metadata=None) -> None:
        # parse metadata
        self._parse_metadata(metadata=metadata)

        # get input data
        self._parse_inputdata(inputs=data)

    def __getitem__(self, index):
        return_data = {}
        if self.dx_hist is not None:
            dx_hist = self.dx_hist[index]
            if self.metadata['dx']['mode'] == 'tensor':
                dx_ts, dx_len = self._dense_visit_to_tensor(dx_hist, self.metadata['dx']['voc'])
                return_data['dx'] = dx_ts
                return_data['dx_lens'] = dx_len
            else:
                visit_dict, dx_len = self._tensor_visit_to_dense(dx_hist)
                return_data['dx'] = visit_dict
                return_data['dx_lens'] = dx_len
                
        if self.rx_hist is not None:
            rx_hist = self.rx_hist[index]
            if self.metadata['rx']['mode'] == 'tensor':
                rx_ts, rx_len = self._dense_visit_to_tensor(rx_hist, self.metadata['rx']['voc'])
                return_data['rx'] = rx_ts
                return_data['rx_lens'] = rx_len
            else:
                visit_dict, rx_len = self._tensor_visit_to_dense(rx_hist)
                return_data['rx'] = visit_dict
                return_data['rx_lens'] = rx_len
                
        if self.enroll_hist is not None:
            enroll_hist = self.enroll_hist[index]
            if self.metadata['hist']['mode'] == 'tensor':
                enroll_ts, enroll_len = self._dense_trial_to_tensor(enroll_hist) # return a dict with keys corresponding to order
                return_data['hist'] = enroll_ts
                return_data['hist_lens'] = enroll_len
            else:
                enroll_list, enroll_len = self._tensor_trial_to_dense(enroll_hist)
                return_data['hist'] = enroll_list
                return_data['hist_lens'] = enroll_len
                
        if self.feature is not None:
            return_data['x'] = self.feature[index]
            
        if self.eth_label is not None:
            return_data['eth_label'] = self.eth_label[index]
        
        return return_data

    def __len__(self):
        return len(self.visit)

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
        
        if 'max_visit' in metadata:
            self.max_visit = metadata['max_visit']
            
        if 'max_trial' in metadata:
            self.max_trial = metadata['max_trial']

    def _parse_inputdata(self, inputs):
        if 'x' in inputs: self.feature = inputs['x']
        if 'dx' in inputs: self.dx_hist = inputs['dx']
        if 'rx' in inputs: self.rx_hist = inputs['rx']
        if 'hist' in inputs: self.enroll_hist = inputs['hist']
        if 'eth_label' in inputs: self.eth_label = inputs['eth_label']

    def _dense_visit_to_tensor(self, visits, max_size):
        if not isinstance(visits, list):
            numVisits = self.max_visit
            for i in range(self.max_visit):
                if visits[i].sum() == 0:
                    numVisits = i
                    break
            return visits, numVisits
        
        res = np.zeros((self.max_visit, max_size), dtype=int)
        for i, visit in enumerate(visits):
            # clip if the max visit is larger than self.max_visit
            if i >= self.max_visit: break
            res[i, visit] = 1
        return res, min(len(visits), self.max_visit)
    
    def _tensor_visit_to_dense(self, visits):
        if isinstance(visits, list):
            return visits, len(visits)
        
        res = []
        for i in range(len(visits)):
            # clip if the max visit is larger than self.max_visit
            if i >= self.max_visit: break
            codes = np.nonzero(visits[i])[0].tolist()
            if codes == []: break
            res.append(codes)
        return res, min(len(res), self.max_visit)
    
    def _dense_trial_to_tensor(self, trials):
        if not isinstance(trials, list):
            numTrials = self.max_trial
            for i in range(self.max_visit):
                if trials[i].sum() == 0:
                    numTrials = i
                    break
            return trials, numTrials
        
        max_size = 1
        for i in range(len(trials)):
            max_size = trials[i][0]
            break
        
        res = np.zeros((self.max_trial, max_size), dtype=int)
        for i, trial in enumerate(trials):
            # clip if the max trial is larger than self.max_trial
            if i >= self.max_trial: break
            res[i, :-1] = trial[0]
            res[i, -1:] = trial[1]
        return res, min(len(trial), self.max_trial)
    
    def _tensor_trial_to_dense(self, trials):
        if isinstance(trials, list):
            return trials, len(trials)
        
        res = []
        for i in range(len(trials)):
            # clip if the max trial is larger than self.max_trial
            if i >= self.max_trial: break
            if trials[i].sum() == 0: break
            res.append((trials[i][:-1], trials[i][-1]))
        return res, min(len(res), self.max_visit)
    
    def get_label(self):
        return self.eth_label