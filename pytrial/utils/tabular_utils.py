'''
Several utilities for processing tabular datasets.

TODO:

add LabelEncoder, OneHotEncoder, FrequencyEncoder (all encoders for categorical features) that supports

max_categories and min_frequencies, refer to https://scikit-learn.org/stable/modules/preprocessing.html.
'''
from collections import defaultdict
import pickle
from pathlib import Path
import pdb
import os
from copy import deepcopy

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler as sk_standardscaler
from sklearn.preprocessing import MinMaxScaler as sk_minmaxscaler
import rdt
from rdt.hyper_transformer import Config
from rdt.errors import InvalidConfigError
from rdt.transformers import BaseTransformer
from rdt.transformers import LabelEncoder, BinaryEncoder, UnixTimestampEncoder
from rdt.transformers.numerical import FloatFormatter
from rdt.transformers.categorical import OneHotEncoder, FrequencyEncoder

class StandardScaler(FloatFormatter):
    '''Transformer for numerical data.

    This transformer scales all numerical values within the same column using
    `sklearn.preprocessing.StandardScaler`.
    
    Null values are replaced using a `NullTransformer` from rdt.

    Parameters
    ----------
    missing_value_replacement: object or None
        Indicate what to do with the null values. If an integer or float is given,
        replace them with the given value. If the strings ``'mean'`` or ``'mode'`` are
        given, replace them with the corresponding aggregation. If ``None`` is given,
        do not replace them. Defaults to ``None``.

    enforce_min_max_values: bool (default=False)
        Whether or not to clip the data returned by ``reverse_transform`` to the min and
        max values seen during ``fit``. Defaults to ``False``.

    learn_rounding_scheme (bool):
        Whether or not to learn what place to round to based on the data seen during ``fit``.
        If ``True``, the data returned by ``reverse_transform`` will be rounded to that place.
        Defaults to ``False``.

    missing_value_generation (str or None):
        The way missing values are being handled. There are three strategies:

            * ``random``: Randomly generates missing values based on the percentage of
                missing values.
            * ``from_column``: Creates a binary column that describes whether the original
                value was missing. Then use it to recreate missing values.
            * ``None``: Do nothing with the missing values on the reverse transform. Simply
                pass whatever data we get through.

    computer_representation (dtype):
        Accepts ``'Int8'``, ``'Int16'``, ``'Int32'``, ``'Int64'``, ``'UInt8'``, ``'UInt16'``,
        ``'UInt32'``, ``'UInt64'``, ``'Float'``.
        Defaults to ``'Float'``.

    model_missing_values (bool):
        **Deprecated** Whether to create a new column to indicate which values were null or not. The column
        will be created only if there are null values. If ``True``, create the new column if
        there are null values. If ``False``, do not create the new column even if there
        are null values. Defaults to ``False``.
    '''
    _dtype = None
    _min_value = None
    _max_value = None
    random_states = None

    def __init__(self,
        missing_value_replacement=None,
        enforce_min_max_values=False,
        learn_rounding_scheme=False,
        computer_representation='Float',
        missing_value_generation="random",
        model_missing_values=None,
        ):
        self.output_properties = {None: {'sdtype': 'float', 'next_transformer': None}}
        super().__init__(
            missing_value_replacement=missing_value_replacement,
            model_missing_values=model_missing_values,
            learn_rounding_scheme=learn_rounding_scheme,
            enforce_min_max_values=enforce_min_max_values,
            computer_representation=computer_representation,
            missing_value_generation=missing_value_generation,
        )
        self.standard_transformer = sk_standardscaler()

    def _fit(self, data):
        '''
        Fit the transformer to the data.

        Parameters
        ----------
        data: pd.Series
            Data to fit.
        '''
        super()._fit(data)
        data = super()._transform(data)
        self.standard_transformer.fit(data[:,None])

    def _transform(self, data):
        '''
        Transform numerical data.

        Parameters
        ----------
        data: pd.Series
            Data to transform.

        Returns
        -------
            np.ndarray
        '''
        transformed = super()._transform(data)
        transformed = self.standard_transformer.transform(transformed[:,None])
        return transformed.flatten()

    def _reverse_transform(self, data):
        '''
        Convert the transformed data back to the original format.

        Parameters
        ----------
            data: pd.Series or np.ndarray
                Data to be reversely transformed.

        Returns
        -------
            np.ndarray
        '''
        if not isinstance(data, np.ndarray):
            data = data.to_numpy()

        if self.missing_value_replacement is not None:
            data = self.null_transformer.reverse_transform(data)

        if self.enforce_min_max_values:
            data = data.clip(self._min_value, self._max_value)

        if not isinstance(data, np.ndarray):
            data = data.to_numpy()

        data = self.standard_transformer.inverse_transform(data[:,None])

        is_integer = np.dtype(self._dtype).kind == 'i'
        if self.learn_rounding_scheme or is_integer:
            data = data.round(self._rounding_digits or 0)

        if pd.isna(data).any() and is_integer:
            return data

        return data.astype(self._dtype)

class MinMaxScaler(StandardScaler):
    '''Transformer for numerical data.

    This transformer scales all numerical values within the same column using
    `sklearn.preprocessing.MinMaxScaler`.

    Null values are replaced using a `NullTransformer` from rdt.

    Parameters
    ----------
    missing_value_replacement: object or None
        Indicate what to do with the null values. If an integer or float is given,
        replace them with the given value. If the strings ``'mean'`` or ``'mode'`` are
        given, replace them with the corresponding aggregation. If ``None`` is given,
        do not replace them. Defaults to ``None``.

    enforce_min_max_values: bool (default=False)
        Whether or not to clip the data returned by ``reverse_transform`` to the min and
        max values seen during ``fit``. Defaults to ``False``.

    learn_rounding_scheme (bool):
        Whether or not to learn what place to round to based on the data seen during ``fit``.
        If ``True``, the data returned by ``reverse_transform`` will be rounded to that place.
        Defaults to ``False``.

    model_missing_values (bool):
        Whether to create a new column to indicate which values were null or not. The column
        will be created only if there are null values. If ``True``, create the new column if
        there are null values. If ``False``, do not create the new column even if there
        are null values. Defaults to ``False``.
    '''

    _dtype = None
    _min_value = None
    _max_value = None

    def __init__(self,
        missing_value_replacement=None,
        enforce_min_max_values=False,
        learn_rounding_scheme=False,
        model_missing_values=None,
        computer_representation='Float',
        ):
        super().__init__(
            missing_value_replacement=missing_value_replacement,
            enforce_min_max_values=enforce_min_max_values,
            learn_rounding_scheme=learn_rounding_scheme,
            model_missing_values=model_missing_values,
            computer_representation=computer_representation)

        self.standard_transformer = sk_minmaxscaler()


class HyperTransformer(rdt.HyperTransformer):
    '''
    A subclass of `rdt.HyperTransformer` to set special setups.
    '''
    _DTYPES_TO_SDTYPES = {
        'i': 'categorical', # change the default from numerical to categorical for integers
        'f': 'numerical',
        'O': 'categorical',
        'b': 'boolean',
        'M': 'datetime',
    }
    def __init__(self):
        self._default_sdtype_transformers = {
            'numerical': StandardScaler(missing_value_replacement='mean'),
            'categorical': LabelEncoder(),
            'boolean': BinaryEncoder(missing_value_replacement='mode'),
            'datetime': UnixTimestampEncoder(missing_value_replacement='mean'),
        }
        self.field_sdtypes = {}
        self.field_transformers = {}
        self._specified_fields = set()
        self._validate_field_transformers()
        self._valid_output_sdtypes = self._DEFAULT_OUTPUT_SDTYPES
        self._multi_column_fields = self._create_multi_column_fields()
        self._transformers_sequence = []
        self._output_columns = []
        self._input_columns = []
        self._fitted_fields = set()
        self._fitted = False
        self._modified_config = False
        self._transformers_tree = defaultdict(dict)
        self.computer_representation = None

    def detect_initial_config(self, data, verbose=False):
        """Print the configuration of the data.
        This method detects the ``sdtype`` and transformer of each field in the data
        and then prints them as a json object.
        NOTE: This method completely resets the state of the ``HyperTransformer``.
        Args:
            data (pd.DataFrame):
                Data which will have its configuration detected.

            verbose (bool):
                Whether print user message or not.
        """
        # Reset the state of the HyperTransformer
        self.field_sdtypes = {}
        self.field_transformers = {}

        # Set the sdtypes and transformers of all fields to their defaults
        self._learn_config(data)

        if verbose:
            self._user_message('Detecting a new config from the data ... SUCCESS')
            self._user_message('Setting the new config ... SUCCESS')

        config = Config({
            'sdtypes': self.field_sdtypes,
            'transformers': self.field_transformers
        })

        if verbose:
            self._user_message('Config:')
            self._user_message(config)

    @staticmethod
    def _validate_transformers(column_name_to_transformer):
        """Validate the given transformers are valid.
        Args:
            column_name_to_transformer (dict):
                Dict mapping column names to transformers to be used for that column.
        Raises:
            Error:
                Raises an error if ``column_name_to_transformer`` contains one or more
                invalid transformers.
        """
        invalid_transformers_columns = []
        update_transformers = {}
        for column_name, transformer in column_name_to_transformer.items():
            if transformer and not isinstance(transformer, BaseTransformer):
                if isinstance(transformer, str):
                    if get_transformer(transformer) is None:
                        invalid_transformers_columns.append(column_name)
                    else:
                        # update the column_name_to_transformer dict with str inputs
                        update_transformers[column_name] = get_transformer(transformer)()

        if invalid_transformers_columns:
            raise InvalidConfigError(
                f'Invalid transformers for columns: {invalid_transformers_columns}. '
                'Please assign an rdt transformer instance to each column name.'
            )
        
        column_name_to_transformer.update(update_transformers)
    
    def _set_field_sdtype(self, data, field):
        clean_data = data[field].dropna()
        kind = clean_data.infer_objects().dtype.kind
        if kind == 'i':
            # decide if it is categorical or binary for input integers
            if len(clean_data.unique()) <= 2:
                kind = 'b'
            else:
                kind = 'O'
        self.field_sdtypes[field] = self._DTYPES_TO_SDTYPES[kind]


def read_csv_to_df(file_loc, header_lower=True, usecols=None, dtype=None,
                   low_memory=True, encoding=None, index_col=None):
    """Read in csv files with necessary processing
    Parameters
    ----------
    file_loc
    header_lower
    low_memory
    Returns
    -------
    """
    if dtype != None:
        df = pd.read_csv(file_loc, usecols=usecols, dtype=dtype,
                         low_memory=low_memory, encoding=encoding, index_col=index_col)
    else:
        df = pd.read_csv(file_loc, usecols=usecols, low_memory=low_memory,
                         encoding=encoding, index_col=index_col)

    if header_lower:
        df.columns = df.columns.str.lower()
    return df

def read_excel_to_df(file_loc, header_lower=True, usecols=None, dtype=None):
    """Read in excel files with necessary processing
    Parameters
    ----------
    file_loc
    header_lower
    low_memory
    Returns
    -------
    """
    if dtype != None:
        df = pd.read_excel(file_loc, usecols=usecols, dtype=dtype)
    else:
        df = pd.read_excel(file_loc, usecols=usecols)
    if header_lower:
        df.columns = df.columns.str.lower()
    return df

def read_txt_to_df(file_loc, header_lower=True, usecols=None, dtype=None,
                     low_memory=True, encoding=None):
    """Read in excel files with necessary processing

    Parameters
    ----------
    file_loc
    header_lower
    low_memory

    """
    if dtype != None:
        df = pd.read_table(file_loc, usecols=usecols, dtype=dtype,
                           low_memory=low_memory, encoding=encoding)
    else:
        df = pd.read_table(file_loc, usecols=usecols, low_memory=low_memory,
                           encoding=encoding)

    if header_lower:
        df.columns = df.columns.str.lower()
    return df

def read_text_to_list(file_loc, encoding='utf-8'):
    '''
    Read raw text files into a list one line by one line
    '''
    with open(file_loc, 'r', encoding=encoding) as f:
        x = [l.strip() for l in f.readlines()]
    return x


def load_table_config(data_dir, discriminate_bin_feat=False):
    '''
    Load the categorical, numerical, binary feature configuration from the local dataset folder.

    Parameters
    ----------
    data_dir: str
        The target dataset folder.

    discriminate_bin_feat: bool
        Whether discriminate binary and categorical feature or not.

    '''
    num_filepath = os.path.join(data_dir, 'numerical_feature.txt')
    if os.path.exists(num_filepath):
        num_feat_list = read_text_to_list(num_filepath)
    else:
        num_feat_list = []
    
    bin_filepath = os.path.join(data_dir, 'binary_feature.txt')
    if os.path.exists(bin_filepath):
        bin_feat_list = read_text_to_list(bin_filepath)
    else:
        bin_feat_list = []
    bin_feat_list = [c.strip().lower() for c in bin_feat_list]

    # get all column names
    data_filepath = os.path.join(data_dir, 'data_processed.csv')
    df = read_csv_to_df(data_filepath, index_col=0)

    all_feat_list = [c.strip().lower() for c in df.columns.tolist()]
    cat_feat_list = [c for c in all_feat_list if c not in num_feat_list and c not in bin_feat_list and 'target_label' not in c]
    num_feat_list = [c.strip().lower() for c in num_feat_list]

    if not discriminate_bin_feat:
        # merge bin and cat
        cat_feat_list.extend(bin_feat_list)

    # get cat cardinalities
    cat_cardinalities = []
    for c in cat_feat_list:
        cat_cardinalities.append(df[c].nunique())

    return {
        'columns': all_feat_list,
        'num_feat': num_feat_list,
        'bin_feat': bin_feat_list,
        'cat_feat': cat_feat_list,
        'cat_cardinalities': cat_cardinalities,
    }

def get_transformer(name):
    mapping = {
        'labelencoder':LabelEncoder,
        'onehotencoder':OneHotEncoder,
        'binaryencoder':BinaryEncoder,
        'floatformatter':FloatFormatter,
        'frequencyencoder':FrequencyEncoder,
        'unixtimeformatter':UnixTimestampEncoder,
    }
    if isinstance(name, str):
        return mapping.get(name.lower(), None)
    else:
        raise ValueError('The input transformer name must be a string. Get {} instead.'.format(type(name)))