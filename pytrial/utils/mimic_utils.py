'''
Some functions for mimic-iii data preprocessing.
Code is adapted from https://github.com/zzachw/PyHealth/blob/master/pyhealth/data/base_mimic.py.
'''

import numpy as np
import pandas as pd
import re
import os
import pdb

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

class MIMIC_Data:
    """The data template to store MIMIC data. Customized fields can be added
    in each parse_xxx methods.
    Parameters
        ----------
        patient_id
        time_duration
        selection_method
    """

    def __init__(self, patient_id, time_duration, selection_method):
        self.data = {}
        self.data['patient_id'] = str(patient_id)
        self.data['admission_list'] = []
        self.time_duration = time_duration
        self.selection_method = selection_method

    # TODO add more parse_* function on procedure and on medications

    def parse_patient(self, pd_series, mapping_dict=None):
        if mapping_dict is None:
            self.data['gender'] = pd_series['gender'].values[0]
            self.data['dob'] = pd_series['dob'].values[0]
        else:
            self.data['gender'] = pd_series[mapping_dict['gender']].values[0]
            self.data['dob'] = pd_series[mapping_dict['dob']].values[0]

    def parse_admission(self, pd_df):
        # TODO: implement the mapping dict
        for ind, row in pd_df.iterrows():
            # each admission is stored as a seperate dictionary and
            # added to admission_list
            admission_event = {}
            admission_event['admission_id'] = row['hadm_id']
            admission_event['admission_date'] = row['admittime']
            admission_event['discharge_date'] = row['dischtime']
            admission_event['death_indicator'] = int(~pd.isna(pd_df['deathtime']).values[0])

            # more elements can be added here by taking elements from row
            self.data['admission_list'].append(admission_event)

    def parse_diagnosis(self, pd_df):
        for i, admission_event in enumerate(self.data['admission_list']):
            temp_df = pd_df.loc[
                pd_df['hadm_id'] == admission_event['admission_id']]

            diag_code_list = temp_df['icd9_code'].tolist()
            admission_event['diagnosis'] = diag_code_list
            self.data['admission_list'][i] = admission_event

    def parse_prescription(self, pd_df, save_dir=''):
        if len(self.data['admission_list']) == 0:
            raise ValueError(
                "No admission information found. Parse admission info first.")

        # make saving directory if needed
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        for i, admission_event in enumerate(self.data['admission_list']):
            # get all the events
            temp_df = pd_df.loc[
                pd_df['hadm_id'] == admission_event['admission_id']]
            if not temp_df.empty:
                # save csv location
                # raw event
                admission_event['prescription_csv'] = str(
                    self.data['patient_id']) + '_' + str(
                    admission_event['admission_id']) + '_prescription' + '.csv'

                temp_df = temp_df.sort_values(by='startdate')

                temp_df.to_csv(
                    os.path.join(save_dir, admission_event['prescription_csv']),
                    index=False,
                )
            self.data['admission_list'][i] = admission_event

    def parse_procedure(self, pd_df):
        for i, admission_event in enumerate(self.data['admission_list']):
            temp_df = pd_df.loc[
                pd_df['hadm_id'] == admission_event['admission_id']]

            prod_code_list = temp_df['icd9_code'].tolist()
            admission_event['procedure'] = prod_code_list
            self.data['admission_list'][i] = admission_event

    def parse_icu(self, pd_df, mapping_dict=None):
        if len(self.data['admission_list']) == 0:
            raise ValueError(
                "No admission information found. Parse admission info first.")

        for i, admission_event in enumerate(self.data['admission_list']):
            # print(ind, self.data['patient_id'], admission_event['admission_id'])
            temp_df = pd_df.loc[
                pd_df['hadm_id'] == admission_event['admission_id']]
            # print(temp_df.shape)
            for ind, row in temp_df.iterrows():
                admission_event['icustay_id'] = row['icustay_id']
            self.data['admission_list'][i] = admission_event

    def parse_event(self, pd_df, save_dir='', event_mapping_df='',
                    var_list=None):

        if len(self.data['admission_list']) == 0:
            raise ValueError(
                "No admission information found. Parse admission info first.")

        # make saving directory if needed
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        for i, admission_event in enumerate(self.data['admission_list']):

            # get all the events
            temp_df = pd_df.loc[
                pd_df['hadm_id'] == admission_event['admission_id']]

            if not temp_df.empty:
                # save csv location
                # raw event
                admission_event['event_csv'] = str(
                    self.data['patient_id']) + '_' + str(
                    admission_event['admission_id']) + '_event' + '.csv'

                # sort it by the date and time
                temp_df = temp_df.sort_values(by='charttime')

                first_index = temp_df['charttime'].index[0]
                # print(temp_df['charttime'][first_index])
                temp_df['first_entry'] = pd.to_datetime(
                    temp_df['charttime'][first_index])

                # calculate the time difference in seconds since the first entry
                temp_df['secs_since_entry'] = (temp_df['charttime'] - temp_df[
                    'first_entry']).dt.total_seconds()
                # speed up by dropping date format
                temp_df['charttime'] = temp_df['charttime'].astype(str)
                temp_df['first_entry'] = temp_df['first_entry'].astype(str)

                # save the raw event data to local csv
                temp_df.to_csv(
                    os.path.join(save_dir, admission_event['event_csv']),
                    index=False)

                # generate episode data
                # handle all events data by aggregation during a time window
                episode_df = self.generate_episode(
                    temp_df, duration=self.time_duration,
                    event_mapping_df=event_mapping_df,
                    var_list=var_list)

                if episode_df.shape[0] >= 2:
                    # save episode location only if there are more than 2 sequencnes
                    # do not save if it is empty or just a single sequence
                    admission_event['episode_csv'] = str(
                        self.data['patient_id']) + '_' + str(
                        admission_event['admission_id']) + '_episode.csv'

                    # save the episode data to local csv
                    episode_df.to_csv(
                        os.path.join(save_dir, admission_event['episode_csv']),
                        index=False)

                # update the dictionary
                self.data['admission_list'][i] = admission_event

        return temp_df  # for debug purpose

    def write_record(self, temp_list, temp_df, var):
        # TODO: this may be sped up by aggreagte on itemid, and then use the last one
        if not temp_df.empty:
            # clean specific column
            if var == 'temperature':
                temp_df.loc[:, 'value'] = clean_temperature(temp_df).copy()
            elif var == 'diastolic blood pressure':
                temp_df.loc[:,'value'] = clean_dbp(temp_df).copy()
            elif var == 'systolic blood pressure':
                temp_df.loc[:,'value'] = clean_sbp(temp_df).copy()
            elif var == 'capillary refill rate':
                temp_df.loc[:,'value'] = clean_crr(temp_df).copy()
            elif var == 'weight':
                temp_df.loc[:,'value'] = clean_weight(temp_df).copy()
            elif var == 'height':
                temp_df.loc[:,'value'] = clean_height(temp_df).copy()
            elif var == 'fraction inspired oxygen':
                temp_df.loc[:,'value'] = clean_fio2(temp_df).copy()
            elif var == 'height':
                temp_df.loc[:,'glucose'] = clean_lab(temp_df).copy()
            elif var == 'ph':
                temp_df.loc[:,'value'] = clean_lab(temp_df).copy()
            elif var == 'oxygen saturation':
                temp_df.loc[:,'value'] = clean_o2sat(temp_df).copy()

            if self.selection_method == 'last':
                # take the last record within the range
                temp_list.append(temp_df.iloc[-1]['value'])
            elif self.selection_method == 'mean':
                temp_list.append(temp_df['value'].mean())
                # if there is no information, just skip
        else:
            temp_list.append('')

        return temp_list

    def generate_episode_headers(self, var_list):
        """Generate the header for episode file
        Parameters
        ----------
        var_list
        Returns
        -------
        """

        self.episode_headers_ = ['secs_since_entry']
        for var in var_list:
            # self.episode_headers_.extend([var, var + '_unit'])
            self.episode_headers_.append(var)

    def generate_episode(self, pd_df, duration, event_mapping_df, var_list):

        self.generate_episode_headers(var_list)

        # when it is in the memory, process it directly
        max_time_diff = pd_df['secs_since_entry'].max()

        # only keep the events we are interested in
        key_df = event_mapping_df[event_mapping_df['level2'].isin(var_list)]

        # rounding up
        n_episode = int(np.ceil(max_time_diff / duration))

        # print(n_episode)

        episode_df = pd.DataFrame(columns=self.episode_headers_)

        for j in range(n_episode):
            threshold_l = j * duration
            threshold_h = (j + 1) * duration

            # find all the events within the duration
            slice_df = pd_df.loc[(pd_df['secs_since_entry'] >= threshold_l) & (
                    pd_df['secs_since_entry'] < threshold_h)]

            # need some sort on the time so just get the last value within the range
            # can join on multiple things, weight for instance
            temp_df = key_df.merge(slice_df, left_on='itemid',
                                   right_on='itemid')

            # valid information is available, otherwise skip
            if not temp_df.empty:
                temp_df.sort_values(by='secs_since_entry', inplace=True)

                # initialize the record with timestamp
                temp_list = [threshold_h]

                # iterate over variables at interests
                for var in var_list:
                    var_df = temp_df[temp_df['level2'] == var]
                    # write records for each variable at interest
                    temp_list = self.write_record(temp_list, var_df, var)

                # append to the major episode dataframe
                epi_df = pd.DataFrame(temp_list).transpose()
                epi_df.columns = self.episode_headers_
                episode_df = pd.concat([episode_df, epi_df], axis=0)

                # ###############################################################
                # # need to iterate over rows, may not be sufficiently faster.
                # # on park
                # get the last record for each variable
                # episode_df = temp_df.groupby(['level2']).agg({
                #     'itemid' : 'last',
                #     'value' : 'last',
                #     'valueuom' : 'last',
                #     })
                # self.a = episode_df
                # print(episode_df)
                # ###############################################################

        # return the episode df
        return episode_df


##############################################################################

def parallel_parse_tables(patient_id_list, patient_df, admission_df, icu_df,
                          event_df, event_mapping_df,
                          prescription_df, procedure_df, diagnosis_df,
                          duration,
                          selection_method, var_list, save_dir):
    """Parallel methods to process patient information in batches

    Parameters
    ----------
    patient_id_list: list[int]
        A list of patient subject_id.

    patient_df: pd.DataFrame
        Patient table read from 'PATIENTS.csv'.

    admission_df: pd.DataFrame
        Admission table read from 'ADMISSIONS.csv'.

    icu_df: pd.DataFrame
        ICU stay table read from 'ICUSTAYS.csv'.

    event_df: pd.DataFrame
        Patient events read fom 'CHARTEVENTS.csv' and 'OUTPUTEVENTS.csv'.

    prescription_df: pd.DataFrame
        Prescription table read from 'PRESCRIPTIONS.csv'.

    procedure_df: pd.DataFrame
        Procedure table read from 'PROCEDURES_ICD.csv'.

    diagnosis_df: pd.DataFrame
        Diagnosis table read from 'DIAGNOSES_ICD.csv'.

    var_list: list[str]
        Selected event types for event_df.

    Returns
    -------
    """
    valid_data_list, valid_id_list = [], []

    # for i in tqdm(range(len(patient_id_list))):
    for i in tqdm(range(len(patient_id_list))):
        p_id = patient_id_list[i]
        # print('Processing Patient', i + 1, p_id)
        # initialize the
        temp_data = MIMIC_Data(p_id, duration, selection_method)
        p_df = patient_df.loc[patient_df['subject_id'] == p_id]
        a_df = admission_df.loc[admission_df['subject_id'] == p_id]
        i_df = icu_df.loc[icu_df['subject_id'] == p_id]
        e_df = event_df.loc[event_df['subject_id'] == p_id]
        drug_df = prescription_df.loc[prescription_df['subject_id']==p_id]
        diag_df = diagnosis_df.loc[diagnosis_df['subject_id']==p_id]
        prod_df = procedure_df.loc[procedure_df['subject_id']==p_id]

        if not p_df.empty:
            if p_df.shape[0] > 1:
                raise ValueError("Patient ID cannot be repeated")
            temp_data.parse_patient(p_df)

        if not a_df.empty:
            temp_data.parse_admission(a_df)

        if not i_df.empty:
            temp_data.parse_icu(i_df)

        if not e_df.empty:
            temp_data.parse_event(e_df, save_dir=save_dir,
                                  event_mapping_df=event_mapping_df,
                                  var_list=var_list)

        if not diag_df.empty:
            temp_data.parse_diagnosis(diag_df)

        if not drug_df.empty:
            temp_data.parse_prescription(drug_df, save_dir=save_dir)

        if not prod_df.empty:
            temp_data.parse_procedure(prod_df)


        '''
        `temp_data.data` is a dict
        dict{
        'patient_id':xxx,
        'admission_list': [
            {
                'admission_id':xxx, 'admission_data':xxxx,
                'event_csv': 'xxx.csv',
                'episode_csv':'xxxx.csv',
                'diagnosis': [icd1,icd2,...],
                'procedure': [icd1,icd2,...],
                'prescription_csv': 'xxx.csv',
                ....
            },
            ],
        'gender':'F',
        'dob': '2022-02-02 00:00:00',
        ...
        }
        '''

        valid_data_list.append(temp_data)
        valid_id_list.append(p_id)

    return valid_data_list, valid_id_list


# Temperature: map Farenheit to Celsius, some ambiguous 50<x<80
def clean_temperature(df):
    try:
        v = df.value.astype('float32').copy()
    except ValueError:
        print("could not convert string to float in clean_temperature. "
              "Set to NaN")
        v = df.value
        v[:] = np.nan
        return v
    else:
        idx = df.valueuom.fillna('').apply(lambda s: 'F' in s.lower()) | (
                v >= 79)
        v.loc[idx] = (v[idx] - 32) * 5. / 9
        return v

def clean_sbp(df):
    v = df.value.astype(str).copy()
    idx = v.apply(lambda s: '/' in s)
    v.loc[idx] = v[idx].apply(lambda s: re.match('^(\d+)/(\d+)$', s).group(1))
    return v.astype(float)


def clean_dbp(df):
    v = df.value.astype(str).copy()
    idx = v.apply(lambda s: '/' in s)
    v.loc[idx] = v[idx].apply(lambda s: re.match('^(\d+)/(\d+)$', s).group(2))
    return v.astype(float)


def clean_crr(df):
    v = pd.Series(np.zeros(df.shape[0]), index=df.index)
    v[:] = np.nan

    # when df.value is empty, dtype can be float and comparision with string
    # raises an exception, to fix this we change dtype to str
    df_value_str = df.value.astype(str)

    v.loc[(df_value_str == 'Normal <3 secs') | (df_value_str == 'Brisk')] = 0
    v.loc[
        (df_value_str == 'Abnormal >3 secs') | (df_value_str == 'Delayed')] = 1
    return v


# Weight: some really light/heavy adults: <50 lb, >450 lb, ambiguous oz/lb
# Children are tough for height, weight
def clean_weight(df):
    try:
        v = df.value.astype('float32').copy()
    except ValueError:
        print("could not convert string to float in clean_weight. Set to NaN")
        v = df.value
        v[:] = np.nan
        return v
    else:
        # ounces
        idx = df.valueuom.fillna('').apply(lambda s: 'oz' in s.lower())
        v.loc[idx] = v[idx] / 16.
        # pounds
        idx = idx | df.valueuom.fillna('').apply(lambda s: 'lb' in s.lower())
        v.loc[idx] = v[idx] * 0.453592
        return v


# Height: some really short/tall adults: <2 ft, >7 ft)
# Children are tough for height, weight
def clean_height(df):
    try:
        v = df.value.astype('float32').copy()
    except ValueError:
        print("could not convert string to float in clean_height. Set to NaN")
        v = df.value
        v[:] = np.nan
        return v
    else:
        idx = df.valueuom.fillna('').apply(lambda s: 'in' in s.lower())
        v.loc[idx] = np.round(v[idx] * 2.54)
        return v


# FIO2: many 0s, some 0<x<0.2 or 1<x<20
def clean_fio2(df):
    try:
        v = df.value.astype('float32').copy()
    except ValueError:
        print("could not convert string to float in clean_fio2. Set to NaN")
        v = df.value
        v[:] = np.nan
        return v
    else:
        ''' The line below is the correct way of doing the cleaning,
        since we will not compare 'str' to 'float'.
        If we use that line it will create mismatches from the data of the
        paper in ~50 ICU stays. The next releases of the benchmark should use this line.
        '''
        # idx = df.valueuom.fillna('').apply(lambda s: 'torr' not in s.lower()) & (v>1.0)

        ''' The line below was used to create the benchmark dataset that the
        paper used. Note this line will not work in python 3,
        since it may try to compare 'str' to 'float'.
        '''
        # idx = df.valueuom.fillna('').apply(lambda s: 'torr' not in s.lower()) & (df.value > 1.0)

        ''' The two following lines implement the code that was used to create
        the benchmark dataset that the paper used.
        This works with both python 2 and python 3.
        '''
        is_str = np.array(map(lambda x: type(x) == str, list(df.value)),
                          dtype=np.bool)
        idx = df.valueuom.fillna('').apply(
            lambda s: 'torr' not in s.lower()) & (
                      is_str | (~is_str & (v > 1.0)))

        v.loc[idx] = v[idx] / 100.
        return v


# GLUCOSE, PH: sometimes have ERROR as value
def clean_lab(df):
    v = df.value.copy()
    idx = v.apply(
        lambda s: type(s) is str and not re.match('^(\d+(\.\d*)?|\.\d+)$', s))
    v.loc[idx] = np.nan
    return v.astype(float)


# O2SAT: small number of 0<x<=1 that should be mapped to 0-100 scale
def clean_o2sat(df):
    # change "ERROR" to NaN
    v = df.value.copy()
    idx = v.apply(
        lambda s: type(s) is str and not re.match('^(\d+(\.\d*)?|\.\d+)$', s))
    v.loc[idx] = np.nan

    v = v.astype(float)
    idx = (v <= 1)
    v.loc[idx] = v[idx] * 100.
    return v

def convert_to_3digit_icd9(dxStr):
	if dxStr.startswith('E'):
		if len(dxStr) > 4: return dxStr[:4]
		else: return dxStr
	else:
		if len(dxStr) > 3: return dxStr[:3]
		else: return dxStr
