import pdb
from collections import defaultdict

from pytrial.data.patient_data import SequencePatientBase
from pytrial.data.trial_data import TrialDatasetBase
from pytrial.data.patient_data import SeqPatientCollator

class PatientData(SequencePatientBase):
    '''
    Load sequence patient EHR data for patient-trial matching.
    '''
    def __init__(self, data, metadata=None) -> None:
        super().__init__(data, metadata)

class TrialData(TrialDatasetBase):
    '''
    Load trial dataset with eligibility criteria embedding.
    Each corresponds to a set matched patient indices for training
    patient-trial matching models.

    Parameters
    ----------
    data: pd.DataFrame
        A dataframe contains trial information including a column named `criteria` and 
        a column named `label` which contains a set of patient indices as matched.
    '''
    def __init__(self, data, encode_ec=False):
        super().__init__(data)
        # get criteria-level embeddings
        # stored in self.criteria_embedding as a matrix
        # self.df['inclusion_criteria_index'] and self.df['exclusion_criteria_index'] have the corresponding 
        # criteria indices for each trial.
        if encode_ec:
            self.get_ec_sentence_embedding()

    def __getitem__(self, index):
        # only get EC index and embeddings
        row = self.df.iloc[index]
        inc_ec_index = row['inclusion_criteria_index']
        if len(inc_ec_index) == 0: inc_ec_index.append(0)
        exc_ec_index = row['exclusion_criteria_index']
        if len(exc_ec_index) == 0: exc_ec_index.append(0) # pad
        output = {'inc_ec_index':inc_ec_index, 'exc_ec_index':exc_ec_index}
        
        inc_ec_emb = self.inc_ec_embedding[inc_ec_index]
        exc_ec_emb = self.exc_ec_embedding[exc_ec_index]
        nct_id = row['nctid']

        output.update(
            {
            'inc_ec_emb':inc_ec_emb,
            'exc_ec_emb':exc_ec_emb,
            'nct_id':nct_id,
            }
        )

        return output

class TrialCollator:
    '''
    Support the collation of trial records for patient-trial matching.
    '''
    def __call__(self, inputs):
        output = defaultdict(list)
        for x in inputs:
            output['inc_ec_index'].append(x['inc_ec_index'])
            output['exc_ec_index'].append(x['exc_ec_index'])
            output['inc_ec_emb'].append(x['inc_ec_emb'])
            output['exc_ec_emb'].append(x['exc_ec_emb'])
            output['nct_id'].append(x['nct_id'])
        return output

class PatientCollator(SeqPatientCollator):
    '''
    Support the collation of sequential patient EHR records for patient-trial matching.
    '''
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