'''Implement the data class for trial outcome task'''

import os
import pdb
from pytrial.data.trial_data import TrialOutcomeDatasetBase

class TrialOutcomeDataset(TrialOutcomeDatasetBase):

    def __init__(self, data):
        self.df = data
        self.nct2inc_emb = None
        self.nct2exc_emb = None

    def get_ec_sentence_embedding(self, criteria_column='criteria'):
        '''
        Process the eligibility criteria of each trial,
        get the criterion-level emebddings stored in dict.

        Parameters
        ----------
        criteria_column: str
            The column name of eligibility criteria in the dataframe.
        '''
        if self.inc_ec_embedding is None or self.exc_ec_embedding is None:
            self.criteria_column = criteria_column
            self._process_ec(criteria_column=criteria_column)
            self._collect_cleaned_sentence_set()
            # get the embedding of each criterion
            self._get_ec_emb()

        return {
            "inc_ec_emb": self.inc_ec_embedding, 
            "exc_ec_emb": self.exc_ec_embedding,
            "inc_ec": self.inc_vocab,
            "exc_ec": self.exc_vocab,
            }

    def get_nct_to_ec_emb(self):
        '''
        Get the ec embedding on trial-level, i.e., the average of criterion embeddings.
        Only used when the criterion-level embedding is already obtained by `self.get_ec_sentence_embedding`.
        '''
        if self.inc_ec_embedding is None or self.exc_ec_embedding is None:
            raise ValueError("""The criterion-level embedding is not obtained yet. You should call `self.get_ec_sentence_embedding` first.""")

        if self.nct2inc_emb is None or self.nct2exc_emb is None:
            self.nct2exc_emb = {}
            self.nct2inc_emb = {}
            for idx, row in self.df.iterrows():
                if len(row['inclusion_criteria_index']) > 0:
                    self.nct2inc_emb[row['nctid']] = self.inc_ec_embedding[row['inclusion_criteria_index']].mean(axis=0)
                
                if len(row['exclusion_criteria_index']) > 0:
                    self.nct2exc_emb[row['nctid']] = self.exc_ec_embedding[row['exclusion_criteria_index']].mean(axis=0)

        return {
            "nct2incemb": self.nct2inc_emb,
            "nct2excemb": self.nct2exc_emb,
            }