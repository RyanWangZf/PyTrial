'''
Implement dataset for trial search model training.
'''
import pdb
from collections import defaultdict

import pandas as pd
from transformers import AutoTokenizer
from torch import Tensor, device

from pytrial.data.trial_data import TrialDataCollator

class TrialSearchCollator(TrialDataCollator):
    '''
    The basic collator for trial search tasks.
    '''
    def __init__(self,
        bert_name,
        max_seq_length,
        fields,
        device,
        tag_field=None,
        ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(bert_name)
        self.max_length = max_seq_length
        self.fields = fields
        self.device = device
        self.tag_field = tag_field

    def __call__(self, features):
        return_dict = defaultdict(list)
        batch_df = pd.DataFrame(features)
        batch_df.fillna('', inplace=True)

        fields = self.fields
        return_dict.update(self._batch_tokenize(batch_df=batch_df, fields=fields))
        if self.tag_field is not None:
            return_dict[self.tag_field] = batch_df[self.tag_field].tolist()
        return return_dict

    def _batch_tokenize(self, batch_df, fields):
        return_dict = {}
        for field in fields:
            texts = batch_df[field].tolist()
            tokenized = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
            return_dict[field] = tokenized
        return return_dict

def batch_to_device(batch, target_device: device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            if 'cuda' in target_device:
                batch[key] = batch[key].cuda()
    return batch
