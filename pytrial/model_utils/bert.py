'''
Provide an easy-to-access to all pretrained BERT-like models based on transformers@huggingface
cuz many models use pretrained bert embeddings.
'''
import pdb

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from tqdm.autonotebook import trange


class BERT(nn.Module):
    '''
    The pretrained BERT model for getting text embeddings.

    Parameters
    ----------
    bertname: str (default='emilyalsentzer/Bio_ClinicalBERT')
        The name of pretrained bert to get from huggingface models hub: https://huggingface.co/models.
        Or pass the dir where the local pretrained bert is available.

    proj_dim: int or None
        A linear projection head added on top of the bert encoder. Note that if given,
        the projection head is RANDOMLY initialized and needs further training.

    max_length: int
        Maximum acceptable number of tokens for each sentence.

    device: str
        The device of this model, typically be 'cpu' or 'cuda:0'.

    Examples
    --------
    >>> model = BERT()
    >>> emb = model.encode('The goal of life is comfort.')
    >>> print(emb.shape)
    '''
    is_train=None
    def __init__(self, bertname='emilyalsentzer/Bio_ClinicalBERT', proj_dim=None, max_length=512, device='cpu'):
        super().__init__()
        self.projection_head = None
        self.model = AutoModel.from_pretrained(bertname, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(bertname)
        self.tokenizer.__dict__['model_max_length'] = max_length
        if proj_dim is not None:
            self.projection_head = nn.Linear(768, proj_dim, bias=False)
            self.projection_head.to(device)
        self.device = device
        self.model.to(device)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, return_hidden_states=False):
        '''
        Forward pass of the model. 
        
        Parameters
        ----------
        input_ids: torch.Tensor
            The input token ids with shape [batch_size, seq_len].
        
        attention_mask: torch.Tensor
            The attention mask with shape [batch_size, seq_len].
        
        token_type_ids: torch.Tensor
            The token type ids with shape [batch_size, seq_len].
        
        return_hidden_states: bool
            Whether to return the hidden states of all layers.
        '''
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
        if not return_hidden_states:
            embed = output['pooler_output']
            if self.projection_head is not None:
                embed = self.projection_head(embed)
            return embed
        else:
            hidden_states = output['hidden_states'] # input embeds+12 layers, 13 embeds in total
            return hidden_states

    def encode(self, input_text, is_train=False, batch_size=None):
        '''
        Encode the input texts into embeddings.

        Parameters
        ----------
        input_text: str or list[str]
            A sentence or a list of sentences to be encoded.

        is_train: bool
            Set True if this model's parameters will update by learning.

        batch_size: int
            How large batch size to use when encoding long documents with many sentences.
            When set `None`, will encode all sentences at once.

        Returns
        -------
        outputs: torch.Tensor
            The encoded sentence embeddings with size [num_sent, emb_dim]
        '''
        self.is_train = is_train
        if batch_size is not None:
            # smart batching
            all_embeddings = []
            sentences_sorted, length_sorted_idx = self._smart_batching(input_text)
            for start_index in trange(0, len(input_text), batch_size, desc=f'BERT encoding total samples {len(input_text)}'):
                sentences_batch = sentences_sorted[start_index:start_index+batch_size]
                embs = self._encode_batch(sentences_batch)
                all_embeddings.extend(embs)
            
            all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
            return torch.stack(all_embeddings)
        else:
            return self._encode_batch(input_text)

    def _encode_batch(self, input_text):
        inputs = self.tokenizer(input_text, truncation=True, padding=True, return_tensors='pt')
        inputs = self._to_device(inputs)
        if not self.is_train:
            with torch.no_grad():
                outputs = self.forward(**inputs)
        else:
            outputs = self.forward(**inputs)
        return outputs
    
    def _smart_batching(self, input_text):
        length_sorted_idx = np.argsort([-len(sen) for sen in input_text])
        sentences_sorted = [input_text[idx] for idx in length_sorted_idx]
        return sentences_sorted, length_sorted_idx

    def _to_device(self, inputs):
        for k,v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)
        return inputs

if __name__ == '__main__':
    bert = BERT()
    emb = bert.encode(['asdadasdas', 'asdasdawqeqw'])
    print(emb.shape)
