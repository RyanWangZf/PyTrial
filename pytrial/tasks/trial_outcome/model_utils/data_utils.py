import pdb
import os
import itertools

import numpy as np
import torch
import random

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.random.set_rng_state(torch.manual_seed(seed).get_state())

def admet_collate_fn(x):
    smiles_lst = [i[0] for i in x]
    label_vec = default_collate([int(i[1]) for i in x])  ### shape n, 
    return [smiles_lst, label_vec]

def smiles_txt_to_lst(text):
    """
        "['CN[C@H]1CC[C@@H](C2=CC(Cl)=C(Cl)C=C2)C2=CC=CC=C12', 'CNCCC=C1C2=CC=CC=C2CCC2=CC=CC=C12']" 
    """
    text = text[1:-1]
    lst = [i.strip()[1:-1] for i in text.split(',')]
    return lst

def icdcode_text_2_lst_of_lst(text):
    text = text[2:-2]
    lst_lst = []
    for i in text.split('", "'):
        i = i[1:-1]
        lst_lst.append([j.strip()[1:-1] for j in i.split(',')])
    return lst_lst

def list_text_2_list(text):
    text = text[2:-2]
    lst = [i.strip() for i in text.split("', '")]
    return lst

def flatten_stacked_list(lst):
    return list(itertools.chain(*lst))

def clean_protocol(protocol):
    try:
        protocol = protocol.lower()
    except:
        pdb.set_trace()
    protocol_split = protocol.split('\n')
    filter_out_empty_fn = lambda x: len(x.strip())>0
    strip_fn = lambda x: x.strip()
    protocol_split = list(filter(filter_out_empty_fn, protocol_split))	
    protocol_split = list(map(strip_fn, protocol_split))
    return protocol_split

def split_protocol(protocol):
    protocol_split = clean_protocol(protocol)
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

def collect_trials_from_topic(topic):
    '''
    Collect all trials from a topic.

    Parameters
    ----------
    topic: Topic
        A `Topic` object.
    
    Returns
    -------
    trials: List[str]
        A list of `Trial`'s `nct_id`.
    '''
    trials = []
    for trial_step in topic.trials:
        ids = [trial.attr_dict['nctid'] for trial in trial_step]
        trials.extend(ids)
    return trials