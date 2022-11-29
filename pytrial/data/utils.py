import pdb
import torch
import numpy as np 

from pytrial.data.patient_data import SequencePatientBase

def concat_dataset(*datasets):
    '''
    Concatenate two torch.utils.data.Dataset objects.
    At present, this function ONLY supports `pytrial.data.patient_data.SequencePatient` objects.
    It will ignore the part if one of the dataset does not have.
    For example, if one of the datasets does not have `label`, then the output concatenated dataset will not have `label`.

    Parameters
    ----------
    datasets: tuple[torch.utils.data.Dataset]
        A tuple of torch.utils.data.Dataset objects.
    '''
    for dataset in datasets:
        assert isinstance(dataset, SequencePatientBase), '`dataset` must be a `SequencePatient` object.'
    
    # concat the data
    label, visit, feature = [], [], []
    for dataset in datasets:
        visit.extend(dataset.visit)
        if dataset.feature is not None:
            feature.append(dataset.feature)
        if dataset.label is not None:
            label.append(dataset.label)

    data = {'v': visit}

    if len(feature) == len(datasets):
        feature = np.concatenate(feature, axis=0)
        data['x'] = feature
    
    if len(label) == len(datasets):
        label = np.concatenate(label, axis=0)
        data['y'] = label

    cls = SequencePatientBase
    
    new_data = cls(
        data=data,
        metadata=dataset.metadata,
        )
    return new_data

def split_dataset(dataset, split=0.8):
    '''
    Split a dataset into two datasets.

    At present, this function ONLY supports `pytrial.data.patient_data.SequencePatient` objects.
    '''
    assert isinstance(dataset, SequencePatientBase), '`dataset` must be a `SequencePatient` object.'

    cls = SequencePatientBase
    n = len(dataset)
    n1 = int(n * split)
    n2 = n - n1
    data1, data2 = torch.utils.data.random_split(dataset, [n1, n2])
    indice1 = data1.indices
    indice2 = data2.indices

    data1 =  {'v': [dataset.visit[ind] for ind in indice1],}
    
    if dataset.feature is not None:
        data1['x'] = dataset.feature[indice1]

    if dataset.label is not None:
        if isinstance(dataset.label, list):
            dataset.label = np.array(dataset.label)
        data1['y'] = dataset.label[indice1]

    new_data1 = cls(
        data=data1,
        metadata=dataset.metadata,
        )
    
    data2 = {'v': [dataset.visit[ind] for ind in indice2],}
    if dataset.feature is not None:
        data2['x'] = dataset.feature[indice2]

    if dataset.label is not None:
        if isinstance(dataset.label, list):
            dataset.label = np.array(dataset.label)
        data2['y'] = dataset.label[indice2]

    new_data2 = cls(
        data=data2,
        metadata=dataset.metadata,
        )

    return new_data1, new_data2