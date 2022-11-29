import pdb

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

def rocauc(pred, label):
    '''
    For binary classification.

    Parameters
    ----------
    pred: np.ndarray or torch.Tensor
        Prediction in shape (N, ) or (N, 1)
    
    label: np.ndarray or torch.Tensor
        Prediction in shape (N, ) or (N, 1)
    '''
    if isinstance(pred, torch.Tensor): pred = pred.detach().cpu().numpy()
    if isinstance(label, torch.Tensor): label = label.cpu().numpy()
    if pred.shape != label.shape: pred = pred.reshape(label.shape)
    score = roc_auc_score(label, pred)
    return {'auc':score}

def accuracy(pred, label):
    '''
    For multiclass classification.

    Parameters
    ----------
    pred: np.ndarray or torch.Tensor
        Prediction in shape (N, C)
    
    label: np.ndarray or torch.Tensor
        Prediction in shape (N, C) or in shape (N, )
    '''
    if isinstance(pred, torch.Tensor): pred = pred.detach().cpu().numpy()
    if isinstance(label, torch.Tensor): label = label.cpu().numpy()

    if len(label.shape) == 2:
        label = np.argmax(label, 1)
    
    pred = np.argmax(pred, 1)
    score = np.mean(pred == label)
    return {'acc':score}

def macro_auc(pred, label):
    '''
    For multilabel classification.

    Parameters
    ----------
    pred: np.ndarray or torch.Tensor
        Prediction in shape (N, C)
    
    label: np.ndarray or torch.Tensor
        Prediction in shape (N, C)
    '''
    if isinstance(pred, torch.Tensor): pred = pred.detach().cpu().numpy()
    if isinstance(label, torch.Tensor): label = label.cpu().numpy()

    score = roc_auc_score(label, pred)
    return {'auc':score}

def mse(pred, label):
    '''
    For regression.

    Parameters
    ----------
    pred: np.ndarray or torch.Tensor
        Prediction in shape (N, ) or (N, 1)
    
    label: np.ndarray or torch.Tensor
        Prediction in shape (N, ) or (N, 1)
    '''
    if isinstance(pred, torch.Tensor): pred = pred.detach().cpu().numpy()
    if isinstance(label, torch.Tensor): label = label.cpu().numpy()
    if pred.shape != label.shape: pred = pred.reshape(label.shape)

    score = np.mean(np.square(pred - label))
    return {'mse': score}

METRICS_DICT = {
    'binary': rocauc,
    'multiclass': accuracy,
    'multilabel': macro_auc,
    'regression': mse,
}
