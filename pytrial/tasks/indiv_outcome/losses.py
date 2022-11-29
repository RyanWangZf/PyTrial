import pdb

import torch
from torch import nn

class XentLoss(nn.Module):
    '''
    The basic xent loss
    for multi-class classification.
    '''
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.xent_loss = nn.CrossEntropyLoss()
    
    def forward(self, inputs):
        logits = self.model(inputs)
        loss = self.xent_loss(logits, inputs['y'])
        return {'loss_value':loss}

class BinaryXentLoss(nn.Module):
    '''
    The basic binary xent loss
    for binary classification.
    '''
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.xent_loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs):
        logits = self.model(inputs)
        y = inputs['y'].float()
        if y.shape != logits.shape:
            logits = logits.reshape(y.shape)
        loss = self.xent_loss(logits, y)
        return {'loss_value':loss}

class MultilabelBinaryXentLoss(nn.Module):
    '''
    The basic binary xent loss
    for multilabel classification.
    '''
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.xent_loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs):
        logits = self.model(inputs)
        loss = self.xent_loss(logits, inputs['y'])
        return {'loss_value':loss}


class MSELoss(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.loss = nn.MSELoss()

    def forward(self, inputs):
        logits = self.model(inputs)
        loss = self.loss(logits, inputs['y'])
        return {'loss_value':loss}