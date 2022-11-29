import pdb

import numpy as np
import torch
from torch import nn
from torch.autograd import grad as torch_grad

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
        logits = self.model(inputs['x'])
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
        logits = self.model(inputs['x'])
        y = inputs['y']
        if y.shape != logits.shape:
            logits = logits.reshape(y.shape)
        loss = self.xent_loss(logits, inputs['y'])
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
        logits, labels = self.model(inputs)
        loss = self.xent_loss(logits, labels)
        return {'loss_value':loss}
    
class MultilabelBinaryXentLossWithKLDivergence(nn.Module):
    '''
    The basic binary xent loss
    for multilabel classification with
    KL Divergence for VAE regularization.
    '''
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.xent_loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, kl_weight=1.0):
        logits, labels, kl_loss = self.model(inputs)
        loss = self.xent_loss(logits, labels)
        tot_loss = loss + (kl_weight * kl_loss)
        return {'loss_value': tot_loss}

class MSELoss(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.loss = nn.MSELoss()

    def forward(self, inputs):
        logits = self.model(inputs['x'])
        loss = self.loss(logits, inputs['y'])
        return {'loss_value':loss}


'''
Two GAN based loss for GAN-based generator.
'''
class GeneratorLoss(nn.Module):
    name = 'generator_loss'
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs):
        outputs = self.model(inputs)
        y_fake = outputs['y_fake']

        # naive GAN loss
        # loss_g = - y_fake.log().mean()

        # WGAN loss, more stable
        loss_g = -torch.mean(y_fake)

        return {'loss_value':loss_g}

class DiscriminatorLoss(nn.Module):
    name = 'discriminator_loss'
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
    
    def forward(self, inputs):
        outputs = self.model(inputs)
        y_real = outputs['y_real']
        y_fake = outputs['y_fake']

        # naive GAN loss
        # loss_d = - (1 - y_fake).log().mean() - y_real.log().mean()

        # WGAN loss, more stable
        loss_d = torch.mean(y_fake) - torch.mean(y_real)

        return {'loss_value':loss_d}

class DiscriminatorLossGP(nn.Module):
    name = 'discriminator_loss'
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
    
    def forward(self, inputs):
        outputs = self.model(inputs)
        y_real = outputs['y_real']
        y_fake = outputs['y_fake']

        # naive GAN loss
        # loss_d = - (1 - y_fake).log().mean() - y_real.log().mean()

        # WGAN loss, more stable
        loss_d = torch.mean(y_fake) - torch.mean(y_real)

        # gradient penalty
        gp = self.compute_gradient_penalty(outputs)
        loss_d += gp

        return {'loss_value':loss_d}

    def compute_gradient_penalty(self, outputs):
        '''
        Requires `self.model.infer_discriminator` implemented.
        '''
        if 'interpolates' in outputs:
            interpolates = outputs['interpolates']
            d_interpolates = outputs['d_interpolates']
        else:
            x_fake = outputs['x_fake']
            x_real = outputs['y']
            alpha = torch.tensor(np.random.randn(x_real.size(0))).to(x_fake.device)
            interpolates = (alpha * x_real + (1-alpha)*x_fake).requires_grad_(True).float()
            d_interpolates = self.model.infer_discriminator(interpolates)
        grad_outputs=torch.ones(d_interpolates.size()).to(d_interpolates.device)
        gradients = torch_grad(outputs=d_interpolates, inputs=interpolates, grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
