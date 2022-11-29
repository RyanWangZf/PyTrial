import pdb

import torch
from torch import nn

class COMPOSELoss(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.sm_loss = nn.CosineEmbeddingLoss(margin=0.3, reduction='none')
        self.model = model

    def forward(self, inputs):
        # forward
        pred = self.model(inputs)

        # compute label loss
        inc_ce_loss = self._compute_label_loss(pred, inputs, 'inc')
        exc_ce_loss = self._compute_label_loss(pred, inputs, 'exc')

        # compute embedding loss
        sm_loss_inc = self._compute_embed_loss(pred, inputs, 'inc')
        sm_loss_exc = self._compute_embed_loss(pred, inputs, 'exc')

        # all loss
        loss_value = (inc_ce_loss + exc_ce_loss + sm_loss_inc + sm_loss_exc) / 2

        return {'loss_value':loss_value}

    def _compute_label_loss(self, pred, inputs, ec='inc'):
        pred_inc = pred[f'logit_{ec}'] # bs_trial, bs_ehr, num_ec, num_label
        num_sample = min(pred_inc.shape[0], pred_inc.shape[1])        
        pred_inc_logit = pred_inc[range(num_sample),range(num_sample)].permute(0,2,1) # bs_trial==bs_patient, num_label, num_ec
        inc_label = inputs[f'{ec}_label'] # bs_trial, num_ec
        inc_label_mask = inputs[f'{ec}_emb_mask']
        
        # mask out unknown predictions to prevent model from making unknown predictions all the time
        unk_label_mask = torch.ones(inc_label_mask.shape, device=inc_label_mask.device)
        unk_label_mask[inc_label == 2] = 0
        inc_label_mask = inc_label_mask * unk_label_mask

        # compute masked loss
        inc_ce_loss = self.ce_loss(pred_inc_logit, inc_label.long())
        inc_ce_loss = torch.sum(inc_ce_loss * inc_label_mask) / inc_label_mask.sum()
        return inc_ce_loss

    def _compute_embed_loss(self, pred, inputs, ec='inc'):
        response_inc = pred[f'response_{ec}']
        query_inc = pred[f'query_{ec}']
        num_sample = min(response_inc.shape[0], response_inc.shape[1])
        response_inc_emb = response_inc[range(num_sample), range(num_sample)] # bs, num_ec, emb_dim
        query_inc_emb = query_inc[range(num_sample), range(num_sample)] # bs, num_ec, emb_dim
        inc_label = inputs[f'{ec}_label'] # bs_trial, num_ec
        inc_label_mask = inputs[f'{ec}_emb_mask']

        # build similarity loss label
        similarity_label = torch.ones(inc_label.shape, device=inc_label.device)
        unk_label_mask = torch.ones(inc_label.shape, device=inc_label.device)
        similarity_label[inc_label == 0] = -1 # unmatched
        similarity_label[inc_label == 1] = 1 # matched
        similarity_label[inc_label == 2] = 1
        unk_label_mask[inc_label == 2] = 0
        inc_label_mask = inc_label_mask * unk_label_mask

        sm_loss = self.sm_loss(response_inc_emb.view(-1, response_inc_emb.shape[-1]), query_inc_emb.view(-1, query_inc_emb.shape[-1]), similarity_label.view(-1,))
        sm_loss = torch.sum(inc_label_mask.view(-1) * sm_loss) / inc_label_mask.sum()

        return sm_loss


class DeepEnrollLoss(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.model = model

    def forward(self, inputs):
        # forward
        pred = self.model(inputs)

        # compute label loss
        inc_ce_loss = self._compute_label_loss(pred, inputs, 'inc')
        exc_ce_loss = self._compute_label_loss(pred, inputs, 'exc')

        # all loss
        loss_value = (inc_ce_loss + exc_ce_loss) / 2

        return {'loss_value':loss_value}

    def _compute_label_loss(self, pred, inputs, ec='inc'):
        pred_inc = pred[f'logit_{ec}'] # bs_trial, bs_ehr, num_ec, num_label
        num_sample = min(pred_inc.shape[0], pred_inc.shape[1])        
        pred_inc_logit = pred_inc[range(num_sample),range(num_sample)].permute(0,2,1) # bs_trial==bs_patient, num_label, num_ec
        inc_label = inputs[f'{ec}_label'] # bs_trial, num_ec
        inc_label_mask = inputs[f'{ec}_emb_mask']
        
        # mask out unknown predictions to prevent model from making unknown predictions all the time
        unk_label_mask = torch.ones(inc_label_mask.shape, device=inc_label_mask.device)
        unk_label_mask[inc_label == 2] = 0
        inc_label_mask = inc_label_mask * unk_label_mask

        # compute masked loss
        inc_ce_loss = self.ce_loss(pred_inc_logit, inc_label.long())
        inc_ce_loss = torch.sum(inc_ce_loss * inc_label_mask) / inc_label_mask.sum()
        return inc_ce_loss

