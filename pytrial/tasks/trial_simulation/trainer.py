import pdb
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from pytrial.utils.trainer import Trainer
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional

class SeqSimTrainer(Trainer):
    '''
    Subclass the original trainer and provide specific evaluation functions and test_dataloader for sequence simulation.
    '''
    pass

class SeqSimEvaTrainer(Trainer):
    '''
    Subclass the original trainer and provide specific evaluation functions and test_dataloader for sequence simulation.
    '''
    def prepare_input(self, data):
        '''
        Prepare inputs for sequence simulation models.

        Parameters
        ----------
        data: dict[list]
            A single patient records.

        idx: int
            The patient index.

        vdx: int
            The target visit index.

        '''
        return self.model._prepare_input(data)
    
    def train_one_iteration(self, 
        max_grad_norm=None,
        warmup_steps=None,
        use_amp=None, 
        scaler=None,
        train_loss_dict=None):
        
        skip_scheduler = False
        num_train_objectives = len(self.train_dataloader)
        for train_idx in range(num_train_objectives):
            data_iterator = self.data_iterators[train_idx]
            loss_model = self.loss_models[train_idx]
            loss_model.zero_grad()
            loss_model.train()
            optimizer = self.optimizers[train_idx]
            scheduler = self.schedulers[train_idx]
            try:
                data = next(data_iterator)
            except StopIteration:
                data_iterator = iter(self.train_dataloader[train_idx])
                self.data_iterators[train_idx] = data_iterator
                data = next(data_iterator)
                
            
            inputs = self.prepare_input(data)
            if use_amp:
                with autocast():
                    loss_return = loss_model(inputs)
                    loss = loss_return['loss_value']
            else:
                loss_return = loss_model(inputs)
                loss = loss_return['loss_value']

            # update for a single sample's all visits
            if use_amp:
                scale_before_step = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                skip_scheduler = scaler.get_scale() != scale_before_step
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                optimizer.step()

            train_loss_dict[train_idx].append(loss.item())
        
        if use_amp:
            skip_scheduler = scaler.get_scale() != scale_before_step

        if not skip_scheduler and warmup_steps > 0:
            scheduler.step()

    def _build_optimizer(self,
        loss_models,
        warmup_steps,
        num_train_steps,
        optimizer_param,
        weight_decay,
        scheduler=None,
        ):
        optimizer_class = torch.optim.AdamW
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            param_optimizer = list(loss_model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_param)
            if warmup_steps > 0:
                scheduler_obj = self._build_scheduler(optimizer, scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)
            else:
                scheduler_obj = None
            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)
        return optimizers, schedulers
    
class SeqSimSyntegTrainer(Trainer):
    '''
    Subclass the original trainer and provide specific evaluation functions and test_dataloader for sequence simulation.
    '''
    def __init__(self, model: nn.Module, train_objectives: List[Tuple[DataLoader, nn.Module]], condition=False, test_data=None, test_metric=None, less_is_better=False, load_best_at_end=True, n_gpus=1, output_dir='./checkpoints/', **kwargs):
        super().__init__(model, train_objectives, test_data, test_metric, less_is_better, load_best_at_end, n_gpus, output_dir, **kwargs)
        self.condition = condition
    
    def prepare_input(self, data):
        '''
        Prepare inputs for sequence simulation models.

        Parameters
        ----------
        data: dict[list]
            A single patient records.

        idx: int
            The patient index.

        vdx: int
            The target visit index.

        '''
        return self.model._prepare_input(data)
    
    def add_condition(self, data):
        return self.model._add_condition(data)
    
    def train_one_iteration(self, 
        max_grad_norm=None,
        warmup_steps=None,
        use_amp=None, 
        scaler=None,
        train_loss_dict=None):
        
        skip_scheduler = False
        num_train_objectives = len(self.train_dataloader)
        for train_idx in range(num_train_objectives):
            data_iterator = self.data_iterators[train_idx]
            loss_model = self.loss_models[train_idx]
            loss_model.zero_grad()
            loss_model.train()
            optimizer = self.optimizers[train_idx]
            scheduler = self.schedulers[train_idx]
            try:
                data = next(data_iterator)
            except StopIteration:
                data_iterator = iter(self.train_dataloader[train_idx])
                self.data_iterators[train_idx] = data_iterator
                data = next(data_iterator)
                
            
            inputs = self.prepare_input(data)
            if self.condition:
                inputs = self.add_condition(inputs)
            if use_amp:
                with autocast():
                    loss_return = loss_model(inputs)
                    loss = loss_return['loss_value']
            else:
                loss_return = loss_model(inputs)
                loss = loss_return['loss_value']

            # update for a single sample's all visits
            if use_amp:
                scale_before_step = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                skip_scheduler = scaler.get_scale() != scale_before_step
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                optimizer.step()

            train_loss_dict[train_idx].append(loss.item())
        
        if use_amp:
            skip_scheduler = scaler.get_scale() != scale_before_step

        if not skip_scheduler and warmup_steps > 0:
            scheduler.step()

    def _build_optimizer(self,
        loss_models,
        warmup_steps,
        num_train_steps,
        optimizer_param,
        weight_decay,
        scheduler=None,
        ):
        optimizer_class = torch.optim.AdamW
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            param_optimizer = list(loss_model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_param)
            if warmup_steps > 0:
                scheduler_obj = self._build_scheduler(optimizer, scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)
            else:
                scheduler_obj = None
            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)
        return optimizers, schedulers

class SeqSimGANTrainer(Trainer):
    '''
    Subclass the original trainer and provide specific evaluation functions and test_dataloader for sequence simulation.
    '''
    def prepare_input(self, data, idx, vdx):
        '''
        Prepare inputs for sequence simulation models.

        Parameters
        ----------
        data: dict[list]
            A single patient records.

        idx: int
            The patient index.

        vdx: int
            The target visit index.

        '''
        return self.model._prepare_input(data, idx, vdx)
    
    def train_one_iteration(self, 
        max_grad_norm=None,
        warmup_steps=None,
        use_amp=None, 
        scaler=None,
        train_loss_dict=None):
        '''
        Default training one iteration steps, can be subclass can reimplemented.
        '''
        skip_scheduler = False
        num_train_objectives = len(self.train_dataloader)
        for train_idx in range(num_train_objectives):
            data_iterator = self.data_iterators[train_idx]
            loss_model = self.loss_models[train_idx]
            loss_model.zero_grad()
            loss_model.train()
            optimizer = self.optimizers[train_idx]
            scheduler = self.schedulers[train_idx]
            try:
                data = next(data_iterator)
            except StopIteration:
                data_iterator = iter(self.train_dataloader[train_idx])
                self.data_iterators[train_idx] = data_iterator
                data = next(data_iterator)

            # go through each single patient take the next visit as the target
            for idx, _ in enumerate(data['x']):
                num_visit = self.model._get_num_visit(data, idx)

                if num_visit < 2: # ignore patient with less than two visits
                    continue

                # go through each visit
                loss = 0 # for each sample
                for vdx in range(1, num_visit):
                    inputs = self.prepare_input(data, idx, vdx)
                    if use_amp:
                        with autocast():
                            loss_return = loss_model(inputs)
                            loss_value = loss_return['loss_value']
                    else:
                        loss_return = loss_model(inputs)
                        loss_value = loss_return['loss_value']
                    loss += loss_value

                # update for a single sample's all visits
                if use_amp:
                    scale_before_step = scaler.get_scale()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    skip_scheduler = scaler.get_scale() != scale_before_step
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                    optimizer.step()

            train_loss_dict[train_idx].append(loss.item())
        
        if use_amp:
            skip_scheduler = scaler.get_scale() != scale_before_step

        if not skip_scheduler and warmup_steps > 0:
            scheduler.step()

    def _build_optimizer(self,
        loss_models,
        warmup_steps,
        num_train_steps,
        optimizer_param,
        weight_decay,
        scheduler=None,
        ):
        optimizer_class = torch.optim.AdamW
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            param_optimizer = list(loss_model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

            if loss_model.name == 'discriminator_loss':
                # print([n for n, p in param_optimizer if (not any(nd in n for nd in no_decay)) and ('dis_module' in n)])
                optimizer_grouped_parameters = [
                    {'params': [p for n, p in param_optimizer if (not any(nd in n for nd in no_decay)) and ('dis_module' in n)], 'weight_decay': weight_decay},
                    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]
            
            else:
                # build optimizer for generator
                optimizer_grouped_parameters = [
                    {'params': [p for n, p in param_optimizer if (not any(nd in n for nd in no_decay)) and ('dis_module' not in n)], 'weight_decay': weight_decay},
                    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]
                # print([n for n, p in param_optimizer if (not any(nd in n for nd in no_decay)) and ('dis_module' not in n)])

            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_param)
            if warmup_steps > 0:
                scheduler_obj = self._build_scheduler(optimizer, scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)
            else:
                scheduler_obj = None
            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)
        
        return optimizers, schedulers