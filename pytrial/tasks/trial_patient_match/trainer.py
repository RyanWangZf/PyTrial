import pdb
import math

import numpy as np

from pytrial.utils.trainer import Trainer

class PatientTrialTrainer(Trainer):
    def prepare_input(self, data):
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
            # note that for each train objective there are two dataloaders:
            # patient dataloader and trial dataloader

            patient_data_iterator = self.data_iterators[train_idx][0]
            trial_data_iterator = self.data_iterators[train_idx][1]
            loss_model = self.loss_models[train_idx]
            loss_model.zero_grad()
            loss_model.train()
            optimizer = self.optimizers[train_idx]
            scheduler = self.schedulers[train_idx]

            # prepare input for training
            patient_batch = self._get_a_train_batch(patient_data_iterator, train_idx, 0)
            trial_batch = self._get_a_train_batch(trial_data_iterator, train_idx, 1)
            patient_batch.update(trial_batch)
            inputs = self.prepare_input(patient_batch)

            # update model using input batch of data
            if use_amp:
                loss_value, skip_scheduler, scale_before_step = self._update_one_iteration_amp(loss_model=loss_model, data=inputs, optimizer=optimizer, scaler=scaler, max_grad_norm=max_grad_norm)
            else:
                loss_value = self._update_one_iteration(loss_model=loss_model, data=inputs, optimizer=optimizer, max_grad_norm=max_grad_norm)

            train_loss_dict[train_idx].append(loss_value.item())

            if use_amp:
                skip_scheduler = scaler.get_scale() != scale_before_step

            if not skip_scheduler and warmup_steps > 0:
                scheduler.step()

    def evaluate(self):
        patient_loader, trial_loader = self.test_dataloader
        res = self.model._predict_on_dataloader(patient_loader, trial_loader, return_label=True)

        # trial-level acc
        pred_trial = res['pred_trial']
        label_trial = res['label_trial']
        pred_trial = np.concatenate([p[1] for p in pred_trial])
        acc = np.sum(pred_trial == label_trial) / np.prod(label_trial.shape)

        # criteria-level acc
        inc_ec_acc, exc_ec_acc = [], []
        for i, pred_inc in enumerate(res['pred_inc']):
            pred_inc = np.argmax(pred_inc.squeeze(0), -1) # all_patient, num_ec
            label_inc = res['label_trial_inc'][i]
            label_mask = np.ones(label_inc.shape)
            label_mask[label_inc == 2] = 0 # only count on label is match or unmatch
            inc_ec_acc_ = np.sum((pred_inc == label_inc).astype(int) * label_mask) / np.sum(label_mask)
            inc_ec_acc.append(inc_ec_acc_)

        for i, pred_exc in enumerate(res['pred_exc']):
            pred_exc = np.argmax(pred_exc.squeeze(0), -1) # all_patient, num_ec
            label_exc = res['label_trial_exc'][i]
            label_mask = np.ones(label_exc.shape)
            label_mask[label_exc == 2] = 0 # only count on label is match or unmatch
            exc_ec_acc_ = np.sum((pred_exc == label_exc).astype(int) * label_mask) / np.sum(label_mask)
            exc_ec_acc.append(exc_ec_acc_)

        return {
            'trial-level-acc':acc, 
            'inc-level-acc':np.mean(inc_ec_acc), 
            'exc-level-acc':np.mean(exc_ec_acc)
            }

    def get_test_dataloader(self, test_data):
        return self.model.get_test_dataloader(test_data)
    
    def _build_data_iterator(self, epochs):
        self.data_iterators = [[iter(dataloader_) for dataloader_ in dataloader] for dataloader in self.train_dataloader]
        steps_per_epoch = max([len(dataloader_) for dataloader_ in self.train_dataloader[0]])
        num_train_steps = int(steps_per_epoch * epochs)
        return  steps_per_epoch, num_train_steps

    def _get_a_train_batch(self, data_iterator, train_idx, data_idx):
        try:
            data = next(data_iterator)

        except StopIteration:
            data_iterator = iter(self.train_dataloader[train_idx][data_idx])
            self.data_iterators[train_idx][data_idx] = data_iterator
            data = next(data_iterator)
            
        return data