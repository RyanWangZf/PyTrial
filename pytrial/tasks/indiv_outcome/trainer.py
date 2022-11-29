import pdb

from torch.utils.data import DataLoader

from pytrial.utils.trainer import Trainer
from .metrics import METRICS_DICT

class IndivTabTrainer(Trainer):
    '''
    Subclass the original trainer and provide specific evaluation functions and test_dataloader.
    '''
    def evaluate(self):
        mode = self.model.config['mode']
        metric_fn = METRICS_DICT[mode]
        res = self.model._predict_on_dataloader(self.test_dataloader)
        eval_res = metric_fn(res['pred'], res['label'])
        return eval_res

    def get_test_dataloader(self, test_data):
        dataset = self.model._build_dataset(test_data)
        dataloader = DataLoader(dataset, 
            batch_size=self.model.config['batch_size'], 
            shuffle=False, 
            num_workers=self.model.config['num_worker'],
            pin_memory=True,
            )        
        return dataloader

class IndivSeqTrainer(Trainer):
    '''
    Subclass the original trainer for indiv_outcome.sequence models.
    '''
    def prepare_input(self, data):
        return self.model._prepare_input(data)

    def evaluate(self):
        mode = self.model.config['mode']
        metric_fn = METRICS_DICT[mode]
        res = self.model._predict_on_dataloader(self.test_dataloader)
        if res['pred'].shape != res['label']:
            res['pred'] = res['pred'].reshape(res['label'].shape)
        eval_res = metric_fn(res['pred'], res['label'])
        return eval_res

    def get_test_dataloader(self, test_data):
        return self.model.get_test_dataloader(test_data)
