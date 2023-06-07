'''
Provide data functions for trial data simulation.
'''
from collections import defaultdict
from pytrial.data.trial_data import TrialDatasetStructured, TrialDataCollator
from pytrial.data.site_data import SiteBase,  SiteBaseDemographics, ModalitySiteBase
from torch.utils.data import Dataset
import numpy as np
import torch

class TrialSiteSimple(Dataset):
    '''
    Load trial-site dataset with trial and site features.
    Each corresponds to a set of historical enrollment
    values for training trial site selection models.

    Parameters
    ----------
    site_data: df
        A df used to generate site data
    trial_data: pd.DataFrame
        A dataframe containing trial information including a set of paired trials named `label` and `enrollment` 
        which contains a mapping to sites in the site data that participated in the trial and the corresponding
        enrollments of those sites.
    '''
    def __init__(self, site_data, trial_data):
        self.sites = SiteBaseDemographics(site_data) if 'demographics' in site_data else SiteBase(site_data)
        self.mappings = trial_data.pop('label')
        self.enrollments = trial_data.pop('enrollment')
        self.trials = TrialDatasetStructured(trial_data)
        
    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        sitesPerTrial = self.mappings[idx]
        sample = {"trial": self.trials[idx], 
                  "site": [self.sites[int(s)] for s in sitesPerTrial],
                  "label": self.enrollments[idx], 
                  "eth_label": None if isinstance(self.sites, SiteBase) else [self.sites.get_label(sites) for sites in sitesPerTrial],
                 }
        return sample
    
class TrialSiteModalities(Dataset):
    '''
    Load trial-site dataset with trial and site features.
    Each corresponds to a set of historical enrollment
    values for training trial site selection models.

    Parameters
    ----------
    site_data: df
        A df used to generate site data
    trial_data: pd.DataFrame
        A dataframe containing trial information including a set of paired trials named `label` and `enrollment` 
        which contains a mapping to sites in the site data that participated in the trial and the corresponding
        enrollments of those sites.
    '''
    def __init__(self, site_data, trial_data):
        self.sites = ModalitySiteBase(site_data)
        self.mappings = trial_data.pop('label')
        self.enrollments = trial_data.pop('enrollment')
        self.trials = TrialDatasetStructured(trial_data)

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        sitesPerTrial = self.mappings[idx]
        siteFeatures = [self.sites[trialSites] for trialSites in sitesPerTrial]
        siteFeatures = {k: np.stack([d[k] for d in siteFeatures]) for k in siteFeatures[0]}
        sample = {"trial": self.trials[idx], 
                  "label": self.enrollments[idx], 
                  "eth_label": siteFeatures['eth_label'],
                  "inv_static": siteFeatures['x'],
                  "dx": siteFeatures['dx'],
                  "dx_len": siteFeatures['dx_lens'],
                  "rx": siteFeatures['rx'],
                  "rx_len": siteFeatures['rx_lens'],
                  "enroll_hist": siteFeatures['hist'],
                  "enroll_hist_len": siteFeatures['hist_lens'],
                  "inv_mask": np.zeros((len(sitesPerTrial), 4), dtype=np.int8)
                 }
        return sample

class SiteSelectionBaseCollator:
    '''
    support make collation of unequal sized list of batch for densely stored
    sequential visits data.

    Parameters
    ----------
    config: dict
        
        {

        'has_demographics': in 'true' or 'false'

        }
        
    '''
    has_demographics = False
    config = {'has_demographics':'False'} 

    def __init__(self, config=None):
        if config is not None:
            self.config.update(config)
            
        if str(self.config['has_demographics']) == 'True': self.has_demographics = True
        else: self.has_demographics = False

    def __call__(self, inputs):
        '''
        Paramters
        ---------
        inputs = {
            'trial': list[np.ndarray], 
        
            'site': list[list[np.ndarray]],
            
            'label': list[list[int]],
            
            'eth_label' None or list[list[np.ndarray]]
        }

        Returns
        -------
        outputs = {
            'trial': tensor, 
        
            'site': tensor,
            
            'label': tensor,
            
            'eth_label' None or tensor
        }
        '''
        # init output dict
        return_data = defaultdict(list)

        return_data['trial'] = torch.FloatTensor(np.array([i['trial'] for i in inputs])).squeeze(1)
        return_data['site'] = torch.FloatTensor(np.array([i['site'] for i in inputs])).squeeze(2)
        return_data['label'] = torch.FloatTensor(np.array([i['label'] for i in inputs]))

        # processing all
        if self.has_demographics:
            return_data['eth_label'] = torch.FloatTensor(np.array([i['eth_label'] for i in inputs]))

        return return_data
    
class SiteSelectionModalitiesCollator:
    '''
    support make collation of unequal sized list of batch for densely stored
    sequential visits data.

    Parameters
    ----------
    config: dict
        
        {

        'visit_mode': in 'dense' or 'tensor',

        'trial_mode': in 'dense' or 'tensor',
        
        'has_demographics': in 'true' or 'false',

        }
        
    '''
    is_tensor_visit = False
    is_tensor_trial = False
    has_demographics = False
    config = {'visit_mode':'dense', 'trial_mode':'dense', 'has_demographics':'false'} 

    def __init__(self, config=None):
        if config is not None:
            self.config.update(config)
            
        if self.config['visit_mode'] == 'tensor': self.is_tensor_visit = True
        else: self.is_tensor_visit = False

        if self.config['trial_mode'] == 'tensor': self.is_tensor_trial = True
        else: self.is_tensor_trial = False
        
        if self.config['has_demographics'] == 'true': self.has_demographics = True
        else: self.has_demographics = False

    def __call__(self, inputs):
        '''
        Paramters
        ---------
        inputs = {
            'trial': list[np.ndarray], 
            
            'label': list[list[int]],
            
            'eth_label' None or list[list[np.ndarray]]
            
            'inv_static': list[list[np.ndarray]]
            
            'dx': list[list[np.ndarray]]
            
            'dx_len': list[list[int]]
            
            'rx': list[list[np.ndarray]]
            
            'rx_len': list[list[int]]
            
            'enroll_hist': list[list[np.ndarray]]
            
            'enroll_hist_len': list[list[int]]
            
            'inv_mask': list[list[np.ndarray]]
        }

        Returns
        -------
        outputs = {
            'trial': tensor,
            
            'label': tensor,
            
            'eth_label' None or tensor,
            
            'inv_static': tensor
            
            'dx': tensor
            
            'dx_len': tensor
            
            'rx': tensor
            
            'rx_len': tensor
            
            'enroll_hist': tensor
            
            'enroll_hist_len': tensor
            
            'inv_mask': tensor
        }
        '''
        
        # init output dict
        return_data = defaultdict(list)
        
        inputs = {k: np.stack([d[k] for d in inputs]) for k in inputs[0]}
        
        return_data['trial'] = torch.FloatTensor(np.array(inputs['trial']))
        return_data['label'] = torch.FloatTensor(np.array(inputs['label']))
        return_data['inv_static'] = torch.FloatTensor(np.array(inputs['inv_static']))
        return_data['dx'] = torch.FloatTensor(np.array(inputs['dx']))
        return_data['dx_len'] = torch.LongTensor(np.array(inputs['dx_len']))
        return_data['rx'] = torch.FloatTensor(np.array(inputs['rx']))
        return_data['rx_len'] = torch.LongTensor(np.array(inputs['rx_len']))
        return_data['enroll_hist'] = torch.FloatTensor(np.array(inputs['enroll_hist']))
        return_data['enroll_hist_len'] = torch.LongTensor(np.array(inputs['enroll_hist_len']))
        return_data['inv_mask'] = torch.LongTensor(np.array(inputs['inv_mask']))

        # processing all
        if self.has_demographics:
            return_data['eth_label'] = torch.FloatTensor(np.array(inputs['eth_label']))
        
        return return_data