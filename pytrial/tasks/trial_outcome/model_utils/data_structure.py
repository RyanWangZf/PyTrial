from collections import defaultdict
import pdb

import numpy as np
from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence

from .data_utils import collect_trials_from_topic

class Trial:
    '''
    Contains information about a single trial.

    A trial is associated with multiple properties.
    
    Parameters
    ----------
    nctid : str
        The unique identifier of a trial. Usually be the nctid linked to clnicaltrials.gov.

    title: str
        The title of the clinical trial.

    label: int
        The label of the clinical trial. 1 for success, 0 for failure. 

    status: str
        The status of the trial. Usually be 'Completed', 'Terminated', 'Withdrawn', 'Enrolling by invitation', 'Active, not recruiting', 'Recruiting', 'Suspended', 'Approved for marketing', 'Temporarily not available', 'Available', 'No longer available', 'Unknown status'.

    year: int
        The year when the trial starts.
    
    end_year: int (default=None)
        The year when the trial ends.

    phase: str (default=None)
        The phase of the trial. Usually be 'Phase 1', 'Phase 2', 'Phase 3', 'Phase 4', 'N/A'.

    diseases: list[str] (default=None)
        A list of diseases (in ICD codes) associated with the trial.
    
    drugs: list[str] (default=None)
        A list of drug names associated with the trial.
    
    smiles: list[str] (default=None)
        A list of SMILE string described the associated the drugs.

    inc_criteria: str (default=None)
        The inclusion criteria of the trial.
    
    exc_criteria: str (default=None)
        The exclusion criteria of the trial.

    description: str (default=None)
        The description of the trial.

    why_stop: str (default=None)
        The reason why a trial stops.

    Attributes
    ----------
    attr_dict: dict
        A dictionary of all the attributes of the trial.
    '''
    def __init__(self, nctid=None, title=None, label=None, status=None, year=None, end_year=None, phase=None, diseases=None, drugs=None, smiles=None, inc_criteria=None, exc_criteria=None, description=None, why_stop=None, **kwargs):
        self.attr_dict = {
            'nctid': nctid,
            'title': title,
            'label': label,
            'status': status,
            'year': year,
            'end_year': end_year,
            'phase': phase,
            'diseases': diseases,
            'drugs': drugs,
            'smiles': smiles,
            'inclusion_criteria': inc_criteria,
            'exclusion_criteria': exc_criteria,
            'description': description,
            'why_stop': why_stop,
        }
        self.attr_dict.update(kwargs)
    
    def __repr__(self):
        line = f"\n\tTrial with id {self.attr_dict['nctid']}:" \
            f"\n\ttitle: {self.attr_dict['title']}" \
                f"\n\tyear: {self.attr_dict['year']}" \
                    f"\n\tstatus: {self.attr_dict['status']}"
        return line

class Topic:
    '''
    Contains the information about a sequence of trial lists.

    A `Topic` is associated with multiple `Trial` which are conducted at different timestamps.

    Parameters
    ----------
    topic_id: str
        The topic this sequence of trial belongs to.
    
    trials: List[List[Trial]]
        A list of lists of trials. Each list of trials is conducted at the same timestamp.
    
    Attributes
    ----------
    topic_id: str
        The topic this sequence of trial belongs to.
    
    trials: List[List[Trial]]
        A list of lists of trials. Each list of trials is conducted at the same timestamp.
    '''
    def __init__(self, topic_id, trials):
        self.topic_id = topic_id
        self.trials = trials
        self.number_of_trials = sum([len(trial_step) for trial_step in self.trials])
    
    def __repr__(self):
        line = f"Trial topic id {self.topic_id}:" \
            f"\n\t# of timesteps: {len(self.trials)}; # of trials: {self.number_of_trials}."
        return line
    
    def __getitem__(self, idx):
        '''
        Get the trials at the given index of timestep.
        '''
        return self.trials[idx]
    
    def __len__(self):
        '''
        Get the number of timesteps.
        '''
        return len(self.trials)
    
    def add(self, topic):
        '''
        Add trials in a topic object to the current topic object.

        Parameters
        ----------
        topic: Topic
            The topic object to be added.
        '''
        assert self.topic_id == topic.topic_id, 'The topic id of the two topic objects are not the same.'
        i = 0
        for trial_step in topic.trials:
            timestamp = trial_step[0].attr_dict['year']
            while i < len(self.trials) and self.trials[i][0].attr_dict['year'] < timestamp:
                i += 1
            if i == len(self.trials):
                self.trials.append(trial_step) # append to the last of the list
            else:
                # insert into the specific timestamp, the year might not be the same as the timestamp
                # because added trials might be between the two timestamps in the original topic
                self.trials[i].extend(trial_step)     
        self.number_of_trials = sum([len(trial_step) for trial_step in self.trials])

class SequenceTrial(Dataset):
    '''
    A dataset consisting of multiple topics of clinical trials. Each topic is a list of clinical trials under this topic ordered with their timestamps.

    Parameters
    ----------
    topics: List[Topic]
        A list of `Topic` objects.
    
    Attributes
    ----------
    topics: List[Topic]
        A list of `Topic` objects.
    
    new_trials: List[Trial]
        A list of new trials that are not in the original topics after added.
        Every add operation will add an new list of trials to this list.
        This list is used to keep track of the new trials added to the dataset then
        can be used to evaluate the model's performance on newly added trials.
    '''
    def __init__(self, topics):
        self.topics = topics
        self.indices = np.array([topic.topic_id for topic in self.topics])
        self.new_trials = []

        # build trials from topics
        self.trials = defaultdict(list) # a dictionary of trials with topic id as key
        for topic in self.topics:
            nctids = collect_trials_from_topic(topic)
            self.trials[topic.topic_id].extend(nctids)
    
    def __len__(self):
        return len(self.topics)
    
    def __getitem__(self, idx):
        return self.topics[idx]

    def __repr__(self):
        number_of_trials = sum([topic.number_of_trials for topic in self.topics])
        line = f"TrialSequence: # of topics: {len(self.topics)}; # of trials: {number_of_trials}."
        return line

    def add(self, topics):
        '''
        Add a new topic or merge a topic of trials to the dataset.

        Parameters
        ----------
        topic: Topic or List[Topic]
            A `Topic` or list of `Topic` objects.
        '''
        if isinstance(topics, Topic):
            topics = [topics]
        
        new_trials = []
        for topic in topics:
            if topic.topic_id in self.indices:
                idx = np.where(self.indices == topic.topic_id)[0][0]
                self.topics[idx].add(topic)
            else:
                print(f'{topic.topic_id} not found, add a new topic.')
                self.topics.append(topic)
                self.indices = np.append(self.indices, topic.topic_id)
            trial_ids = collect_trials_from_topic(topic)
            new_trials.extend(trial_ids)
            self.trials[topic.topic_id].extend(trial_ids)
        self.new_trials.append(new_trials)