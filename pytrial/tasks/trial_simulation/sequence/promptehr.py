import pdb
from copy import deepcopy

import promptehr # third-part promptEHR package

from .base import SequenceSimulationBase
from ..data import SequencePatient

class BuildModel:
    def __new__(self, config):
        model = promptehr.PromptEHR(
            **config,
            )
        return model

class PromptEHR(SequenceSimulationBase):
    '''Implement a BART equiped with prompt learning for synthetic patient records simulation [1]_.

    Parameters
    ----------
    code_type: list[str]
        A list of code types that the model will learn and generate. For example, ``code_type=['diag','prod','med']``.

    token_dict: dict[list]
        A dictionary of new tokens (code events, e.g., ICD code) that the model needs to learn and generate.
        For example, ``token_dict={'diag':['D01','D02','D03'], 'prod':['P01','P02','P03'], 'med':['M01','M02','M03']}``.

    n_num_feature: int
        Number of numerical patient baseline features. Notice that it assumes that the input
        baseline features are **ALWAYS** numerical feature first. That is to say,
        the input baseline ``feature = [num1, num2, .., num_n, cat1, cat2,...]``.

    cat_cardinalities: list[int]
        The number of categories for each categorical patient baseline features.
        The input baseline ``feature = [num1, num2, .., num_n, cat1, cat2,...]``.

    epochs: int
        Num training epochs in total.

    batch_size: int
        Training batch size.

    eval_batch_size: int
        Evaluation batch size.
    
    evaluation_steps: int
        How many steps of updates then try to evaluate the trained models.
    
    learning_rate: float
        Training learning rate.
    
    weight_decay: float
        Training weight decay.
    
    num_worker: int
        Numer of dataloading paralleled processes.
    
    output_dir: str
        Training logs output to this folder.

    device: str or list[int]
        Should be str like `cuda:0` or `cpu`, otherwise should be a list GPU ids.
    
    Notes
    -----
    .. [1] Wang, Z. and Sun, J. (2022). PromptEHR: Conditional Electronic Healthcare Records Generation with Prompt Learning. EMNLP 2022.
    '''
    model = None
    sample_config = {
        'num_beams': 1, # >1: beam_sample; =1: sample_gen
        'no_repeat_ngram_size': 1,
        'do_sample': True,
        'num_return_sequences': 1,
        'code_type': '',
        'top_k': 1,
        'temperature': 1.0,
        'max_length': 6,
    }
    def __init__(self,
        code_type=None,
        n_num_feature=None,
        cat_cardinalities=None,
        epochs=50,
        batch_size=16,
        eval_batch_size=16,
        evaluation_steps=1000,
        learning_rate=5e-5,
        weight_decay=1e-4,
        num_worker=8,
        device='cuda:0',
        experiment_id='trial_simulation.sequence.promptehr',
        ):
        super().__init__(experiment_id)
        self.config = {
            'code_type': code_type,
            'n_num_feature':n_num_feature,
            'cat_cardinalities':cat_cardinalities,
            'epoch':epochs,
            'batch_size':batch_size,
            'eval_batch_size':eval_batch_size,
            'eval_step':evaluation_steps,
            'learning_rate':learning_rate,
            'weight_decay':weight_decay,
            'num_worker':num_worker,
            'output_dir':self.checkout_dir,
            'device':device,
        }
        config = deepcopy(self.config)
        self.model = BuildModel(config)
        self.device = device

    def fit(self, train_data, val_data=None):
        '''
        Fit PromptEHR model on the input training EHR data.

        Parameters
        ----------
        train_data: SequencePatient
            A `SequencePatient` contains patient records where 'v' corresponds to 
            visit sequence of different events.

        val_data: SequencePatient
            A `SequencePatient` contains patient records where 'v' corresponds to 
            visit sequence of different events.
        '''
        self.model.fit(train_data=train_data, val_data=val_data)

    def predict(self, test_data, n_per_sample=None, n=None, sample_config=None, verbose=True):
        '''
        Generate synthetic records based on input real patient seq data.

        Parameters
        ----------
        test_data: SequencePatient
            A `SequencePatient` contains patient records where 'v' corresponds to 
            visit sequence of different events.
        
        n: int
            How many samples in total will be generated.
        
        n_per_sample: int
            How many samples generated based on each indivudals.

        sample_config: dict
            Configuration for sampling synthetic records, key parameters:

            - 'num_beams': Number of beams in beam search, if set `1` then beam search is deactivated;
            - 'top_k': Sampling from top k candidates.
            - 'temperature': temperature to make sampling distribution flater or skewer.
        
        verbose: bool
            If print the progress bar or not.

        Returns
        -------
        fake_data: SequencePatient
            Synthetic patient records in `SequencePatient` format.
        '''
        return_res = self.model.predict(
            test_data=test_data, 
            n_per_sample=n_per_sample, 
            n=n, 
            sample_config=sample_config,
            verbose=verbose)

        fake_metadata = deepcopy(test_data.metadata)
        fake_data = {
            'v':return_res['visit'], 
            'x':return_res['feature'],
            }
        if 'y' in return_res:
            fake_data['y'] = return_res['y']
        fake_data = SequencePatient(data=fake_data, metadata=fake_metadata)
        return fake_data

    def load_model(self, checkpoint):
        '''
        Load model and the pre-encoded trial embeddings from the given
        checkpoint dir.

        Parameters
        ----------
        checkpoint: str
            The input dir that stores the pretrained model.
        '''
        self.model.load_model(checkpoint)

    def save_model(self, output_dir):
        '''
        Save the learned simulation model to the disk.

        Parameters
        ----------
        output_dir: str
            The dir to save the learned model.
        '''
        self.model.save_model(output_dir)

    def evaluate(self, test_data):
        '''
        Evaluate the perplexity of trained PromptEHR model on the input data, will test the perplexity
        for each type of codes.
        
        Parameters
        ----------
        test_data: PatientSequence
            Standard sequential patient records in `PatientSequence` format.
        '''
        self.model.evaluate(test_data)

    def from_pretrained(self, input_dir='./simulation/pretrained_promptEHR'):
        '''
        Load pretrained PromptEHR model and make patient EHRs generation.
        Pretrained model was learned from MIMIC-III patient sequence data.

        Parameters
        ----------
        input_dir: str
            The path to the pretrained model. If no found, will download from online and save
            to this folder.
        '''
        self.model.from_pretrained(input_dir=input_dir)
        self.config.update(self.model.config)
    
    def update_config(self, config):
        '''
        Update the model configuration. Will be useful when load pretrained model
        and want to finetune on new data.

        Parameters
        ----------
        config: dict
            The configuration for the model.
        '''
        self.config.update(config)
        self.model.update_config(config)