Basic Trial Data Class
======================

.. contents:: Table of Contents
    :depth: 2

**PyTrial** offers several basic data classes for organizing patient and trial data:

- :doc:`Document: pytrial.data.patient_data <../pytrial.data.patient_data>`

- :doc:`Document: pytrial.data.trial_data <../pytrial.data.trial_data>`

They define the basic structure of the input data and then the convenience for the
next model training and predicting. Thus, we use these standard data classes as the inputs for many tasks.
Here, we will show two examples for trial data building.


Trial Data: Trial Document Data
-------------------------------

Colab example is available at `Example: data.trial_data.TrialDatasetBase <https://colab.research.google.com/drive/1E9whQAu7YVyRzVpLcuO_FAvZYhDOjDsa?usp=sharing>`_.

We have a trial document data class :class:`pytrial.data.trial_data.TrialDatasetBase` for organizing the trial data.

The first step is also loading the raw data:

.. code:: python

    from pytrial.data.demo_data import load_trial_outcome_data

    # load raw demo data, will download from the remote if not available in local
    data = load_trial_outcome_data()

    data.keys()
    '''
    dict_keys(['data'])
    '''

    data['data']
    '''
    A pd.DataFrame
    '''

Building a trial document dataset is as easy as building a tabular data like

.. code:: python

    from pytrial.data.trial_data import TrialDatasetBase

    # build a trial document dataset
    trial_dataset = TrialDatasetBase(data=data['data'].iloc[:100])
    
    trial_dataset.df['inclusion_criteria'][:2]
    '''
    0    [inclusion criteria:, -  age >= 18 years., -  ...
    1    [inclusion criteria:, -  rare tumor, -  metast...
    Name: inclusion_criteria, dtype: object
    '''

    len(trial_dataset.df['inclusion_criteria'].iloc[0])
    '''
    12
    '''


In the above example, ``trial_dataset.df`` is the transformed version of the raw input dataframe.
Specifically, it splits raw crieteria into a list of single criterion as:

.. code:: python

    trial_dataset.df['inclusion_criteria'][0]
    '''

    ['inclusion criteria:',
    '-  age >= 18 years.',
    '-  histological or cytological diagnosis of metastatic stage iv or locally advanced,',
    'amenable to local therapy with curative intent.',
    ...
    ]
    '''

We can further use a BERT model to encode each criterion to get their criterion-level embeddings by

.. code:: python

    # get the embeddings of inclusion and exclusion criteria sentences
    # using a pretrained BERT model
    # need GPU to run this
    trial_dataset.get_ec_sentence_embedding()
    
    trial_dataset.inc_ec_embedding.shape
    '''
    torch.Size([1986, 768])
    '''

    len(trial_dataset.inc_vocab)
    '''
    1986
    '''

    trial_dataset.inc_vocab.idx2word[2]
    '''
    '-  age >= 18 years.'
    '''

After we call ``get_ec_sentence_embedding``, the dataset will have new attributes
for inclusion/exclusion criteria and their embeddings.

