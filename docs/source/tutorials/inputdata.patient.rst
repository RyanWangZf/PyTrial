Basic Patient Data Class
========================

.. contents:: Table of Contents
    :depth: 2

**PyTrial** offers several basic data classes for organizing patient and trial data:

- :doc:`Document: pytrial.data.patient_data <../pytrial.data.patient_data>`

- :doc:`Document: pytrial.data.trial_data <../pytrial.data.trial_data>`

They define the basic structure of the input data and then the convenience for the
next model training and predicting. Thus, we use these standard data classes as the inputs for many tasks.
Here, we will show two examples for patient data building.

We categorize the input data into two types: **tabular** and **sequential** patient data. The former is
the data that can be represented as a table: each row is a patient and each column is a feature. The latter is 
the data that each patient has a sequence of visits, where each visit has multiple features, e.g., events, lab tests, etc.


Patient Data: Tabular
---------------------

Colab example is available at `Example: data.patient_data.TabularPatientBase <https://colab.research.google.com/drive/15zmeaahSC6v_sWZcMoSD_F6uHdT5g3px?usp=sharing>`_.

We have :py:class:`pytrial.data.patient_data.TabularPatientBase` for the tabular patient data.

.. code:: python
    
    from pytrial.data.patient_data import TabularPatientBase

Consider we get patient data in ``pandas.DataFrame`` format but the raw features are a mixture of texts, numbers, and missing values.
We usually need to preprocess the data before passing it to models. ``TabularPatientBase`` provides a convenient way to do this.

Let's first load the raw demo data for creating a ``TabularPatientBase`` instance.

.. code:: python

    # load the raw demo data
    from pytrial.data.demo_data import load_trial_patient_tabular
    data = load_trial_patient_tabular()

    # parse the raw data
    df = data['data']
    metadata = data['metadata']

Then, we can pass the raw dataframe to the target data class.


.. code:: python
    
    import rdt

    # create a TabularPatientBase instance
    patient_data = TabularPatientBase(
        df=df, # this contains the raw dataframe
        metadata= {

            'sdtypes': {
                'race': 'categorical',
                'target_label': 'boolean',
                }, # this contains the data types of each column
            
            'tranformers': {
                'race': rdt.transformers.FrequencyEncoder(),
                }, # this contains the transformers for each column
            },
        )

A list of available data transformers can be found on `<https://docs.sdv.dev/rdt/transformers-glossary/browse-transformers>`_.
By default, ``TabularPatientBase`` will *automatically* detect the data types of each column and apply the corresponding transformers,
e.g., ``rdt.transformers.FrequencyEncoder`` for categorical features, which means you can leave the ``metadata`` empty all the time, like this:

.. code:: python
    
    # leave the metadata empty and let the class automatically detect the data types and apply transformers
    patient_data = TabularPatientBase(df=df)

However, sometimes you may want to customize the data types and transformers
in case the automatically detected ones are wrong. That is why in the above example
we assign ``'race':'categorical'`` amd ``'race': rdt.tranformers.FrequencyEncoder()``, which will
push the dataclass to follow our custom settings.

Please notice that we are allowed to just pass ``'sdtypes'`` for one column without specifying the corresponding transformer, where the dataclass
will pick the default transformer for the passed data type. as the ``'target_label':'boolean'`` in the above example.

    
We can check the transformed tabular data by

.. code:: python

    # the transformed values
    patient_data.df

Besides, we can actually transform the data back to its original format by

.. code:: python
    
    # transform the data back to its original format
    df_raw = patient_data.reverse_transform()


Or pass another dataframe to the dataclass to be transformed like

.. code:: python

    # pass another dataframe to the dataclass to be transformed
    df_prime_transformed = patient_data.transform(df_prime)



Patient Data: Sequence
----------------------

Colab example is available at `Example: data.patient_data.SequencePatientBase <https://colab.research.google.com/drive/1GL2LifS9aZUYDc2npgsZ2zTUHfibuUo5?usp=sharing>`_.

We have :py:class:`pytrial.data.patient_data.SequencePatientBase` for the sequential patient data.

.. code:: python
    
    from pytrial.data.patient_data import SequencePatientBase


Load the raw demo data to see how to create a ``SequencePatientBase`` instance.

.. code:: python

    from pytrial.data.demo_data import load_synthetic_ehr_sequence
    data = load_synthetic_ehr_sequence()
    data.keys()
    '''
    dict_keys(['visit', 'feature', 'order', 'n_num_feature', 'y', 'voc', 'cat_cardinalities'])
    '''

    # the raw visit data
    data['visit'][0]
    '''
    [
        [[0, 1, 2, 3, 5, 7, 41, 313, 1], [0, 1, 82], [2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 51, 19, 26]],
        [[0, 1, 10, 69], [1, 4], [0, 2, 3, 6, 7, 41, 12, 13, 14, 16, 52, 54, 22, 28]]
    ]
    '''

    # the order
    data['order']
    '''
    ['diag', 'prod', 'med']
    '''

    # the vocabulary
    data['voc']
    '''
    {'diag': <promptehr.data.Voc at 0x7f150c615cd0>,
    'prod': <promptehr.data.Voc at 0x7f14af789750>,
    'med': <promptehr.data.Voc at 0x7f14af7b7310>}
    '''

In the above example, ``data['visit']`` should be a list of patients, where each patient is a list of visits, where each visit is a list of events.
That is, ``data['visit'][0]`` is the visits of the first patient, where ``data['visit'][0][0]`` is the first visit.

It should be noted that inside ``data['visit'][0][0]`` there are three lists, where each list contains
several events of the same type, for example, diagnosed diseases represented by ICD codes.

The ``data['order']`` is the order of the events in each visit, which should be the same for all patients. In the above example,
``data['order']`` is ``['diag', 'prod', 'med']``, which means the first list of ``data['visit'][0][0]`` is the diagnosed diseases, and so on.

The ``data['voc']`` contains the vocabularies for each event type. Each voc objective should have the same format as :py:class:`pytrial.data.vocab_data.Vocab`.


Once we have the raw data, we can create a ``SequencePatientBase`` instance.

.. code:: python

    # create a ``SequencePatientBase`` instance
    seqdata = SequencePatientBase(
        data={'v':data['visit'], 'y':data['y'], 'x':data['feature']},
        metadata={
            'visit':{
                'mode':'dense', 
                'order': data['order'] # need to parse the ``order`` here 
                },
            'label':{'mode':'tensor'},
            'voc':data['voc'], # need to parse the ``voc`` here
            'max_visit':20,
            }
        )

The parameter ``data`` contains the raw data including visits, label, and baseline features;
the parameter ``metadata`` customize the output data format.

Then, we can check the transformed data by

.. code:: python

    from torch.utils.data import DataLoader
    from pytrial.data.patient_data import SeqPatientCollator # we need a collation function to process the input SequencePatient dataset

    # let's see the outputs
    collate_fn = SeqPatientCollator()
    loader = DataLoader(seqdata, batch_size=2, collate_fn=collate_fn, num_workers=0)
    loader = iter(loader)
    batch = next(loader)
    print(batch.keys())
    '''
    dict_keys(['v', 'x', 'y'])
    '''

    batch['v'].keys()
    '''
    dict_keys(['diag', 'prod', 'med'])
    '''
    
    batch['v']['diag'][0]
    '''
    [[0, 1, 2, 3, 5, 7, 41, 313, 1], [0, 1, 10, 69]]
    '''


The dataloader returns the visits with keys corresponding to ``data['order']``, i.e., ``['diag', 'prod', 'med']`` in the above example.
``batch['v']['diag'][0]`` is the diagnosis events for the first patient, where there are two visits.

