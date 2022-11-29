PyTrial API & Pipeline
======================

.. contents:: Table of Contents
    :depth: 2

As described in :doc:`Intro 1: Overview of PyTrial <overview>`, **PyTrial** maintains a consistent user interface,
including common functions like ``fit``, ``predict``, ``save_model``, and ``load_model``. Each task is defined by 
its input and output data.

Therefore, the general pipeline is as follows, we take a patient-level outcome prediction task as an example:

1. Prepare the input data for the task.

.. code:: python
   
    from pytrial.data.patient_data import TabularPatientBase
    from pytrial.utils.tabular_utils import MinMaxScaler
    from pytrial.utils.tabular_utils import read_csv_to_df

    # Read the data
    df = read_csv_to_df('./tabular_patient_outcome_data.csv', index_col=0)
    label = df['target_label']
    df = df.drop(['target_label'], axis=1)

    # Build the dataset for the specific task
    # Here, we are working on individual patient level outcome prediction taking tabular inputs.
    dataset = TabularPatientBase(df,
        metadata={
            'transformers':
                {'age': MinMaxScaler()}, # specify the data transformation for specific columns
        })


2. Import the models from the corresponding ``pytrial.tasks`` module.

.. code:: python

    from pytrial.tasks.indiv_outcome.tabular import LogisticRegression

    model = LogisticRegression()

3. Train the model using the ``fit`` function.

.. code:: python

    model.fit(
        {
        'x': dataset,
        'y': label,
        }
    )

4. Make the prediction using the ``predict`` function. And save the model using the ``save_model`` function.

.. code:: python

    # make predictions
    ypred = model.predict({'x': dataset})

    # save the model
    model.save_model('./model')


We can see that, except for the first step for data preparation, the rest of the steps are rather straightforward.
For the sake of supporting the data preparation, we provide a set of basic dataset classes in the ``pytrial.data`` module.
We also provide a set of children classes of them for the specific tasks, e.g., ``pytrial.tasks.trial_patient_match.data.PatientData`` and
``pytrial.tasks.trial_patient_match.data.TrialData`` considering the trial-patient matching task.

We will go through each task with concrete examples in the next chapters.

