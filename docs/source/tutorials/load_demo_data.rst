Load Preprocessed Demo Data
===========================

In ``PyTrial``, we provide a set of preprocessed demo data for quick start. The :doc:`../pytrial.data.demo_data` module provides
a series of functions to load the demo data.

No need to download manually, the function will download the data automatically and save to the local disk, e.g.,

.. code:: python

    from pytrial.data.demo_data import load_synthetic_ehr_sequence

    # load the demo data
    data = load_synthetic_ehr_sequence()

    # or specify the data path
    data = load_synthetic_ehr_sequence(input_dir='./data')


Please refer to :doc:`../pytrial.data.demo_data` for a full list of available demo data.