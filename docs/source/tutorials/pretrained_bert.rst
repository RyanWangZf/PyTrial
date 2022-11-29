Pretrained BERT Model
=====================

**PyTrial** provides an easy-to-use interface to load and use pretrained BERT model.

.. code:: python

    from pytrial.model_utils.bert import BERT

    # Load pretrained BERT model
    model = BERT()

    # encode
    emb = model.encode('The goal of life is comfort.')


The default pretrained BERT used is `emilyalsentzer/Bio_ClinicalBERT <https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT?text=The+goal+of+life+is+%5BMASK%5D.>`_.
You can swith to other pretrained BERT models by specifying the model name as

.. code:: python

    # specify model name
    model = BERT(bertname='bert-base-uncased')

The passed ``bertname`` should be one of the pretrained BERT models listed in `HuggingFace Model Hub <https://huggingface.co/models>`_.