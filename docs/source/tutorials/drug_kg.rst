Drug Knowledge Graph
====================

.. contents:: Table of Contents
    :depth: 2




Drug Transformer
----------------

**PyTrial** also offers utilities for processing and transforming drug related data. The first one is
:py:class:`pytrial.model_utils.drug.DrugTransformer` that works for mapping between different drug
terminologies, e.g., from ATC to NDC, from ATC to SMILES, etc.

A colab example is available at `demo_drug_utils.ipynb <https://github.com/RyanWangZf/PyTrial/blob/main/examples/model_utils/demo_drug_utils.ipynb>`_.

.. code:: python

    from pytrial.model_utils.drug import DrugTransformer

    # Create a transformer
    dt = DrugTransformer()

    # drug name to SMILES
    print(dt.name2smiles('Acetylsalicylic acid'))

    # NDC to ATC4
    print(dt.ndc2atc('00002140701'))

    # ATC4 to NDC
    print(dt.atc2ndc('C01BA'))

    # drug name to NDC
    print(dt.name2ndc('NEO*IV*Gentamicin'))

    # NDC to drug name
    print(dt.ndc2name('63323017302'))

    # ATC4 to drug name
    print(dt.atc2name('S01HA'))

    # drug name to ATC4
    print(dt.name2atc('atomoxetine'))

    # NDC to SMILES
    print(dt.ndc2smiles(['00002140701','00088222033']))

    # ATC4 to SMILES
    print(dt.atc2smiles(['S01HA','N06AX']))



Drug Graph
----------

:py:class:`pytrial.model_utils.drug.DrugGraph` is a class for creating a drug knowledge graph.

A colab example is available at `demo_drug_graph.ipynb <https://github.com/RyanWangZf/PyTrial/blob/main/examples/model_utils/demo_drug_graph.ipynb>`_.

.. code:: python

    from pytrial.model_utils.drug import DrugGraph

    dg = DrugGraph()

    print(dg.graph)
    '''
    DiGraph with 251947 nodes and 303921 edges
    '''

The yielded ``dg.Graph`` is a ``networkx.DiGraph`` object, which can be used to analyze the drug knowledge graph
or create a graph neural network model.


