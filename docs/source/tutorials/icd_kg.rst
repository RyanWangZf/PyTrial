ICD9 & 10 Knowledge Graph
=========================

.. contents:: Table of Contents
    :depth: 2

ICD Knowledge Graph
-------------------

**PyTrial** provides a way of creating ICD code graph and get the patient, children, and siblings of a given node.

The jupyter notebook example is available at `demo_icd_graph.ipynb <https://github.com/RyanWangZf/PyTrial/blob/main/examples/model_utils/demo_icd_graph.ipynb>`_.

To be specific, this function is supported by ``pytrial.model_utils.icd.ICD9Graph``, which returns an
``nxgraph`` property that is a ``networkx.DiGraph`` object.

.. code:: python

    from pytrial.model_utils.icd import ICD9Graph

    # will download from the remote and save
    graph = ICD9Graph()

    from pytrial.model_utils.icd import ICD10Graph
    graph = ICD10Graph(version='2021')


We can the children, parent, siblings of a node as

.. code:: python

    print(graph.children('Multiple gestation placenta status'))

    print(graph.parent('Multiple gestation placenta status'))

    print(graph.siblings('V9101'))


The ``networkx.DiGraph`` object can be used to do more complex graph analysis, for example, graph neural networks.


ICD Knowledge Query
-------------------

We also provide a bunch of functions to query the ICD knowledge graph, available at :doc:`../pytrial.model_utils.icd`.

For example,

.. code:: python

    from pytrial.model_utils.icd import get_icd10_from_nih

    print(get_icd10_from_nih('Multiple gestation placenta status'))

The function :func:`get_icd10_from_nih <pytrial.model_utils.icd.get_icd10_from_nih>` will return the ICD10 code of a given natural language description of a
term. This function is supported by a public API provided by  `<https://clinicaltables.nlm.nih.gov>`_.

The other three available functions are: :func:`get_icd9dx_from_nih <pytrial.model_utils.icd.get_icd9dx_from_nih>`,
:func:`get_icd9sg_from_nih <pytrial.model_utils.icd.get_icd9sg_from_nih>`, and :func:`get_condition_synonym_from_nih <pytrial.model_utils.icd.get_condition_synonym_from_nih>`.



