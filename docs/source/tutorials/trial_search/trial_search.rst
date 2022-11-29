Trial Similarity Search
=======================

.. contents:: Table of Contents
    :depth: 2

We implement trial search task that aims to find similar trials to a given trial. This function is rather useful
when practioners are designing new trials. They can refer to the retrieved historical trials, which provide
valueable information for the trial's design, outcomes, and also look for potential collaborations.
To be more specific, we concentrate on the *dense retrieval* where each trial document is encoded as a dense vector. Such that,
trial similarity is measured through the cosine similarity between the trial vectors.

Formally, a trial document consists of multiple components, as :math:`D = \{x^{title}, x^{intv}, x^{disc}, \dots \}`.
The target of dense retrieval is to convert the input document to a dense vector :math:`V \in R^d`, such that we can
compute the trial's similarities and identify similar trials.



Trial Similarity Search: Index
------------------------------

- `Model 1: Doc2Vec <https://colab.research.google.com/drive/1aPROMs6BbESwzD03Glzj6RlP9RUa0axe?usp=sharing>`_

- `Model 2: WhitenBERT <https://colab.research.google.com/drive/1scHXHxC-pcKu2inq4hiI44YFPH1_mtQr?usp=sharing>`_

- `Model 3: Trial2Vec <https://colab.research.google.com/drive/11-vhoHeV5hzsE-pfdh9mRL3PbwsLA-Cv?usp=sharing>`_


Trial Similarity Search: Example
--------------------------------

Here, we highlight the usage of ``trial_search.trial2vec`` model for this task. Other models have the similar pipeline.
