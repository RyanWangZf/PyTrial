Trial Patient Records Simulation
================================

.. contents:: Table of Contents
    :depth: 2

Synthetic patient records generation is a way around privacy issues when sharing clinical trial records or healthcare data.
Specifically, we want to train a generative model based on the real records, as :math:`p(h^{syn}|h_1,h_2,\dots,h_n; \theta)`, where
:math:`h^{syn}` is the synthetic records, :math:`h_1,h_2,\dots,h_n` are the real records, and :math:`\theta` are the parameters of the model.
When the patient data is a sequence, we can apply the generative model to the conditional generation. Given the
previous visits :math:`v_{1:t-1}`, we can generate the next visit record :math:`v_t` as :math:`v_t \sim p(v_t|v_{1:t-1}; \theta)`.

Depending ono the input patient data format: `tabular` or `sequence`, we have the following two subtasks:
``trial_simulation.tabular`` and ``trial_simulation.sequence``.




Tabular Patient: Index
----------------------

Here is the list of colab examples on each model for this task.

- `Model 1: GaussianCopula <https://colab.research.google.com/drive/1kG-I5oBWPwm3Cpm9A6MsJ9GNfm_iEVpB?usp=sharing>`_

- `Model 2: CopulaGAN <https://colab.research.google.com/drive/1-KTzb1xYPCCkRpAASID7d99HzXvwmHPM?usp=sharing>`_

- `Model 3: CTGAN <https://colab.research.google.com/drive/1MPdV9MIjQPSe_Z6rhS4PnRiP7YU1Pg1d?usp=sharing>`_

- `Model 4: TVAE <https://colab.research.google.com/drive/1HKmpwi2VosqYJion9vmRnW5s77jHaCxA?usp=sharing>`_


Sequential Patient: Index
-------------------------

Here is the list of colab examples on each model for this task.


- `Model 1: RNNGAN <https://colab.research.google.com/drive/16NxxGFvkqVo4SCv0gOO-0pPef8HTzIc6?usp=sharing>`_

- `Model 2: EVA <https://colab.research.google.com/drive/1o4IttNcmgmFQEWkQVKwuZvjFH7iLQvnc?usp=sharing>`_

- `Model 3: SynTEG <https://colab.research.google.com/drive/1S3dPkonI4c0A7uBf-1c9RIMEWA231T_u?usp=sharing>`_

- `Model 4: PromptEHR <https://colab.research.google.com/drive/1EbzLdSwTrbgsEgz8z70qzTLQWiPWlyRm?usp=sharing>`_


Tabular Patient: Example
------------------------
Here, we highlight the usage of ``trial_simulation.tabular.CTGAN`` model for this task.



Sequential Patient: Example
---------------------------
Here, we highlight the usage of ``trial_simulation.sequence.PromptEHR`` model for this task.
