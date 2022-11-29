Patient-Trial Matching
======================

.. contents:: Table of Contents
    :depth: 2

The patient-trial matching is a process that identifies patients who are eligible to participate in a clinical trial. 
The matching process is based on the patient's electronic healthcare records (EHRs) and the trial's inclusion/exclusion criteria. 
To formulate it as a machine learning problem, we first need to define the input and output of the matching process.

A patient's EHR can be represented by a sequence of visits :math:`V = \{v_1, v_2, ..., v_n\}`, where each visit
:math:`v_i` is a sequence of events :math:`v_i = \{o_{i1}, o_{i2}, ..., o_{im}\}`. We can encode :math:`V` to a compact
embedding :math:`E_p \in R^d` as the patient's representation. The trial's inclusion/exclusion criteria can be represented by a
sequence of inclusion/exclusion criteria :math:`C = \{c_1, c_2, ..., c_m\}`. We can encode :math:`C` to a compact embedding
:math:`E_c \in R^{m\times d}` the trial's representation (each criterion is a vector). Such that, the neural matching process
is just compute the similarity between :math:`E_p` and :math:`E_c^m` to get the criterion-level affinity.



Patient-Trial Matching: Index
-----------------------------

Here is the list of colab examples on each model for this task.

- `Model 1: DeepEnroll <https://colab.research.google.com/drive/12JK9DCyHvMZuylgWZ6cDDLx8JC6sMait?usp=sharing>`_

- `Model 2: COMPOSE <https://colab.research.google.com/drive/1Ya5lStDgVFg9cw9i6uSehUXjPFBjLQgs?usp=sharing>`_

Patient-Trial Matching: Example
-------------------------------

Here, we highlight the usage of ``trial_patient_match.deepenroll`` model for this task. Other models have the similar pipeline.


