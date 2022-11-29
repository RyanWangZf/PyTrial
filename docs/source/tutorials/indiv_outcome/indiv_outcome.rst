Individual Patient Outcome Prediction
=====================================

.. contents:: Table of Contents
    :depth: 2

Making individual outcome predictions is the basic AI for healthcare task. We need to specify the target
response to predict, e.g., mortality :math:`y\in [0,1]`, readmission :math:`y \in [0,1]`, length of stay :math:`y \in [0,\infty]`, 
transform the input trial patient records to either static descriptive features :math:`x \in \mathbb{R}^d`` where :math:`d` is the number of features 
or sequential event features :math:`x \in \mathbb{R}^{v \times d}` where :math:`v` is the number of visits, then predict the target using the processed data.

Such that, depending on the input patient data format: `tabular` or `sequence`, we have the following two subtasks: ``indiv_outcome.tabular`` and ``indiv_outcome.sequence``.


Tabular Patient: Index
----------------------

Here is the list of colab examples on each model for this task.

- `Model 1: LogisticRegression <https://colab.research.google.com/drive/17J72_cZu1oCftrHuHkP_Wr1TyCcFeKiO?usp=sharing>`_

- `Model 2: XGBoost <https://colab.research.google.com/drive/1_uk8t6nXsJVIzsqe06SAPorolfFqYJlX?usp=sharing>`_

- `Model 3: MLP <https://colab.research.google.com/drive/1LBat2cfq_DjoC9eh1tk0y8MyeGYilxaj?usp=sharing>`_

- `Model 4: FTTransformer <https://colab.research.google.com/drive/1E0tAF17VtoZ-DKteIzYLbEC7dvTuEE3m?usp=sharing>`_

- `Model 5: TransTab <https://colab.research.google.com/drive/196BECLTNk-0c7XeUME6Z-iJzlfF402tM?usp=sharing>`_


Sequential Patient: Index
-------------------------

Here is the list of colab examples on each model for this task.

- `Model 1: RNN <https://colab.research.google.com/drive/1zaRtzpHoyNLipvya5hpIojoZIohG7Jf6?usp=sharing>`_

- `Model 2: RETAIN <https://colab.research.google.com/drive/19BVCKZLOpczr1gcAB_HiWgoaPSdbocSN?usp=sharing>`_

- `Model 3: RAIM <https://colab.research.google.com/drive/1GA2MHqwo-CFBkHoNvDlpwhMezgN-Ri0g?usp=sharing>`_

- `Model 4: Dipole <https://colab.research.google.com/drive/1A02leVsc6nEyWCmBGhLlqt1kEgYTcrU0?usp=sharing>`_

- `Model 5: StageNet <https://colab.research.google.com/drive/1x1X7nnKT4Rn1uwPCAW7pRM9VA7Lyy5uY?usp=sharing>`_



Tabular: Example
----------------

Here, we highlight the usage of ``indiv_outcome.tabular.transtab`` model for this task. Besides learning and predicting on a single tabular dataset, ``transtab`` shows promising performances 
on learning `across` different datasets. It sheds light on training a fundational model for clinical trials.



Sequence: Example
-----------------

Here, we highlight the usage of ``indiv_outcome.sequence.rnn`` model for this task. Other more advanced models have the similar pipeline.








