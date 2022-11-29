Trial Outcome Prediction
========================

.. contents:: Table of Contents
    :depth: 2

Making clinical trial outcome prediction is valuable at the planning stage. 
We can avoid running clinical trials that are probable to terminate early. 
Stakeholders can save unnecessary funding and time from those failing trials, 
reinvest in other promising trials or re-design those failing trials for a better outcome.

Formally, given the multi-aspect information of trials, e.g., drug molecules involved in this trial :math:`\{m_1,m_2, \dots\}`, 
trial eligibility criteria :math:`\{c_1,c_2 \dots\}`, target diseases :math:`\{d_1,d_2,\dots\}`, predict the trial outcomes :math:`y \in \{0,1\}` before the start of Phase I, 
where :math:`y=0` indicates the trial failure and :math:`y=1` indicates success. 
Or, add the trial patient records :math:`X=\{x_1,x_2,\dots\}` collected from previous stages, 
and predict the outcomes of Phase II or III :math:`y \in \{0,1\}`.




Trial Outcome Prediction: Index
--------------------------------

Here is the list of colab examples on each model for this task.

- `Model 1: LogisticRegression <https://colab.research.google.com/drive/1TKm3kNl3U1M_MClg145WP0eCT-ze8c87?usp=sharing>`_

- `Model 2: XGBoost <https://colab.research.google.com/drive/1sIZ8UP8n-fFkwOIlvduIkaQYay8f7U1n?usp=sharing>`_

- `Model 3: MLP <https://colab.research.google.com/drive/16EFeEWyXV17Q0mlEsmdz5rCmp2fL-6g-?usp=sharing>`_

- `Model 4: HINT <https://colab.research.google.com/drive/1RVVHjQYOaV2LnL7ugoF860XRCGmK6hMQ?usp=sharing>`_


Trial Outcome Prediction: Example
---------------------------------

Here, we highlight the usage of ``trial_outcome.hint`` model for this task.
