# PyTrial: AI-Driven In Silico Clinical Trial Optimization

**PyTrial** is designed for both ML researchers and medical practioners. We provide a comprehensive collection of AI methods
for a series of clinical trial tasks. [[Tutorial]](https://pytrial.readthedocs.io/en/latest/tutorial.html).

---

**PyTrial** is featured for:

- Unified APIs, detailed documentation, and interactive examples with preprocessed demo data for every implemented algorithm.

- Cutting-edge AI4Trial algorithms reproduced from the most recent top-venue papers.

- Off-the-shelf pipelines for various clinical trial tasks: *patient outcome prediction*, *trial site selection*, *trial outcome prediction*, *patient-trial matching*, *trial similarity search*, and *trial data simulation*.

- Scalability to future research and development based on the PyTrial's architecture.

## Installation

- It is easy to install PyTrial from github source:

```bash
pip install git+https://github.com/RyanWangZf/pytrial.git@main
```

The package is tested on ``python==3.7``.

We **DO NOT** recommend downloading from PyPI temporarily because PyTrial is undergoing development swiftly.

## Philosophy
> In PyTrial, performing a task boils down to three steps: load data -> define model -> fit and predict.

To minimize the efforts learning to use PyTrial, we keep a consistent user interface for all tasks all models, i.e.,

```python
model.fit(train_data, val_data)

model.predict(test_data)

model.save_model(save_dir)

model.load_model(load_dir)
```

hence all tasks are defined the *input* and *output*. All we need to do is to prepare for the input following the protocol.

## Documentation

>We provide the following tutorials to help users get started with our PyTrial. After go through all these chapters, you will become the expert in AI for clinical trials and are ready to explore the frontier of this field.

### The principle of PyTrial

- [Intro 1: Overview of PyTrial](https://pytrial.readthedocs.io/en/latest/tutorials/overview.html)
- [Intro 2: PyTrial API & Pipeline](https://pytrial.readthedocs.io/en/latest/tutorials/pipeline.html)
- [Intro 3: Basic Patient Data Class](https://pytrial.readthedocs.io/en/latest/tutorials/inputdata.patient.html)
- [Intro 4: Basic Trial Data Class](https://pytrial.readthedocs.io/en/latest/tutorials/inputdata.trial.html)

### Tutorials for each task

- [Task 1: Individual Patient Outcome Prediction (`tasks.indiv_outcome`)](https://pytrial.readthedocs.io/en/latest/tutorials/indiv_outcome/indiv_outcome.html)
- [Task 2: Clinical Trial Site Selection (`tasks.site_selection`)](https://pytrial.readthedocs.io/en/latest/tutorials/site_selection/site_selection.html)
- [Task 3: Trial Outcome Prediction (`tasks.trial_outcome`)](https://pytrial.readthedocs.io/en/latest/tutorials/trial_outcome/trial_outcome.html)
- [Task 4: Patient-Trial Matching (`tasks.trial_patient_match`)](https://pytrial.readthedocs.io/en/latest/tutorials/trial_patient_match/trial_patient_match.html)
- [Task 5: Trial Similarity Search (`tasks.trial_search`)](https://pytrial.readthedocs.io/en/latest/tutorials/trial_search/trial_search.html)
- [Task 6: Trial Patient Records Simulation (`tasks.trial_simulation`)](https://pytrial.readthedocs.io/en/latest/tutorials/trial_simulation/trial_simulation.html)

### Additional utilities

- [Misc 1: Load Preprocessed Demo Data](https://pytrial.readthedocs.io/en/latest/tutorials/load_demo_data.html)
- [Misc 2: Prepare Oncology Trial Patient Data](https://pytrial.readthedocs.io/en/latest/tutorials/trial_patient_data.html)
- [Misc 3: Pretrained BERT Model](https://pytrial.readthedocs.io/en/latest/tutorials/pretrained_bert.html)
- [Misc 4: ICD9 & 10 Knowledge Graph](https://pytrial.readthedocs.io/en/latest/tutorials/icd_kg.html)
- [Misc 5: Drug Knowledge Graph](https://pytrial.readthedocs.io/en/latest/tutorials/drug_kg.html)


## Citing PyTrial

 If you use PyTrial in a scientific publication, we would appreciate citations to:

```bibtex
@misc{pytrial2022,
    title = {PyTrial: AI-driven In Silico Clinical Trial Optimization},
    author = {Wang, Zifeng and Theodorou, Brandon and Fu, Tianfan and Sun, Jimeng},
    year = {2022},
    month = {11},
    organization = {SunLab@UIUC},
    note = {Version 0.0.1},
}
```



