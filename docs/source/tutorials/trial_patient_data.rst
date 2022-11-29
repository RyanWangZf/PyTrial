Prepare Oncology Trial Patient Data
===================================

We provide an example of how to transform the raw patient-level records to sequence of patient data.

The jupyter notebook example is available at `process_NCT00174655.ipynb <https://github.com/RyanWangZf/PyTrial/blob/main/demo_data/demo_patient_sequence/trial/process_NCT00174655.ipynb>`_.

The raw input data comes from Project Data Sphere (PDS) and is available at `<https://data.projectdatasphere.org/projectdatasphere/html/access>`_.
You need to create an account (it's free) and download the data from trial **NCT00174655**, put the raw data
under the folder ``./breast cancer/NCT00174655/``.

Note that the raw data are in SAS form ``.sas7bdat``. We need to install ``pip install sas7bdat`` package to read the data.

