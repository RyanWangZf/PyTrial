'''
Implement Logistic Regression model for tabular individual outcome
prediction in clinical trials.
'''
import pdb
import os
import joblib
import pickle

from sklearn.linear_model import LogisticRegression as lr_model

from pytrial.data.patient_data import TabularPatientBase
from pytrial.utils.check import check_checkpoint_file, check_model_dir, check_model_config_file, make_dir_if_not_exist
from .base import TabularIndivBase

class BuildModel:
    def __new__(self, config):
        if config['dual']:
            solver = 'liblinear'
        else:
            solver = 'lbfgs'
        model = lr_model(
            C = 1/config['weight_decay'],
            dual = config['dual'],
            solver = solver,
            max_iter = config['epochs'],
            )
        return model

class LogisticRegression(TabularIndivBase):
    '''
    Implement Logistic Regression model for tabular individual outcome
    prediction in clinical trials. Now only support `binary classification`.

    Parameters
    ----------
    weigth_decay: float
        Regularization strength for l2 norm; must be a positive float.
        Like in support vector machines, smaller values specify weaker regularization.

    dual: bool
        Dual or primal formulation. Dual formulation is only implemented for l2 penalty with liblinear solver.
        Prefer `dual=False` when `n_samples > n_features`.

    epochs: int
        Maximum number of iterations taken for the solvers to converge.

    experiment_id: str, optional (default='test')
        The name of current experiment. Decide the saved model checkpoint name.
    '''
    def __init__(self,
        weight_decay=1,
        dual=False,
        epochs=100,
        experiment_id='test',
        ) -> None:
        super().__init__(experiment_id=experiment_id)
        self.config = {
            'weight_decay': weight_decay,
            'dual': dual,
            'epochs': epochs,
            'experiment_id': experiment_id,
            'model_name': 'logistic_regression',
        }
        self._save_config(self.config)

    def fit(self, train_data, valid_data=None):
        '''Train logistic regression model to predict patient outcome
        with tabular input data.

        Parameters
        ----------
        train_data: dict
            {

            'x': TabularPatientBase or pd.DataFrame,

            'y': pd.Series or np.ndarray

            }

            - 'x' contain all patient features; 
            - 'y' contain labels for each row.

        valid_data: Ignored.
            Not used, present heare for API consistency by convention.
        '''
        self._input_data_check(train_data)
        self._build_model()

        if isinstance(train_data['x'], TabularPatientBase):
            dataset = train_data['x']
            x_feat = dataset.df
            y = train_data['y']
        else:
            x_feat = train_data['x']
            y = train_data['y']

        self._fit_model(x_feat, y)

    def predict(self, test_data):
        '''
        Make prediction probability based on the learned model.
        Save to `self.result_dir`.

        Parameters
        ----------
        test_data: dict
            {

            'x': TabularPatientBase or pd.DataFrame,

            'y': pd.Series or np.ndarray

            }

            - 'x' contain all patient features; 
            - 'y' contain labels for each row. Ignored for prediction function.

        Returns
        -------
        ypred: np.ndarray
            The predicted probability for each patient.
            
            - For binary classification, return shape (n, );
            - For multiclass classification, return shape (n, n_class).

        '''
        self._input_data_check(test_data)

        dataset = test_data['x']
        if isinstance(dataset, TabularPatientBase):
            x_feat = dataset.df
        else:
            x_feat = dataset

        ypred = self.model.predict_proba(x_feat)
        if ypred.shape[1] == 2: # binary
            ypred = ypred[:,1]

        # save results to dir
        pickle.dump(ypred, open(os.path.join(self.result_dir, 'pred.pkl'), 'wb'))
        if isinstance(test_data, dict):
            if 'y' in test_data:
                pickle.dump(test_data['y'], open(os.path.join(self.result_dir, 'label.pkl'), 'wb'))

        return ypred

    def save_model(self, output_dir=None):
        '''
        Save the learned logistic regression model to the disk.

        Parameters
        ----------
        output_dir: str or None
            The dir to save the learned model.
            If set None, will save model to `self.checkout_dir`.
        '''
        if output_dir is not None:
            make_dir_if_not_exist(output_dir)
        else:
            output_dir = self.checkout_dir

        self._save_config(self.config, output_dir=output_dir)
        ckpt_path = os.path.join(output_dir, 'indiv-tabular.model')
        joblib.dump(self.model, ckpt_path)

    def load_model(self, checkpoint=None):
        '''
        Save the learned logistic regression model to the disk.

        Parameters
        ----------
        checkpoint: str or None
            - If a directory, the only checkpoint file `.model` will be loaded.
            - If a filepath, will load from this file;
            - If None, will load from `self.checkout_dir`.
        '''
        if checkpoint is None:
            checkpoint = self.checkout_dir

        checkpoint_filename = check_checkpoint_file(checkpoint, suffix='model')
        config_filename = check_model_config_file(checkpoint)
        self.model = joblib.load(checkpoint_filename)
        self.config = self._load_config(config_filename)

    def _build_model(self):
        self.model = BuildModel(self.config)

    def _fit_model(self, x_feat, y):
        self.model.fit(x_feat, y)
