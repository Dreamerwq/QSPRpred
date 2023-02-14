"""Here the QSPRmodel classes can be found.

At the moment there is a class for sklearn type models
and one for a keras DNN model. To add more types a model class should be added, which
is a subclass of the QSPRModel type.
"""
import json
import math
import os
import os.path
import sys
from datetime import datetime
from functools import partial

import numpy as np
import optuna
import pandas as pd
import sklearn_json as skljson
from qsprpred import DEFAULT_DEVICE, DEFAULT_GPUS
from qsprpred.logs import logger
from qsprpred.models.interfaces import QSPRModel
from qsprpred.models.neural_network import STFullyConnected
from qsprpred.models.tasks import ModelTasks
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, ParameterGrid, train_test_split
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.svm import SVC, SVR


class QSPRsklearn(QSPRModel):
    """Model initialization, fit, cross validation and hyperparameter optimization for classifion/regression models.

    Attributes:
    data: instance QSPRDataset
    alg:  instance of estimator
    parameters (dict): dictionary of algorithm specific parameters

    Methods:
    init_model: initialize model from saved hyperparameters
    fit: build estimator model from entire data set
    objective: objective used by bayesian optimization
    bayesOptimization: bayesian optimization of hyperparameters using optuna
    gridSearch: optimization of hyperparameters using gridSearch
    """

    def __init__(self, base_dir, data, alg, alg_name, parameters=None):

        super().__init__(base_dir, data, alg, alg_name, parameters=parameters)
        # Adding scoring functions available for hyperparam optimization:
        self._supported_scoring = [
            'average_precision', 'neg_brier_score', 'neg_log_loss', 'roc_auc',
            'roc_auc_ovo', 'roc_auc_ovo_weighted', 'roc_auc_ovr', 'roc_auc_ovr_weighted']

        assert (len(set([prop.task.isClassification() for prop in self.data.targetProperties])) ==
                1), "All target properties must have the same task for sklearn multi-output models."

        # initialize models with defined parameters
        if self.parameters:
            self.model = self.alg.set_params(**self.parameters)
        else:
            if type(self.alg) in [SVC, SVR]:
                logger.warning("parameter max_iter set to 10000 to avoid training getting stuck. \
                                Manually set this parameter if this is not desired.")
                self.model = self.alg.set_params(max_iter=10000)
            else:
                self.model = self.alg

        logger.info('parameters: %s' % self.parameters)
        logger.debug('Model initialized: %s' % self.out)

        os.makedirs(os.path.dirname(self.out), exist_ok=True)

    def fit(self):
        """Build estimator model from entire data set."""
        X_all = self.data.getFeatures(concat=True).values
        y_all = self.data.getTargetPropertiesValues(concat=True).values.ravel()

        fit_set = {'X': X_all}

        if type(self.alg).__name__ == 'PLSRegression':
            fit_set['Y'] = y_all
        else:
            fit_set['y'] = y_all

        logger.info('Model fit started: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.model.fit(**fit_set)
        logger.info('Model fit ended: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        skljson.to_json(self.model, '%s.json' % self.out)

    def evaluate(self, save=True):
        """Make predictions for crossvalidation and independent test set.

        arguments:
            save (bool): don't save predictions when used in bayesian optimization
        """
        folds = self.data.createFolds()
        X, X_ind = self.data.getFeatures()
        y, y_ind = self.data.getTargetPropertiesValues()

        # initialize arrays for predictions
        if self.data.isMultiTask():
            if self.data.targetProperties[0].task == ModelTasks.REGRESSION:
                cvs = np.zeros((y.shape[0], len(self.data.targetProperties)))
            else:
                cvs = [np.zeros((y.shape[0], self.data.nClasses(prop))) for prop in self.data.targetProperties]
        else:
            if self.data.targetProperties[0].task == ModelTasks.REGRESSION:
                cvs = np.zeros(y.shape[0])
            else:
                cvs = np.zeros((y.shape[0], self.data.nClasses(self.data.targetProperties[0])))

        fold_counter = np.zeros(y.shape[0])

        # cross validation
        for i, (X_train, X_test, y_train, y_test, idx_train, idx_test) in enumerate(folds):
            logger.info('cross validation fold %s started: %s' % (i, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

            fold_counter[idx_test] = i

            fit_set = {'X': X_train}

            # self.data.createFolds() returns numpy arrays by default so we don't call `.values` here
            if type(self.alg).__name__ == 'PLSRegression':
                fit_set['Y'] = y_train.ravel()
            else:
                fit_set['y'] = y_train.ravel()
            self.model.fit(**fit_set)

            if self.data.targetProperties[0].task == ModelTasks.REGRESSION:
                cvs[idx_test] = self.model.predict(X_test)
            else:
                if not self.data.isMultiTask():
                    cvs[idx_test] = self.model.predict_proba(X_test)
                else:
                    preds = self.model.predict_proba(X_test)
                    for idx in range(len(self.data.targetProperties)):
                        cvs[idx][idx_test] = preds[idx]

            logger.info('cross validation fold %s ended: %s' % (i, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

        # fitting on whole trainingset and predicting on test set
        fit_set = {'X': X}

        if type(self.alg).__name__ == 'PLSRegression':
            fit_set['Y'] = y.values.ravel()
        else:
            fit_set['y'] = y.values.ravel()

        self.model.fit(**fit_set)

        if self.data.targetProperties[0].task == ModelTasks.REGRESSION:
            inds = self.model.predict(X_ind)
        else:
            inds = self.model.predict_proba(X_ind)

        # save crossvalidation results
        if save:
            train, test = y.add_prefix('Label_'), y_ind.add_prefix('Label_')
            for prop in self.data.targetProperties:
                train[prop.name], test[prop.name] = X[prop.name], X_ind[prop.name]
                if prop.task.isClassification:
                    train[f'Score'], test['Score'] = np.argmax(cvs, axis=1), np.argmax(inds, axis=1)
                    train = pd.concat([train, pd.DataFrame(cvs).add_prefix('Prob_')], axis=1)
                    test = pd.concat([test, pd.DataFrame(inds).add_prefix('Prob_')], axis=1)
                else:
                    train['Score'], test['Score'] = cvs, inds
            train['Fold'] = fold_counter
            train.to_csv(self.out + '.cv.tsv', sep='\t')
            test.to_csv(self.out + '.ind.tsv', sep='\t')

        return cvs

    def gridSearch(self, search_space_gs, scoring=None, n_jobs=1):
        """Optimization of hyperparameters using gridSearch.

        Arguments:
            search_space_gs (dict): search space for the grid search
            scoring (Optional[str, Callable]): scoring function for the grid search.
            n_jobs (int): number of jobs for hyperparameter optimization

        Note: Default `scoring=None` will use explained_variance for regression,
        roc_auc_ovr_weighted for multiclass, and roc_auc for binary classification.
        For a list of the available scoring functions see:
        https://scikit-learn.org/stable/modules/model_evaluation.html
        """
        if scoring is None:
            if self.data.targetProperties[0].task == ModelTasks.REGRESSION:
                scoring = 'explained_variance'
            elif self.data.targetProperties[0].nClasses > 2:  # multiclass
                scoring = 'roc_auc_ovr_weighted'
            else:
                scoring = 'roc_auc'
        grid = GridSearchCV(self.alg, search_space_gs, n_jobs=n_jobs, verbose=1, cv=(
            (x[4], x[5]) for x in self.data.createFolds()), scoring=scoring, refit=False)

        X, X_ind = self.data.getFeatures()
        y, y_ind = self.data.getTargetPropertiesValues()
        fit_set = {'X': X, 'y': y.iloc[:, 0].values.ravel()}
        logger.info('Grid search started: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        grid.fit(**fit_set)
        logger.info('Grid search ended: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        logger.info('Grid search best parameters: %s' % grid.best_params_)
        with open('%s_params.json' % self.out, 'w') as f:
            json.dump(grid.best_params_, f)

        self.model = self.alg.set_params(**grid.best_params_)

    def bayesOptimization(self, search_space_bs, n_trials, scoring=None, n_jobs=1):
        """Bayesian optimization of hyperparameters using optuna.

        Arguments:
            search_space_gs (dict): search space for the grid search
            n_trials (int): number of trials for bayes optimization
            scoring (Optional[str, Callable]): scoring function for the optimization.
            n_jobs (int): the number of parallel trials

        Example of search_space_bs for scikit-learn's MLPClassifier:
        >>> model = QSPRsklearn(base_dir='.', data=dataset,
        >>>                     alg = MLPClassifier(), alg_name="MLP")
        >>>  search_space_bs = {
        >>>    'learning_rate_init': ['float', 1e-5, 1e-3,],
        >>>    'power_t' : ['discrete_uniform', 0.2, 0.8, 0.1],
        >>>    'momentum': ['float', 0.0, 1.0],
        >>> }
        >>> model.bayesOptimization(search_space_bs=search_space_bs, n_trials=10)

        Avaliable suggestion types:
        ['categorical', 'discrete_uniform', 'float', 'int', 'loguniform', 'uniform']

        Note: Default `scoring=None` will use explained_variance for regression,
        roc_auc_ovr_weighted for multiclass, and roc_auc for binary classification.
        For a list of the available scoring functions see:
        https://scikit-learn.org/stable/modules/model_evaluation.html
        """
        print('Bayesian optimization can take a while for some hyperparameter combinations')
        if n_jobs > 1:
            logger.warning("At the moment n_jobs>1 not available for bayesoptimization. n_jobs set to 1")
            n_jobs = 1

        study = optuna.create_study(direction='maximize')
        logger.info('Bayesian optimization started: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        study.optimize(lambda trial: self.objective(trial, scoring, search_space_bs), n_trials, n_jobs=n_jobs)
        logger.info('Bayesian optimization ended: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        trial = study.best_trial

        logger.info('Bayesian optimization best params: %s' % trial.params)
        with open('%s_params.json' % self.out, 'w') as f:
            json.dump(trial.params, f)

        self.model = self.alg.set_params(**trial.params)

    def objective(self, trial, scoring, search_space_bs):
        """Objective for bayesian optimization.

        Arguments:
            trial (int): current trial number
            scoring (Optional[str]): scoring function for the objective.
            search_space_bs (dict): search space for bayes optimization
        """
        bayesian_params = {}

        for key, value in search_space_bs.items():
            if value[0] == 'categorical':
                bayesian_params[key] = trial.suggest_categorical(key, value[1])
            elif value[0] == 'discrete_uniform':
                bayesian_params[key] = trial.suggest_float(key, value[1], value[2], step=value[3])
            elif value[0] == 'float':
                bayesian_params[key] = trial.suggest_float(key, value[1], value[2])
            elif value[0] == 'int':
                bayesian_params[key] = trial.suggest_int(key, value[1], value[2])
            elif value[0] == 'loguniform':
                bayesian_params[key] = trial.suggest_float(key, value[1], value[2], log=True)
            elif value[0] == 'uniform':
                bayesian_params[key] = trial.suggest_float(key, value[1], value[2])

        print(bayesian_params)
        self.model = self.alg.set_params(**bayesian_params)

        y, y_ind = self.data.getTargetProperties()
        score_func = self.get_scoring_func(scoring)
        try:
            score = score_func(y, self.evaluate(save=False))
        except ValueError:
            logger.exception(
                "Only one class present in y_true. ROC AUC score is not defined in that case. Score set to -1.")
            score = -1
        return score

    def get_scoring_func(self, scoring):
        """Get scoring function from sklearn.metrics.

        Args:
            scoring (Union[str, Callable]): metric name from sklearn.metrics or
                user-defined scoring function.

        Raises:
            ValueError: If the scoring function is currently not supported by
                GridSearch and BayesOptimization.

        Returns:
            score_func (Callable): scorer function from sklearn.metrics (`str` as input)
            or user-defined function (`callable` as input)
        """
        # TODO: to add support for more scoring functions we will need to ensure that
        # the cross validation returns the correct input for the scoring function.
        # It's possible to inspect that by calling `str(scorer)` and checking the attributes.
        if all([scoring not in self._supported_scoring, isinstance(scoring, str)]):
            raise ValueError("Scoring function %s not supported. Supported scoring functions are: %s"
                             % (scoring, self._supported_scoring))
        elif callable(scoring):
            return scoring
        elif scoring is None:
            if self.data.task == ModelTasks.REGRESSION:
                scorer = metrics.get_scorer('explained_variance')
            elif self.data.nClasses > 2:  # multiclass
                scorer = metrics.get_scorer('roc_auc_ovr_weighted')
            else:
                scorer = metrics.get_scorer('roc_auc')
        else:
            scorer = metrics.get_scorer(scoring)
        return scorer._score_func


class QSPRDNN(QSPRModel):
    """This class holds the methods for training and fitting a Deep Neural Net QSPR model initialization.

    Here the model instance is created and parameters can be defined.

    Attributes:
        data: instance of QSPRDataset
        parameters (dict): dictionary of parameters to set for model fitting
        device (cuda device): cuda device
        gpus (int/ list of ints): gpu number(s) to use for model fitting
        patience (int): number of epochs to wait before early stop if no progress on validiation set score
        tol (float): minimum absolute improvement of loss necessary to count as progress on best validation score
    """

    def __init__(self, base_dir, data, parameters=None, device=DEFAULT_DEVICE, gpus=DEFAULT_GPUS, patience=50, tol=0):

        self.n_class = max(1, data.nClasses)
        super().__init__(
            base_dir,
            data,
            STFullyConnected(
                n_dim=data.X.shape[1],
                n_class=self.n_class,
                device=device,
                gpus=gpus,
                is_reg=data.targetProperties[0].task == ModelTasks.REGRESSION),
            "DNN",
            parameters=parameters)
        self._supported_scoring = [
            'average_precision', 'neg_brier_score', 'neg_log_loss', 'roc_auc',
            'roc_auc_ovo', 'roc_auc_ovo_weighted', 'roc_auc_ovr', 'roc_auc_ovr_weighted']
        self.patience = patience
        self.tol = tol

        # Initialize model with defined parameters
        if self.parameters:
            self.model = self.alg.set_params(**self.parameters)
        else:
            self.model = self.alg
        logger.info('parameters: %s' % self.parameters)
        logger.debug('Model intialized: %s' % self.out)

        self.optimal_epochs = -1

    def fit(self):
        """Train model on the trainings data, determine best model using test set, save best model.

        ** IMPORTANT: evaluate should be run first, so that the average number of epochs from the cross-validation
                        with early stopping can be used for fitting the model.
        """
        if self.optimal_epochs == -1:
            logger.error('Cannot fit final model without first determining the optimal number of epochs for fitting. \
                          first run evaluate.')
            sys.exit()

        X_all = self.data.getFeatures(concat=True).values
        y_all = self.data.getTargetPropertiesValues(concat=True).values

        self.model = self.model.set_params(**{"n_epochs": self.optimal_epochs})
        train_loader = self.model.get_dataloader(X_all, y_all)

        logger.info('Model fit started: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.model.fit(train_loader, None, self.out, patience=-1)
        with open('%s.json' % self.out, 'w') as fp:
            all_params = self.model.__dict__
            hyper_params = {k: all_params[k] for k in all_params if not k.startswith('_') and k not in [
                'training', 'device', 'gpus']}
            json.dump(hyper_params, fp)
        logger.info('Model fit ended: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    def evaluate(self, save=True, ES_val_size=0.1):
        """Make predictions for crossvalidation and independent test set.

        arguments:
            save (bool): wether to save the cross validation predictions
            ES_val_size (float): validation set size for early stopping in CV
        """
        X, X_ind = self.data.getFeatures()
        y, y_ind = self.data.getTargetPropertiesValues()
        indep_loader = self.model.get_dataloader(X_ind.values)
        last_save_epochs = 0

        cvs = np.zeros((y.shape[0], max(1, self.data.nClasses(self.data.targetProperties[0]))))
        fold_counter = np.zeros(y.shape[0])
        for i, (X_train, X_test, y_train, y_test, idx_train, idx_test) in enumerate(self.data.createFolds()):
            y_train = y_train.reshape(-1, 1)
            X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(
                X_train, y_train, test_size=ES_val_size)
            train_loader = self.model.get_dataloader(X_train_fold, y_train_fold)
            ES_valid_loader = self.model.get_dataloader(X_val_fold, y_val_fold)
            valid_loader = self.model.get_dataloader(X_test)
            last_save_epoch = self.model.fit(
                train_loader, ES_valid_loader, '%s_temp' %
                self.out, self.patience, self.tol)
            last_save_epochs += last_save_epoch
            logger.info(f'cross validation fold {i}: last save epoch {last_save_epoch}')
            os.remove('%s_temp_weights.pkg' % self.out)
            cvs[idx_test] = self.model.predict(valid_loader)
            fold_counter[idx_test] = i
        os.remove('%s_temp.log' % self.out)

        if save:
            n_folds = max(fold_counter) + 1
            self.optimal_epochs = int(math.ceil(last_save_epochs / n_folds)) + 1
            self.model = self.model.set_params(**{"n_epochs": self.optimal_epochs})
            train_loader = self.model.get_dataloader(X.values, y.values)
            self.model.fit(train_loader, None, '%s_temp' % self.out, patience=-1)
            os.remove('%s_temp_weights.pkg' % self.out)
            os.remove('%s_temp.log' % self.out)
            inds = self.model.predict(indep_loader)

            train, test = pd.Series(
                y.values.flatten()).to_frame(
                name='Label'), pd.Series(
                y_ind.values.flatten()).to_frame(
                name='Label')
            if self.data.targetProperties[0].task.isClassification():
                train['Score'], test['Score'] = np.argmax(cvs, axis=1), np.argmax(inds, axis=1)
                train = pd.concat([train, pd.DataFrame(cvs)], axis=1)
                test = pd.concat([test, pd.DataFrame(inds)], axis=1)
            else:
                train['Score'], test['Score'] = cvs, inds
            train['Fold'] = fold_counter
            train.to_csv(self.out + '.cv.tsv', sep='\t')
            test.to_csv(self.out + '.ind.tsv', sep='\t')

        if self.data.nClasses(self.data.targetProperties[0]) == 2:
            return cvs[:, 1]
        else:
            return cvs

    def gridSearch(self, search_space_gs, scoring=None, ES_val_size=0.1):
        """Optimization of hyperparameters using gridSearch.

        Arguments:
            search_space_gs (dict): search space for the grid search, accepted parameters are:
                lr (int) ~ learning rate for fitting
                batch_size (int) ~ batch size for fitting
                n_epochs (int) ~ max number of epochs
                neurons_h1 (int) ~ number of neurons in first hidden layer
                neurons_hx (int) ~ number of neurons in other hidden layers
                extra_layer (bool) ~ whether to add extra (3rd) hidden layer
            scoring (Optional[str, Callable]): scoring function for the grid search.
            ES_val_size (float): validation set size for early stopping in CV
        """
        score_func = self.get_scoring_func(scoring)
        logger.info('Grid search started: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        best_score = -np.inf
        for params in ParameterGrid(search_space_gs):
            logger.info(params)

            # do 5 fold cross validation and take mean prediction on validation set as score of parameter settings
            fold_scores = []
            for i, (X_train, X_test, y_train, y_test, idx_train, idx_test) in enumerate(self.data.createFolds()):
                y_train = y_train.reshape(-1, 1)
                logger.info('cross validation fold ' + str(i))
                X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(
                    X_train, y_train, test_size=ES_val_size)
                train_loader = self.model.get_dataloader(X_train_fold, y_train_fold)
                ES_valid_loader = self.model.get_dataloader(X_val_fold, y_val_fold)
                valid_loader = self.model.get_dataloader(X_test)
                self.model.set_params(**params)
                self.model.fit(train_loader, ES_valid_loader, '%s_temp' % self.out, self.patience, self.tol)
                os.remove('%s_temp_weights.pkg' % self.out)
                y_pred = self.model.predict(valid_loader)
                if self.data.nClasses(self.data.targetProperties[0]) == 2:
                    y_pred = y_pred[:, 1]
                fold_scores.append(score_func(y_test, y_pred))
            os.remove('%s_temp.log' % self.out)
            param_score = np.mean(fold_scores)
            if param_score >= best_score:
                best_params = params
                best_score = param_score

        logger.info('Grid search best parameters: %s' % best_params)
        with open('%s_params.json' % self.out, 'w') as f:
            json.dump(best_params, f)
        logger.info('Grid search ended: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        self.model = self.alg.set_params(**best_params)

    def bayesOptimization(self, search_space_bs, n_trials, scoring=None, n_jobs=1):
        """Bayesian optimization of hyperparameters using optuna.

        arguments:
            search_space_gs (dict): search space for the grid search
            n_trials (int): number of trials for bayes optimization
            scoring (Optional[str, Callable]): scoring function for the optimization.
            n_jobs (int): the number of parallel trials
        """
        print('Bayesian optimization can take a while for some hyperparameter combinations')
        # TODO add timeout function

        if n_jobs > 1:
            logger.warning("At the moment n_jobs>1 not available for bayesoptimization. n_jobs set to 1")
            n_jobs = 1

        study = optuna.create_study(direction='maximize')
        logger.info('Bayesian optimization started: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        study.optimize(lambda trial: self.objective(trial, scoring, search_space_bs), n_trials, n_jobs=n_jobs)
        logger.info('Bayesian optimization ended: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        trial = study.best_trial

        logger.info('Bayesian optimization best params: %s' % trial.params)
        with open('%s_params.json' % self.out, 'w') as f:
            json.dump(trial.params, f)

        self.model = self.alg.set_params(**trial.params)

    def objective(self, trial, scoring, search_space_bs):
        """Objective for bayesian optimization.

        arguments:
            trial (int): current trial number
            search_space_bs (dict): search space for bayes optimization
        """
        bayesian_params = {}

        for key, value in search_space_bs.items():
            if value[0] == 'categorical':
                bayesian_params[key] = trial.suggest_categorical(key, value[1])
            elif value[0] == 'discrete_uniform':
                bayesian_params[key] = trial.suggest_discrete_uniform(key, value[1], value[2], value[3])
            elif value[0] == 'float':
                bayesian_params[key] = trial.suggest_float(key, value[1], value[2])
            elif value[0] == 'int':
                bayesian_params[key] = trial.suggest_int(key, value[1], value[2])
            elif value[0] == 'loguniform':
                bayesian_params[key] = trial.suggest_float(key, value[1], value[2], log=True)
            elif value[0] == 'uniform':
                bayesian_params[key] = trial.suggest_float(key, value[1], value[2])

        self.model = self.alg.set_params(**bayesian_params)

        y, y_ind = self.data.getTargetProperties()
        score_func = self.get_scoring_func(scoring)
        try:
            score = score_func(y, self.evaluate(save=False))
        except ValueError:
            logger.exception(
                "Only one class present in y_true. ROC AUC score is not defined in that case. Score set to -1.")
            score = -1
        return score

    def get_scoring_func(self, scoring):
        """Get scoring function from sklearn.metrics.

        Args:
            scoring (Union[str, Callable]): metric name from sklearn.metrics or
                user-defined scoring function.

        Raises:
            ValueError: If the scoring function is currently not supported by
                GridSearch and BayesOptimization.

        Returns:
            score_func (Callable): scorer function from sklearn.metrics (`str` as input)
            or user-defined function (`callable` as input)
        """
        if all([scoring not in self._supported_scoring, isinstance(scoring, str)]):
            raise ValueError("Scoring function %s not supported. Supported scoring functions are: %s"
                             % (scoring, self._supported_scoring))
        elif callable(scoring):
            return scoring
        elif scoring is None:
            if self.data.task == ModelTasks.REGRESSION:
                scorer = metrics.get_scorer('explained_variance')
            elif self.data.nClasses > 2:  # multiclass
                # Calling metrics.get_scorer('roc_auc_ovr_weighted') in this context
                # raises the error `multi_class must be in ('ovo', 'ovr')` so let's avoid it
                scorer = metrics.get_scorer('roc_auc_ovr_weighted')
            else:
                scorer = metrics.get_scorer('roc_auc')
        else:
            scorer = metrics.get_scorer(scoring)
        return scorer._score_func
