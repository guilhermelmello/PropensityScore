from sklearn import metrics
from sklearn.neighbors import KDTree
from sklearn.preprocessing import binarize

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import sys

class PropensityScoreAnalysis:
    """
    Propensity Score Analysis.
    """
    def __init__(self, test, control, keep_cols=None, drop_cols=None):
        # test and control labels
        self._TEST = 1
        self._CTRL = 0

        # keep track of variables
        assert (test.columns == control.columns).all()
        if keep_cols:
            self.columns = keep_cols
        elif drop_cols:
            self.columns = test.columns.drop(drop_cols)
        else:
            self.columns = test.columns

        # prepare test and control datasets
        self.test = test[self.columns].copy(deep=True)
        self.ctrl = control[self.columns].copy(deep=True)

        self.test['_SET_'] = self._TEST
        self.ctrl['_SET_'] = self._CTRL

        # make test and control with no index intersection
        index_int = self.test.index.intersection(self.ctrl.index)
        if len(index_int) > 0:
            self.data = pd.concat([self.test, self.ctrl], axis=0, ignore_index=True)
            self.test = self.data.query(f"_SET_ == {self._TEST}") # for index consistency
            self.ctrl = self.data.query(f"_SET_ == {self._CTRL}") # for index consistency
        else:
            self.data = pd.concat([self.test, self.ctrl], axis=0, ignore_index=False)


    def fit(self, model='logit', nmodels=1, regularized=False,
                random_state=None, **kwargs):
        """
        Fit a propensity score model.

        Parameters
        ----------
        model : str (default: 'logit')
            model to fit the data. Possible models
            are: ['probit'|'logit'|'glm'].
        nmodels : int
            number of models to be fitted.
        regularized :
            If True, nmodels will be fit with regularization.
        **kwargs :
            arguments for model initialization.
        """
        # select the model
        if model == 'probit':
            model_class = sm.Probit
        elif model == 'logit':
            model_class = sm.Logit
        elif model == 'glm':
            model_class = sm.GLM
            if 'family' not in kwargs:
                kwargs['family'] = sm.families.Binomial()
        else:
            raise ValueError(f'Modelo {model} não identificado.')

        self.models = []
        predictions = pd.DataFrame(index=self.data.index)
        for n in range(nmodels):
            sys.stdout.write(f"\rTraining model {n+1:3}/{nmodels}")
            if nmodels == 1 or (len(self.test) == len(self.ctrl)):
                # it will not balance the data
                y = self.data['_SET_']
                X = self.data[self.columns]
            else:
                # will create balanced samples based on minor set.
                sample_size = min(len(self.test), len(self.ctrl))
                tsmp_index = self.test.sample(n=sample_size, random_state=random_state).index.values
                csmp_index = self.ctrl.sample(n=sample_size, random_state=random_state).index.values
                samp_index = np.concatenate([tsmp_index, csmp_index])

                y = self.data.loc[samp_index, '_SET_']
                X = self.data.loc[samp_index, self.columns]

            # fitting model
            model_inst = model_class(y, X, **kwargs)
            if regularized:
                model_res = model_inst.fit_regularized()
            else:
                model_res = model_inst.fit()

            # save fitted model
            self.models.append(model_res)

        # save test and control scores
        self.test_score = self.predict(self.test)
        self.ctrl_score = self.predict(self.ctrl)

        # prepare data for accuracy computation
        targets = pd.concat([self.test['_SET_'], self.ctrl['_SET_']], axis=0)
        predictions = pd.concat([self.test_score, self.ctrl_score], axis=0)
        pred_index = predictions.index
        predictions = binarize(predictions.values.reshape(-1, 1), threshold=0.5)
        predictions = pd.Series(predictions.flatten(), index=pred_index)
        results = pd.DataFrame(dict(
            target=targets,
            predictions=predictions))

        # compute accuracy
        acc = metrics.accuracy_score(results.target, results.predictions)
        print(f"\nAccuracy: {acc*100:5.2f}%")


    def predict(self, data, threshold=None):
        """
        Computes Propensity Score

        Parameters
        ----------
        data : pandas.DataFrame
            input dataset to be scored
        threshold : float (default: None)
            if not None, will return binary classification based
            on threshold. Otherwise, will return a float score.

        Return
        ------
        array_like
            predicted score or classification
        """
        if 'models' not in self.__dict__:
            msg = "No fitted model. You have to use the 'fit' method."
            raise Exception(msg)

        predictions = pd.DataFrame(index=data.index)
        for model in self.models:
            pred = model.predict(data[self.columns])
            predictions = pd.concat([predictions, pred], axis=1)

        score = predictions.mean(axis=1)
        if threshold:
            score = binarize(score.values.reshape(-1, 1), threshold=threshold)
            score = pd.Series(score.flatten(), index=data.index)

        return score



