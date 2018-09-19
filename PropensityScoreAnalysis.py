from sklearn import metrics
from sklearn.preprocessing import binarize

import numpy as np
import pandas as pd
import statsmodels.api as sm

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
        self.test = test[self.columns]
        self.ctrl = control[self.columns]

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


    def fit(self, model='logit', nmodels=1,
            regularized=False, formula=None, **kwargs):
        """Fit a propensity score model."""
        # select the model
        if   model == 'probit': model_class = sm.Probit
        elif model == 'logit':  model_class = sm.Logit
        else:
            raise ValueError(f'Modelo {model} não identificado.')

        self.models = []
        if nmodels == 1 or (len(self.test) == len(self.ctrl)):
            # it will not balance the data
            y = self.data['_SET_']
            X = self.data[self.columns]
            model_inst = model_class(y, X, **kwargs)
            model_res = model_inst.fit()
            self.models.append(model_res)
        else:
            for n in range(nmodels):
                print(f"Treinando modelo {n+1}/{nmodels}.")
                # will create balanced samples based on minor set.
                samp_size = min(len(self.test), len(self.ctrl))
                tsmp_index = self.test.sample(n=samp_size).index.values
                csmp_index = self.ctrl.sample(n=samp_size).index.values
                samp_index = np.concatenate([tsmp_index, csmp_index])
                print(len(tsmp_index), len(csmp_index), len(samp_index))

                y = self.data.loc[samp_index, '_SET_']
                X = self.data.loc[samp_index, self.columns]

                model_inst = model_class(y, X, **kwargs)
                model_res = model_inst.fit()
                self.models.append(model_res)

                #pred = self.model.predict(X)
                #pred = binarize(pred.values.reshape(-1, 1), threshold=0.5)

        #acc = metrics.accuracy_score(y, pred)
        #print("Accuracy:", acc)

