import numpy as np
import pandas as pd

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

