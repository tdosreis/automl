import numpy as np
from sklearn.utils import check_array
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierAnalysis(BaseEstimator, TransformerMixin):

    def __init__(self, thresh=1.5, method='iqr'):
        self.thresh = thresh
        self.method = method

    def _validate_input(self, X, force_all_finite=False):
        X = check_array(X, force_all_finite=force_all_finite)
        return X

    def _which_method(self, X):

        if self.method == 'iqr':
            return self._outlier_iqr(X)

        if self.method == 'z_score':
            return self._z_score(X)

        if self.method == 'mod_z_score':
            return self._mod_z_score(X)

    def _replace_map(self, X):
        replacements = ({
            i: self._which_method(X[:, i])
            for i in range(X.shape[1])
        })
        return replacements

    def fit(self, X, y=None):
        X = self._validate_input(X)
        self.statistics_ = self._replace_map(X)
        return self

    def transform(self, X, force_all_finite=True):
        X = check_array(X, force_all_finite=force_all_finite)
        X_new = X.copy()
        replacements = self.statistics_
        for i in range(X_new.shape[1]):
            self.lb, self.ub = replacements.get(i)
            X_new[:, i][X_new[:, i] < self.lb] = self.lb
            X_new[:, i][X_new[:, i] > self.ub] = self.ub
        return X_new

    @staticmethod
    def _outlier_iqr(x, thresh=1.5, max_lim=1, min_lim=-1):
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        lb = q1 - (iqr * thresh)
        ub = q3 + (iqr + thresh)
#        outliers = np.where((x < lb) | (x > ub))
        return lb, ub
