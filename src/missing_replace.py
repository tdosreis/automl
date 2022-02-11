import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class MissingReplace(BaseEstimator, TransformerMixin):
    """Replace some values in the features with NaN. This is usefull
    when dealing with labeled missing values such as -99999, -1, ...

    Parameters
    ----------
    missing_values: array, optional (default=[nan])
        List of values to be replaced.

    fill_nan_value: float, optional (default = nan)
        Value to be inserted.
    """
    def __init__(self, missing_values=[np.nan], fill_nan_value=np.nan):
        self.missing_values = missing_values
        self.fill_nan_value = fill_nan_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return (X.replace(self.missing_values,
                          self.fill_nan_value))


class InfinityReplace(BaseEstimator, TransformerMixin):
    """Replaces infinities with desired values.

    Parameters
    ----------
    max_infty: float, optional (default=10e9)
        Maximum value to replace +infty

    min_infty: float, optional (default=-10e9)
        Minimum value to replace -infty
    """

    def __init__(self, min_infty=-10e9, max_infty=10e9):
        self.min_infty = min_infty
        self.max_infty = max_infty

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return (X.replace([-np.inf, np.inf],
                          [self.min_infty, self.max_infty]))


class DropLowVariance(BaseEstimator, TransformerMixin):
    """Removes columns with low variance.

    Parameters
    ----------
    threshold: float, optional (default=0.5)
        Absolute percentage of nulls for the column
        to be dropped.
    """
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.data_shape = X.shape
        self.nulls_percentage = (
            {variable: nulls/self.data_shape[0]
             for variable, nulls in X.isnull().sum().to_dict().items()})
        return self

    def transform(self, X):
        self.labels = ([variable for variable, nulls in
                        self.nulls_percentage.items() if
                        nulls > self.threshold])

        return X.drop(self.labels, axis=1)
