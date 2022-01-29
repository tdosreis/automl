import numpy as np
import pandas as pd


class RandomData():

    def __init__(self,
                 n_rows=1000,
                 n_cols=5,
                 n_classes=2,
                 n_outliers=100,
                 n_nulls=100,
                 replacer=np.nan):

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_classes = n_classes
        self.n_outliers = n_outliers
        self.n_nulls = n_nulls
        self.replacer = replacer

    def random_dataframe(self):
        index = self._create_index()
        cols = self._create_cols()
        X = self._create_matrix()
        X = X.T
        y = self._create_target()
        X = pd.DataFrame(X, columns=cols, index=index)
        y = pd.DataFrame(y, columns=['target'], index=index)
        df = pd.concat([X, y], axis=1)
        return df

    def generate_nulls(self, X):
        n_times = 0
        while n_times < self.n_nulls:
            i, j = self._random_coordinates(X)
            X.iloc[i, j] = self.replacer
            n_times += 1

    def generate_outliers(self, X):
        n_times = 0
        while n_times < self.n_outliers:
            i, j = self._random_coordinates(X)
            try:
                mean = np.nanmean(X.iloc[:, j])
                replacer = np.random.normal(loc=mean, scale=4.0)
                X.iloc[i, j] = replacer
            except TypeError:
                pass
            n_times += 1

    def _create_index(self):
        return np.arange(0, self.n_rows, 1)

    def _create_cols(self):
        cols = [f'col_{i}' for i in range(self.n_cols)]
        return cols

    def _create_matrix(self):
        X = (
            np.matrix(
                [np.random.normal(loc=0.0, scale=1.0, size=self.n_rows)
                 for i in range(self.n_cols)]
            )
        )
        return X

    def _create_target(self):
        y = np.random.randint(low=0,
                              high=self.n_classes,
                              size=self.n_rows)
        return y

    @staticmethod
    def _random_coordinates(self, X):
        n_rows, n_cols = X.shape
        random_coordinate = (
            np.random.randint(0, self.n_rows),
            np.random.randint(0, self.n_cols)
        )
        return random_coordinate
