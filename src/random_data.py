import numpy as np
import pandas as pd


class RandomData():

    def __init__(self,
                 n_rows=1000,
                 n_cols=5,
                 n_classes=2,
                 n_outliers=None,
                 n_nulls=None,
                 replacer=[np.nan],
                 categories=None):

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_classes = n_classes
        self.n_outliers = n_outliers
        self.n_nulls = n_nulls
        self.replacer = replacer
        self.categories = categories

    def random_dataframe(self):

        index = self._create_index()
        cols = self._create_cols()

        X = self._create_matrix()
        X = X.T
        y = self._create_target()
        X = pd.DataFrame(X, columns=cols, index=index)

        if self.n_nulls is not None:
            self._generate_nulls(X)

        if self.n_outliers is not None:
            self._generate_outliers(X)

        if self.categories is not None:
            X_categ = self._add_category()
            X = pd.concat([X, X_categ], axis=1)

        y = pd.DataFrame(y, columns=['target'], index=index)

        df = pd.concat([X, y], axis=1)

        return df

    def _add_category(self):

        self.n_categs = len(self.categories)

        columns = [f'col_{i}' for i in
                   range(self.n_cols, self.n_categs + self.n_cols)]

        X_categ = (
            pd.DataFrame(
                np.matrix(
                    [[self._random_index(category) for i in range(self.n_rows)]
                     for category in self.categories]).T
            )
        )

        X_categ = pd.DataFrame(X_categ)
        X_categ.columns = columns

        return X_categ

    def _generate_nulls(self, X):
        n_times = 0
        while n_times < self.n_nulls:
            i, j = self._random_coordinates(X, self.n_rows, self.n_cols)
            X.iloc[i, j] = self._random_index(self.replacer)
            n_times += 1

    def _generate_outliers(self, X):
        n_times = 0
        while n_times < self.n_outliers:
            i, j = self._random_coordinates(X, self.n_rows, self.n_cols)
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
    def _random_coordinates(X, n_rows, n_cols):
        n_rows, n_cols = X.shape
        random_coordinate = (
            np.random.randint(0, n_rows),
            np.random.randint(0, n_cols)
        )
        return random_coordinate

    @staticmethod
    def _random_index(tags):
        index = np.random.randint(0, len(tags))
        return tags[index]
