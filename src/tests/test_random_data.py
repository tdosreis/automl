import pathlib
import numpy as np
from src.random_data import RandomData

ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent.absolute())


class TestRandomData():

    def test_number_of_objects(self):

        rd = RandomData(n_rows=1000,
                        n_cols=10,
                        n_classes=3,
                        n_outliers=None,
                        n_nulls=1000,
                        replacer=[-1, -2],
                        categories=[['A', 'B'],
                                    ['C', 'D'],
                                    ['E', 'F']])

        df = rd.random_dataframe()

        assert np.sum(df.dtypes == 'object') == 3

    def test_number_of_classes(self):

        rd = RandomData(n_rows=1000,
                        n_cols=10,
                        n_classes=5,
                        n_outliers=None,
                        n_nulls=1000,
                        replacer=[-1, -2],
                        categories=[['A', 'B'],
                                    ['C', 'D'],
                                    ['E', 'F']])

        df = rd.random_dataframe()

        assert df['target'].nunique() == 5

    def test_shape_of_dataset(self):

        rd = RandomData(n_rows=1000,
                        n_cols=10,
                        n_classes=5,
                        n_outliers=1000,
                        n_nulls=1000,
                        replacer=[-1, -2],
                        categories=[['A', 'B'],
                                    ['C', 'D'],
                                    ['E', 'F']])

        df = rd.random_dataframe()

        assert df.shape == (1000, 14)
