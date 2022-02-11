import pathlib
import numpy as np
from src.automl import Model
from src.random_data import RandomData

ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent.absolute())


class TestAutoML():

    def test_training_pipeline(self):
        rd = RandomData(n_rows=1000,
                        n_cols=10,
                        n_classes=3,
                        n_outliers=None,
                        n_nulls=1000,
                        replacer=[-1, -2, np.inf],
                        categories=[['A', 'B'],
                                    ['C', 'D'],
                                    ['E', 'F']])

        df = rd.random_dataframe()
        X = df.iloc[:, :-1]
        y = df['target']

        num_vars = ['col_0', 'col_1', 'col_2', 'col_3', 'col_4',
                    'col_5', 'col_6', 'col_7', 'col_8', 'col_9']

        cat_vars = ['col_10', 'col_11', 'col_12']

        m = Model(cat_vars=cat_vars, num_vars=num_vars)

        model = m.train(X, y)

        prediction = model.predict_proba(X)

        assert prediction.shape == (1000, 3)
