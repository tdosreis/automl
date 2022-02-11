import pathlib
import numpy as np
from src.outlier_analysis import OutlierAnalysis

ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent.absolute())


class TestOutlierAnalysis():

    def test_lower_and_upper_bounds(self):
        np.random.seed(42)
        x = np.random.random(size=100)
        x[10] = 5
        x[20] = 10
        x = x.reshape(-1, 1)
        oa = OutlierAnalysis()
        oa.fit_transform(x)
        assert (
                (round(oa.lb, 5) == -0.64991) &
                (round(oa.ub, 5) == 2.82863)
                )
