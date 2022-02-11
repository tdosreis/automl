from outlier_analysis import OutlierAnalysis
from missing_replace import MissingReplace, InfinityReplace, DropLowVariance
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

params = {'classifier__C': [0.001, 0.01, 0.1, 1.0, 10.0],
          'classifier__max_iter': [100, 500, 1000],
          'classifier__penalty': ['l1', 'l2', 'ridge', 'elasticnet'],
          'classifier__solver': ['saga']}


class Model():
    """This is a simple yet useful implementation of a
    wrapped training pipeline for any given dataset.

    Note: code is under development.
    """

    def __init__(self,
                 cat_vars=None,
                 num_vars=None,
                 test_size=0.3,
                 random_state=101,
                 cv=5,
                 missing_tags=[-1, -2],
                 missing_imputer='median',
                 outlier_thresh=3.0):

        self.cat_vars = cat_vars
        self.num_vars = num_vars
        self.missing_tags = missing_tags
        self.missing_imputer = missing_imputer
        self.outlier_thresh = outlier_thresh
        self.test_size = test_size
        self.random_state = random_state
        self.cv = cv

    def train(self, X, y):
        X_train, X_test, y_train, y_test = (
            train_test_split(X, y,
                             test_size=self.test_size,
                             random_state=self.random_state))

        grid = self._set_training_grid()

        model = GridSearchCV(grid,
                             param_grid=params,
                             cv=self.cv,
                             scoring='roc_auc')

        model.fit(X_train, y_train)

        return model

    def _set_training_grid(self):

        missing_replace_pipeline = Pipeline([
            ("infty_replace", InfinityReplace()),
            ("drop_low_variance", DropLowVariance()),
            ("missing_replace", MissingReplace(
                missing_values=self.missing_tags)),
            ("simple_imputer", SimpleImputer(strategy=self.missing_imputer)),
            ("outlier_removal", OutlierAnalysis(thresh=self.outlier_thresh)),
            ("scaler", StandardScaler())
        ])

        category_replace_pipeline = Pipeline([
            ('simple_imputer', SimpleImputer(strategy='most_frequent')),
            ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        transformer = ColumnTransformer([
            ('missing_replace_pipeline',
                missing_replace_pipeline, self.num_vars),
            ('categorical', category_replace_pipeline, self.cat_vars)
        ])

        grid = Pipeline(steps=[('preprocessor', transformer),
                               ('classifier', LogisticRegression())
                               ])

        return grid
