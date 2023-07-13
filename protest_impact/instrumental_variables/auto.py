# generated mostly with GPT4

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.pipeline import Pipeline


class AutoRandomForestRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, cv=5):
        self.cv = cv
        self.pipeline = Pipeline([("rf", RandomForestRegressor())])
        self.param_grid = {
            "rf__n_estimators": [10, 50, 100, 200],
            "rf__max_depth": [None, 10, 20, 30],
            "rf__min_samples_split": [2, 5, 10],
            "rf__min_samples_leaf": [1, 2, 4],
        }
        self.sh = HalvingGridSearchCV(
            self.pipeline,
            self.param_grid,
            scoring="neg_mean_squared_error",
            cv=self.cv,
            n_jobs=-1,
        )

    def fit(self, X, y):
        y = np.ravel(y)
        self.sh.fit(X, y)
        self.best_model_ = self.sh.best_estimator_
        return self

    def predict(self, X):
        return self.best_model_.predict(X)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.sqrt(mean_squared_error(y, y_pred))
