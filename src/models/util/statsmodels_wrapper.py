import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin


class SMWrapper(BaseEstimator, RegressorMixin):
    """A universal sklearn-style wrapper for statsmodels regressors"""

    def __init__(self, model_class, fit_intercept=True, **fit_args):
        self.model_class = model_class
        self.fit_intercept = fit_intercept
        self.fit_args = fit_args

    def fit(self, X, y):
        if self.fit_intercept:
            X = sm.add_constant(X)
        self.model_ = self.model_class(y, X)
        self.results_ = self.model_.fit(**self.fit_args)
        first_param_idx = 1 if self.fit_intercept else 0
        self.coef_ = self.results_.params[first_param_idx:]
        self.intercept_ = self.results_.params[0]
        self.conf_int_ = self.results_.conf_int()[first_param_idx:]
        return self

    def predict(self, X):
        if self.fit_intercept:
            X = sm.add_constant(X)
        return self.results_.predict(X)


sk_ols = SMWrapper(sm.OLS, fit_args=dict(cov_type="HC3"))
