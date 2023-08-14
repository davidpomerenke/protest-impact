import warnings
from functools import partial

import numpy as np
import pandas as pd
from dowhy import CausalModel
from dowhy.causal_estimator import CausalEstimator
from dowhy.causal_estimators.econml import Econml
from dowhy.causal_estimators.propensity_score_weighting_estimator import (
    PropensityScoreWeightingEstimator,
)
from econml.sklearn_extensions.linear_model import (
    StatsModelsLinearRegression,
    StatsModelsRLM,
    WeightedLassoCV,
)
from numba.core.errors import NumbaDeprecationWarning
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from src.cache import cache

# Suppress numba warnings from econml import, see https://github.com/py-why/EconML/issues/807


warnings.simplefilter("ignore", category=NumbaDeprecationWarning)

from econml.dr import DRLearner, ForestDRLearner, LinearDRLearner, SparseLinearDRLearner


def _dowhy_model_estimation(
    estimator: CausalEstimator,
    target: str,
    treatment: str,
    lagged_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Wraps propensity score dowhy estimators, see below.
    """
    treatment_ = treatment + "_lag0"
    effect_modifiers = [
        c
        for c in lagged_df.columns
        if c.startswith("occ_") and c.endswith("_lag0") and c != treatment_
    ]
    instruments = [c for c in lagged_df.columns if c.startswith("weather_")]
    common_causes = [
        c
        for c in lagged_df.columns
        if c not in ([target, treatment_] + effect_modifiers + instruments)
    ]
    model = CausalModel(
        data=lagged_df,
        treatment=treatment_,
        outcome=target,
        common_causes=common_causes,
        effect_modifiers=effect_modifiers,
    )
    estimand = model.identify_effect(method_name="maximal-adjustment")
    estimator = estimator(estimand)
    estimator.fit(lagged_df)
    estimate = estimator.estimate_effect(lagged_df, target_units="att")
    ci = estimate.get_confidence_intervals()
    if isinstance(ci, np.ndarray):
        ci = ci.flatten()
    return pd.DataFrame(
        dict(
            coef=[estimate.value],
            predictor=[treatment],
            ci_lower=[ci[0]],
            ci_upper=[ci[1]],
        )
    )


propensity_model = LogisticRegressionCV(
    solver="newton-cholesky",
    max_iter=1000
)


@cache
def _propensity_weighting(target: str, treatment: str, lagged_df: pd.DataFrame):
    """
    For use with models.time_series.apply_method.
    """
    estimator = partial(
        PropensityScoreWeightingEstimator,
        confidence_intervals=True,
        propensity_score_model=propensity_model,
    )
    return _dowhy_model_estimation(estimator, target, treatment, lagged_df)


@cache
def _doubly_robust(target: str, treatment: str, lagged_df: pd.DataFrame):
    """
    For use with models.time_series.apply_method.
    """
    estimator = partial(
        Econml,
        econml_estimator=LinearDRLearner(
            model_propensity=propensity_model,
            model_regression=StatsModelsLinearRegression(),
        ),
        confidence_intervals=True,
    )
    return _dowhy_model_estimation(estimator, target, treatment, lagged_df)
