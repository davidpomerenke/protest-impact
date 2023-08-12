from functools import partial

import pandas as pd
from dowhy import CausalModel
from dowhy.causal_estimator import CausalEstimator
from dowhy.causal_estimators.econml import Econml
from dowhy.causal_estimators.propensity_score_weighting_estimator import (
    PropensityScoreWeightingEstimator,
)
from econml.dr import LinearDRLearner
from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression
from sklearn.linear_model import LogisticRegressionCV

from src.cache import cache


@cache
def _dowhy_model_estimation(
    estimator: CausalEstimator,
    target: str,
    treatment: str,
    lagged_df: pd.DataFrame,
) -> tuple[list, pd.DataFrame]:
    """
    Wraps propensity score dowhy estimators, see below.
    For use with models.time_series.apply_method.
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
    coefs = pd.DataFrame(
        dict(
            coef=[estimate.value],
            predictor=[treatment],
            ci_lower=ci[0],
            ci_upper=ci[1],
            lag=[0],
        )
    )
    return estimator, coefs


propensity_model = LogisticRegressionCV(
    solver="saga", max_iter=1000, class_weight="balanced"
)

_propensity_weighting = partial(
    _dowhy_model_estimation,
    estimator=partial(
        PropensityScoreWeightingEstimator,
        confidence_intervals=True,
        propensity_score_model=propensity_model,
    ),
)

_doubly_robust = partial(
    _dowhy_model_estimation,
    estimator=partial(
        Econml,
        econml_estimator=partial(
            LinearDRLearner,
            model_propensity=propensity_model,
            model_regression=StatsModelsLinearRegression(),
        ),
        confidence_intervals=True,
    ),
)
