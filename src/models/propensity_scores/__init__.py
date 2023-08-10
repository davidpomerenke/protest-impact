import pandas as pd
from dowhy import CausalModel
from dowhy.causal_estimators.propensity_score_weighting_estimator import (
    PropensityScoreWeightingEstimator,
)
from sklearn.linear_model import LogisticRegression

from src.cache import cache


@cache
def _propensity_weighting(target: str, treatment: str, lagged_df: pd.DataFrame):
    """
    For use with src/models/time_series.py.
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
    estimator = PropensityScoreWeightingEstimator(
        estimand,
        confidence_intervals=True,
        propensity_score_model=LogisticRegression(),
    )
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
