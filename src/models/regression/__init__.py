from functools import partial
from itertools import product
from time import time
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from dowhy import CausalModel
from dowhy.causal_estimators.propensity_score_weighting_estimator import (
    PropensityScoreWeightingEstimator,
)
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from tqdm.auto import tqdm

from src.cache import cache
from src.features.aggregation import all_regions


@cache
def get_lagged_df(
    target: str,
    lags: list[int],
    step: int = 0,
    cumulative: bool = False,
    ignore_group: bool = False,
):
    lagged_dfs = []
    for df in all_regions():
        if df is None:
            continue
        if ignore_group:
            protest_cols = [c for c in df.columns if c.startswith("occ_")]
            protest = df[protest_cols].any(axis=1).astype(int)
            df = df.drop(columns=protest_cols).assign(occ_protest=protest)
        lagged_df = pd.concat(
            [df.shift(-lag).add_suffix(f"_lag{lag}") for lag in lags], axis=1
        )
        lagged_df = lagged_df[
            [
                c
                for c in lagged_df.columns
                if not (c.startswith("media_") and c.endswith("_lag0"))  # no leakage
                and not (
                    c.startswith("weekday_") and not c.endswith("_lag0")
                )  # no weekday lags
            ]
        ]
        y = df[[target]].shift(-step)
        if cumulative:
            y = y.rolling(step + 1).sum()
        df_combined = pd.concat([y, lagged_df], axis=1).dropna()
        lagged_dfs.append(df_combined)
    lagged_df = pd.concat(lagged_dfs).reset_index(drop=True)
    return lagged_df


def disambiguate_target(target: str | list[str] | Literal["protest", "goals", "all"]):
    protest_targets = [
        "media_online_protest",
        "media_online_not_protest",
        "media_print_protest",
        "media_print_not_protest",
    ]
    goal_targets = [
        "media_online_goal",
        "media_online_subsidiary_goal",
        "media_online_framing",
        "media_print_goal",
        "media_print_subsidiary_goal",
        "media_print_framing",
    ]
    match target:
        case [*ts]:
            return ts
        case "protest":
            return protest_targets
        case "goals":
            return goal_targets
        case "all":
            return protest_targets + goal_targets
        case _:
            return [target]


@cache
def _apply_method(
    method_name: str,
    target: str,
    lags: list[int],
    step: int,
    cumulative: bool = False,
    ignore_group: bool = False,
    **kwargs,
):
    method_dict = dict(
        regression=_regression,
        propensity_weighting=_propensity_weighting,
    )
    method = method_dict[method_name]
    lagged_df = get_lagged_df(target, lags, step, cumulative, ignore_group)
    model, coefs = method(target=target, lagged_df=lagged_df, **kwargs)
    coefs["step"] = step
    coefs["target"] = target
    return model, coefs


def apply_method(
    target: str | list[str] | Literal["protest", "goals", "all"],
    steps: int = 7,
    show_progress: bool = True,
    **kwargs,
):
    targets_and_steps = list(product(disambiguate_target(target), range(steps)))
    results = Parallel(n_jobs=8)(
        delayed(_apply_method)(target=target, step=step, **kwargs)
        for target, step in tqdm(targets_and_steps, disable=not show_progress)
    )
    models, params = zip(*results)
    params = pd.concat(params).reset_index(drop=True)
    return models, params


def decode_param_names(
    coefficients: list, feature_names: list[str], data_type: str
) -> pd.DataFrame:
    feature_names = [fn + "_lag0" if not "_lag" in fn else fn for fn in feature_names]
    if data_type == "conf_int":
        df = pd.DataFrame(
            {
                "feature_name": feature_names,
                "ci_lower": coefficients[0],
                "ci_upper": coefficients[1],
            }
        )
    elif data_type == "coef":
        df = pd.DataFrame({"feature_name": feature_names, "coef": coefficients})
    name_parts = df["feature_name"].str.rsplit("_lag", n=1)
    df["predictor"] = name_parts.str[0]
    df["lag"] = name_parts.str[1].astype(int)
    df = df.drop(columns=["feature_name"])
    return df


@cache
def _regression(target: str, lagged_df: pd.DataFrame):
    y = lagged_df[[target]]
    X = lagged_df.drop(columns=[target])
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit(cov_type="HC3")
    coefs = decode_param_names(results.params, X.columns, data_type="coef")
    conf_int = decode_param_names(results.conf_int(), X.columns, data_type="conf_int")
    coefs = coefs.merge(conf_int, on=["predictor", "lag"])
    return model, coefs


regression = partial(apply_method, method_name="regression")


# @cache
def _propensity_weighting(target: str, treatment: str, lagged_df: pd.DataFrame):
    treatment_ = treatment + "_lag0"
    effect_modifiers = [
        c
        for c in lagged_df.columns
        if c.startswith("occ_") and c.endswith("_lag0") and c != treatment_
    ]
    common_causes = [
        c for c in lagged_df.columns if c not in [target, treatment_, effect_modifiers]
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


propensity_weighting = partial(apply_method, method_name="propensity_weighting")


# lags = list(range(-28 * 6, 1))
# steps = 1
# treatment = "occ_protest"
# # target = "protest"
# target = "media_online_protest"
# cumulative = False
# ignore_group = True

# models, results = propensity_weighting(
#     target=target,
#     treatment=treatment,
#     lags=lags,
#     steps=steps,
#     cumulative=cumulative,
#     ignore_group=True,
# )
# print(results)

# treatment_ = treatment + "_lag0"
# estimator = models[(target, 0)]
# lagged_df = get_lagged_df(target, lags, 0, cumulative, ignore_group)
# # determine f1 score
# estimator.estimate_propensity_score_column(lagged_df)
# treatment_pred = (lagged_df["propensity_score"] > 0.5).astype(int)
# treatment_true = lagged_df[treatment_]
# print(classification_report(treatment_true, treatment_pred))
