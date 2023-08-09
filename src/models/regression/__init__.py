from functools import partial
from itertools import product
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from dowhy import CausalModel
from dowhy.causal_estimators.instrumental_variable_estimator import (
    InstrumentalVariableEstimator,
)
from dowhy.causal_estimators.propensity_score_weighting_estimator import (
    PropensityScoreWeightingEstimator,
)
from joblib import Parallel, delayed
from linearmodels.iv import IV2SLS, IVGMM, IVGMMCUE, IVLIML
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
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
    region_dummies: bool = False,
):
    lagged_dfs = []
    for df in all_regions(region_dummies=region_dummies):
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
                # no leakage:
                if not (c.startswith("media_") and c.endswith("_lag0"))
                # no weekday lags:
                and not (c.startswith("weekday_") and not c.endswith("_lag0"))
                # no region lags:
                and not (c.startswith("region_") and not c.endswith("_lag0"))
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


def _apply_method(
    method_name: str,
    target: str,
    lags: list[int],
    step: int,
    cumulative: bool = False,
    ignore_group: bool = False,
    region_dummies: bool = False,
    **kwargs,
):
    method_dict = dict(
        regression=_regression,
        propensity_weighting=_propensity_weighting,
        instrumental_variable=_instrumental_variable,
        instrumental_variable_liml=_instrumental_variable_liml,
    )
    method = method_dict[method_name]
    lagged_df = get_lagged_df(
        target, lags, step, cumulative, ignore_group, region_dummies=region_dummies
    )
    model, coefs = method(target=target, lagged_df=lagged_df, **kwargs)
    coefs["step"] = step
    coefs["target"] = target
    return model, coefs


def apply_method(
    target: str | list[str] | Literal["protest", "goals", "all"],
    steps: int = 7,
    n_jobs=8,
    show_progress: bool = True,
    **kwargs,
):
    targets_and_steps = list(product(disambiguate_target(target), range(steps)))
    results = Parallel(n_jobs=n_jobs)(
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
    instruments = [c for c in lagged_df.columns if c.startswith("weather_")]
    y = lagged_df[[target]]
    X = lagged_df.drop(columns=[target] + instruments)
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit(cov_type="HC3")
    coefs = decode_param_names(results.params, X.columns, data_type="coef")
    conf_int = decode_param_names(results.conf_int(), X.columns, data_type="conf_int")
    coefs = coefs.merge(conf_int, on=["predictor", "lag"])
    return model, coefs


regression = partial(apply_method, method_name="regression")


@cache
def _propensity_weighting(target: str, treatment: str, lagged_df: pd.DataFrame):
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


propensity_weighting = partial(apply_method, method_name="propensity_weighting")


@cache
def _instrumental_variable(
    target: str, treatment: str, instrument: str, lagged_df: pd.DataFrame
):
    assert (
        treatment == "occ_protest"
    ), "general iv cannot control for other protest groups"
    assert instrument == "weather_prcp"
    treatment_ = treatment + "_lag0"
    instrument_ = instrument + "_lag0"
    lagged_df[instrument_] = lagged_df[instrument_] > 0
    model = CausalModel(
        data=lagged_df,
        treatment=treatment_,
        outcome=target,
        instruments=[instrument_],
    )
    estimand = model.identify_effect(method_name="maximal-adjustment")
    estimator = InstrumentalVariableEstimator(
        estimand,
        iv_instrument_name=instrument_,
        confidence_intervals=True,
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


instrumental_variable = partial(apply_method, method_name="instrumental_variable")


# @cache
def _instrumental_variable_liml(
    target: str, treatment: str, instrument: str, lagged_df: pd.DataFrame
):
    assert (
        treatment == "occ_protest"
    ), "general iv cannot control for other protest groups"
    assert isinstance(treatment, str)
    assert isinstance(instrument, str)
    treatment_ = treatment + "_lag0"
    instruments = [c for c in lagged_df.columns if c.startswith("weather_")]
    instrument = instrument + "_lag0"
    # confounders = [
    #     c for c in lagged_df.columns if not c in [target, treatment_] + instruments
    # ]
    media_online_protest = (
        lagged_df[
            [c for c in lagged_df.columns if c.startswith("media_online_protest")]
        ]
        .mean(axis=1)
        .rename("media_online_protest")
    )
    media_online_not_protest = (
        lagged_df[
            [c for c in lagged_df.columns if c.startswith("media_online_not_protest")]
        ]
        .mean(axis=1)
        .rename("media_online_not_protest")
    )
    media_print_protest = (
        lagged_df[[c for c in lagged_df.columns if c.startswith("media_print_protest")]]
        .mean(axis=1)
        .rename("media_print_protest")
    )
    media_print_not_protest = (
        lagged_df[
            [c for c in lagged_df.columns if c.startswith("media_print_not_protest")]
        ]
        .mean(axis=1)
        .rename("media_print_not_protest")
    )
    regions = lagged_df[[c for c in lagged_df.columns if c.startswith("region_")]]
    weather = (
        lagged_df[
            [
                c
                for c in lagged_df.columns
                if c.startswith("weather_prcp") and int(c.split("_lag")[1]) <= -3
            ]
        ]
        .mean(axis=1)
        .rename("previous_weather_prcp")
    )
    weekdays = lagged_df[[c for c in lagged_df.columns if c.startswith("weekday_")]]
    confounders = pd.concat(
        [
            media_online_protest,
            media_online_not_protest,
            media_print_protest,
            media_print_not_protest,
            regions,
            weather,
            # weekdays,
        ],
        axis=1,
    )
    confounders = sm.add_constant(confounders)
    assert instrument == "weather_prcp_lag0"
    lagged_df[instrument] = (lagged_df[instrument] > 0).astype(int)
    model = IVLIML(
        dependent=lagged_df[target],
        exog=confounders,
        endog=lagged_df[treatment_],
        instruments=lagged_df[instrument],
    )
    results = model.fit()
    # print(results)
    ci = results.conf_int()
    coefs = pd.DataFrame(
        dict(
            coef=[results.params[0]],
            predictor=[treatment],
            ci_lower=ci["lower"][0],
            ci_upper=ci["upper"][0],
            lag=[0],
        )
    )
    return model, coefs


instrumental_variable_liml = partial(
    apply_method, method_name="instrumental_variable_liml"
)

if __name__ == "__main__":
    lags = list(range(-7, 1))
    steps = 1
    treatment = "occ_protest"
    target = "media_online_protest"
    cumulative = False
    ignore_group = True

    models, results = instrumental_variable_liml(
        target=target,
        treatment=treatment,
        instrument="weather_prcp",
        lags=lags,
        steps=steps,
        cumulative=cumulative,
        ignore_group=True,
        region_dummies=True,
        n_jobs=1,
    )
    print(results)

    # treatment_ = treatment + "_lag0"
    # estimator = models[(target, 0)]
    # lagged_df = get_lagged_df(target, lags, 0, cumulative, ignore_group)
    # # determine f1 score
    # estimator.estimate_propensity_score_column(lagged_df)
    # treatment_pred = (lagged_df["propensity_score"] > 0.5).astype(int)
    # treatment_true = lagged_df[treatment_]
    # print(classification_report(treatment_true, treatment_pred))
