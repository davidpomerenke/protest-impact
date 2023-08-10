from functools import partial
from itertools import product
from typing import Literal

import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from src.features.time_series import get_lagged_df
from src.models.instrumental_variable import (
    _instrumental_variable,
    _instrumental_variable_liml,
)
from src.models.propensity_scores import _propensity_weighting
from src.models.regression import _regression


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


instrumental_variable_liml = partial(
    apply_method, method_name="instrumental_variable_liml"
)
instrumental_variable = partial(apply_method, method_name="instrumental_variable")
propensity_weighting = partial(apply_method, method_name="propensity_weighting")
regression = partial(apply_method, method_name="regression")
