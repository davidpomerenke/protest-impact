from functools import partial
from itertools import product
from typing import Iterable, Literal

import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from src.cache import cache
from src.features.time_series import get_lagged_df
from src.models.instrumental_variable import (
    _instrumental_variable,
    _instrumental_variable_liml,
)
from src.models.propensity_scores import _doubly_robust, _propensity_weighting
from src.models.regression import _regression


@cache
def apply_method(
    target: str | list[str] | Literal["protest", "goals", "all"],
    steps: Iterable[int] = range(7),
    n_jobs=4,
    show_progress: bool = True,
    **kwargs,
):
    """
    Runs a causal method for multiple steps and multiple targets in parallel and collects the results.
    Results can be plotted with `visualization.visualize.plot_impact_ts`.

    Parameters
    ----------
    target : str or list[str] or "protest" or "goals" or "all"
        The target variable(s), shortcuts permitted, see `disambiguate_target`.
    steps : Iterable[int], optional
        For how many days (typically into the future) the impact should be estimated. `range(7)` would give one week of estimates including the day of the protest. Negative values can be used for placebo tests.
    """
    targets_and_steps = list(product(disambiguate_target(target), steps))
    results = Parallel(n_jobs=n_jobs)(
        delayed(_apply_method)(target=target, step=step, **kwargs)
        for target, step in tqdm(targets_and_steps, disable=not show_progress)
    )
    return pd.concat(results).reset_index(drop=True)


def _apply_method(
    method_name: str,
    target: str,
    lags: list[int],
    step: int,
    cumulative: bool = False,
    ignore_group: bool = False,
    ignore_medium: bool = False,
    positive_queries: bool = True,
    region_dummies: bool = False,
    random_treatment_regional: int | None = None,
    random_treatment_global: int | None = None,
    **kwargs,
):
    """
    Runs a single step within `apply_method`.
    Retrieves data and feeds it to the actual causal method, and collects the results.
    The causal method is identified by string,
    because otherwise the caching of partial functions would not be possible.
    """
    method_dict = dict(
        regression=_regression,
        propensity_weighting=_propensity_weighting,
        doubly_robust=_doubly_robust,
        instrumental_variable=_instrumental_variable,
        instrumental_variable_liml=_instrumental_variable_liml,
    )
    method = method_dict[method_name]
    instr = method_name in ["instrumental_variable", "instrumental_variable_liml"]
    lagged_df = get_lagged_df(
        target=target,
        lags=lags,
        step=step,
        cumulative=cumulative,
        ignore_group=ignore_group,
        ignore_medium=ignore_medium,
        positive_queries=positive_queries,
        region_dummies=region_dummies,
        include_instruments=instr,
        random_treatment_regional=random_treatment_regional,
        random_treatment_global=random_treatment_global,
    )
    coefs = method(target=target, lagged_df=lagged_df, **kwargs)
    coefs["step"] = step
    coefs["target"] = target
    return coefs


regression = partial(apply_method, method_name="regression")
instrumental_variable = partial(apply_method, method_name="instrumental_variable")
instrumental_variable_liml = partial(
    apply_method, method_name="instrumental_variable_liml"
)
propensity_weighting = partial(apply_method, method_name="propensity_weighting")
doubly_robust = partial(apply_method, method_name="doubly_robust")


def disambiguate_target(target: str | list[str] | Literal["protest", "goals", "all"]):
    """
    Just a helper that allows more concise definition of desired target variables.
    """
    general_targets = [
        "media_online_all",
        "media_print_all",
    ]
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
        case "general":
            return general_targets
        case "protest":
            return protest_targets
        case "goals":
            return goal_targets
        case "all":
            return protest_targets + goal_targets + general_targets
        case _:
            return [target]
