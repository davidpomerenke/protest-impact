from typing import Iterable

import pandas as pd

from src.cache import cache
from src.features.aggregation import all_regions


@cache
def get_lagged_df(
    target: str,
    lags: Iterable[int],
    step: int = 0,
    cumulative: bool = False,
    include_instruments: bool = False,
    include_texts: bool = False,
    ignore_group: bool = False,
    ignore_medium: bool = False,
    positive_queries: bool = True,
    text_cutoff: int | None = None,
    region_dummies: bool = False,
    random_treatment: int | None = None,
):
    """
    Include time-series lags, that is, past values of the various variables.
    To be used independently or in combination with src/models/time_series.py.

    Parameters
    ----------
    target : str
        The target variable.
    lags : list[int]
        Range or list of lags. Use negative lags for past values, 0 for the day of the protest. Typical value: range(-6, 1) == [-6, -5, -4, -3, -2, -1, 0], that is, the past week including the day of the protest.
        The following things happen automatically:
          - media variables will not be lagged for the current day to avoid leakage
          - weekday and region dummies will not be lagged because that provides no additional information but introduces multicollinearity
    step : int, optional
        How many steps ahead should the target variable be predicted? Defaults to 0, that is, the day of the protest.
    cumulative : bool, optional
        When True, use cumulative values including the protest day to the specified step (including both ends). Only applies for the target variable.
    include_instruments : bool, optional
        Whether to include instrumental variables.
    include_texts : bool, optional
        Whether to include full texts.
    ignore_group : bool, optional
        When true, aggregates the protests of all groups ("occ_FFF", "occ_ALG", etc.) to a single variable "occ_protest".
    text_cutoff : int, optional
        Shortens the full texts. See there for details.
    region_dummies : bool, optional
        The lagged_df combines data from multiple regions. When region_dummies is True, the applicable region is encoded as a dummy variable (that is, 14 dummies for the 14 regions).
    random_treatment : bool, optional
        When True, the treatments are randomly assigned (sampling with replacement, per region), while all other variables remain the same. This is useful for placebo tests.
    """
    lagged_dfs = []
    for name, df in all_regions(
        include_instruments=include_instruments,
        include_texts=include_texts,
        ignore_group=ignore_group,
        positive_queries=positive_queries,
        ignore_medium=ignore_medium,
        text_cutoff=text_cutoff,
        region_dummies=region_dummies,
        random_treatment=random_treatment,
    ):
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
    lagged_df = pd.concat(lagged_dfs).sort_index().reset_index(drop=True)
    return lagged_df
