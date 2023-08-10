import pandas as pd

from src.cache import cache
from src.features.aggregation import all_regions


@cache
def get_lagged_df(
    target: str,
    lags: list[int],
    step: int = 0,
    cumulative: bool = False,
    include_instruments: bool = False,
    ignore_group: bool = False,
    region_dummies: bool = False,
):
    """
    Include time-series lags, that is, past values of the various variables.
    To be used independently or in combination with src/models/time_series.py.
    """
    lagged_dfs = []
    for name, df in all_regions(
        include_instruments=include_instruments,
        ignore_group=ignore_group,
        region_dummies=region_dummies,
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
