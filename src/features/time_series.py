from typing import Iterable

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.cache import cache
from src.features.aggregation import all_regions


@cache
def get_lagged_df(
    target: str,
    lags: Iterable[int],
    step: int = 0,
    cumulative: bool = False,
    instruments: str | None = None,
    include_texts: bool = False,
    ignore_group: bool = False,
    ignore_medium: bool = False,
    positive_queries: bool = True,
    text_cutoff: int | None = None,
    region_dummies: bool = False,
    random_treatment_regional: int | None = None,
    random_treatment_global: int | None = None,
    add_features: list[str] | None = None,
    return_loadings: bool = False,
    shift_instruments: bool = False,
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
    instruments : bool, optional
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
        instruments=instruments,
        include_texts=include_texts,
        ignore_group=ignore_group,
        positive_queries=positive_queries,
        ignore_medium=ignore_medium,
        text_cutoff=text_cutoff,
        region_dummies=region_dummies,
        random_treatment_regional=random_treatment_regional,
        add_features=add_features,
        instrument_shift=step if shift_instruments else 0,
    ):
        lagged_df = pd.concat(
            [df.shift(-lag).add_suffix(f"_lag{lag}") for lag in lags], axis=1
        )
        lagged_df = lagged_df[
            [
                c
                for c in lagged_df.columns
                # no leakage of outcome:
                if not (c.startswith("media_") and c.endswith("_lag0"))
                # no leakage of protest sizes:
                if not (c.startswith("size_") and c.endswith("_lag0"))
                # no weekday lags:
                and not (c.startswith("weekday_") and not c.endswith("_lag0"))
                # no region lags:
                and not (c.startswith("region_") and not c.endswith("_lag0"))
                # no instrument lags:
                and not (
                    (c.startswith("weather_") or c.startswith("covid_"))
                    and not c.endswith("_lag0")
                )
                # only lag -1 for moving average:
                and not ("_ewm" in c and not c.endswith("_lag-1"))
            ]
        ]
        y = df[[target]].shift(-step if not shift_instruments else 0)
        if cumulative:
            y = y.rolling(step + 1).sum()
        df_combined = pd.concat([y, lagged_df], axis=1).dropna()
        lagged_dfs.append(df_combined)
    lagged_df = pd.concat(lagged_dfs).sort_index().reset_index(drop=True)
    if random_treatment_global is not None:
        w_cols = [c for c in lagged_df.columns if c.startswith("occ_")]
        for i, col in enumerate(w_cols):
            lagged_df[col] = (
                lagged_df[col]
                .sample(frac=1, replace=True, random_state=random_treatment_global + i)
                .values
            )
    if instruments and "pc" in instruments:
        instruments_ = []
        if "weather" in instruments:
            instruments_ += [c for c in lagged_df.columns if c.startswith("weather_")]
        if "covid" in instruments:
            instruments_ += [c for c in lagged_df.columns if c.startswith("covid_")]
        loadings_dfs = []
        for kind in ["seasonal_", "resid_", ""]:
            instruments__ = [c for c in instruments_ if kind in c]
            if len(instruments__) == 0 or not set(instruments__).issubset(
                lagged_df.columns
            ):
                continue
            df_instr = StandardScaler().fit_transform(lagged_df[instruments__])
            pc = PCA()
            pcr = pc.fit_transform(df_instr)
            pc_instruments = [f"pc_{kind}{i}" for i in range(pcr.shape[1])]
            loadings = pc.components_
            loadings_df = pd.DataFrame(
                loadings,
                columns=instruments__,
                index=pc_instruments,
            )
            loadings_dfs.append(loadings_df)
            lagged_df = pd.concat(
                [
                    lagged_df.drop(columns=instruments__),
                    pd.DataFrame(pcr, columns=pc_instruments),
                ],
                axis=1,
            )
        if return_loadings:
            return loadings_dfs
    return lagged_df
