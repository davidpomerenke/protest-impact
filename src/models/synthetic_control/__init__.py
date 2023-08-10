from functools import partial
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib.parallel import Parallel, delayed
from scipy.optimize import fmin_slsqp
from sklearn.preprocessing import RobustScaler, StandardScaler
from tqdm.auto import tqdm

from src.cache import cache
from src.features.aggregation import all_regions
from src.paths import processed_data


def loss_w(W, X, y) -> float:
    # from https://matheusfacure.github.io/python-causality-handbook/15-Synthetic-Control.html
    return np.sqrt(np.mean((y - X.dot(W)) ** 2))


def get_w(X, y):
    # from https://matheusfacure.github.io/python-causality-handbook/15-Synthetic-Control.html
    w_start = [1 / X.shape[1]] * X.shape[1]
    weights = fmin_slsqp(
        partial(loss_w, X=X, y=y),
        np.array(w_start),
        f_eqcons=lambda x: np.sum(x) - 1,
        bounds=[(0.0, 1.0)] * len(w_start),
        disp=False,
    )
    return weights


@cache
def synthetic_control(
    region: str,
    date_: pd.Timestamp,
    rolling: int = 1,
    scale: Literal["demean", None] = "demean",
) -> tuple[pd.Series, pd.Series] | None:
    dfs = all_regions(ignore_group=True, protest_source="acled")
    df_w = [df for name, df in dfs if name == region][0]
    control_regions = [
        (name, df) for name, df in dfs if df[df.index == date_].iloc[0].occ_protest == 0
    ]
    if len(control_regions) == 0:
        print(f"No control regions for {region} on {date_}")
        return None
    idx_pre = (df_w.index >= date_ - pd.Timedelta(days=3 * 28)) & (df_w.index < date_)
    idx_post = (df_w.index >= date_) & (df_w.index < date_ + pd.Timedelta(days=28))
    idx_all = idx_pre | idx_post
    for name, df in dfs:
        df["media"] = (
            (df["media_print_protest"] + df["media_print_not_protest"])
            .rolling(rolling)
            .mean()
            # .diff(7)
        )
        if scale == "demean":
            df["media"] = df["media"] - df["media"].mean()
    df_w = df_w["media"]
    df_c = pd.concat(
        [df["media"] for name, df in control_regions],
        axis=1,
        keys=[name for name, df in control_regions],
    )
    y = df_w[idx_pre]
    df_c = sm.add_constant(df_c)
    X = df_c[idx_pre]
    weights = get_w(X, y)
    y_all = df_w[idx_all]
    y_c_all = pd.Series(df_c[idx_all].values.dot(weights), index=df_w[idx_all].index)
    return y_all, y_c_all


@cache(ignore=["n_jobs"])
def compute_synthetic_controls(
    rolling: int = 1, scale: str | None = "demean", n_jobs: int = 4
):
    dfs = all_regions(ignore_group=True, protest_source="acled")
    protest_dates = []
    for name, df in dfs:
        dates = df[df.occ_protest == 1].index
        for date_ in dates:
            protest_dates.append((name, date_))
    results = Parallel(n_jobs=n_jobs)(
        delayed(synthetic_control)(name, date_, rolling=rolling, scale=scale)
        for name, date_ in tqdm(protest_dates)
    )
    ys, y_cs = [], []
    for (name, date_), result in zip(protest_dates, results):
        if result is None:
            continue
        y, y_c = result
        y.index = (y.index - date_).days
        y_c.index = (y_c.index - date_).days
        y = y.reindex(range(-3 * 28, 28))
        y_c = y_c.reindex(range(-3 * 28, 28))
        ys.append(y)
        y_cs.append(y_c)
    # join on index
    ys = pd.concat(ys, axis=1, keys=[name for name, date_ in protest_dates])
    y_cs = pd.concat(y_cs, axis=1, keys=[name for name, date_ in protest_dates])
    return ys, y_cs


if __name__ == "__main__":
    ys, y_cs = compute_synthetic_controls(n_jobs=1)
    path = processed_data / "synthetic_control"
    path.mkdir(parents=True, exist_ok=True)
    ys.to_csv(path / "y.csv")
    y_cs.to_csv(path / "y_c.csv")
