import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries

from src.cache import cache


def lagged_impact(
    y: pd.DataFrame, x: pd.DataFrame, evaluator: callable
) -> pd.DataFrame:
    dfs = []
    for shift in range(-10, 11):
        y_ = y.shift(shift).dropna()
        x_ = x.loc[y_.index]
        res = evaluator(y_, x_)
        res["shift"] = shift
        dfs.append(res)
    df = pd.concat(dfs).reset_index(drop=True)
    return df


def plot_lagged_impact(
    results: pd.DataFrame,
    predictor: str,
    targets: str = "protest",
    ax: plt.Axes = None,
) -> tuple[plt.Figure, plt.Axes]:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    match targets:
        case "protest":
            targets = ["media_protest", "media_not_protest"]
        case "goals":
            targets = ["media_goal", "media_subsidiary_goal", "media_framing"]
    for target in targets:
        r = results[
            (results["target"] == target)
            & (results["predictor"] == predictor)
            & (results["lag"] == 0)
        ]
        ax.plot(r["shift"], r["coef"], label=target, linewidth=1.5)
        if "lower" in r.columns and "upper" in r.columns:
            ax.fill_between(r["shift"], r["upper"], r["lower"], alpha=0.2)
    ax.set_xticks(range(-10, 11, 1))
    ax.set_xlabel("Shift (days)")
    ax.set_ylabel("Coefficient")
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)
    ax.legend()
    return ax
