import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from protest_impact.time_series import LagDict, TimeSeriesRegressor
from protest_impact.util.statsmodels import SMLinearRegression


def shift_df(df: pd.DataFrame, y_cols: list[str], shift: int) -> pd.DataFrame:
    df_ = df.copy()
    for col in y_cols:
        df_[col] = df_[col].shift(shift)
    df_ = df_.dropna()
    return df_


def lagged_impact(
    dfs: list[pd.DataFrame],
    y_cols: list[str],
    lags: LagDict,
    regressor=SMLinearRegression(),
) -> pd.DataFrame:
    models = dict()
    results = []
    for shift in range(-10, 11):
        dfs_ = [shift_df(df, y_cols, shift) for df in dfs]
        model = TimeSeriesRegressor(regressor, y_cols=y_cols, lags=lags)
        model.fit_multiple(dfs_, static_covariates="dummies")
        res = model.get_coefficients()
        res["shift"] = shift
        cols = ["shift"] + list(res.columns[:-1])
        res = res[cols].sort_values(by=cols)
        results.append(res)
        models[shift] = model
    return models, pd.concat(results)


def plot_lagged_impact(
    results: pd.DataFrame,
    predictor: str,
    targets: str = "protest",
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8, 4))
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
    ax.set_title("Naive regression coefficients")
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    fig.tight_layout()
    return fig, ax
