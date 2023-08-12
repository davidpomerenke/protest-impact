import matplotlib.pyplot as plt
import pandas as pd

from src.models.time_series import disambiguate_target


def plot_impact_ts(
    results: pd.DataFrame,
    predictor: str,
    targets: str = "protest",
    ax: plt.Axes = None,
    ci: bool = True,
    method_name: str = "magic",
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot results from `model.time_series.apply_method` for multiple steps and targets, for a single predictor;
    especially for the causal effects of overall or specific protest occurrence.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    for target in disambiguate_target(targets):
        r = results[
            (results["target"] == target)
            & (results["predictor"] == predictor)
            & (results["lag"] == 0)
        ]
        ax.plot(r["step"], r["coef"], label=target, linewidth=1.5)
        if ci and "ci_lower" in r.columns and "ci_upper" in r.columns:
            ax.fill_between(r["step"], r["ci_upper"], r["ci_lower"], alpha=0.2)
    s = results["step"]
    ax.set_xticks(range(s.min(), s.max() + 1))
    ax.set_xlabel("Day of outcome")
    ax.set_ylabel("Estimate")
    ax.set_title(
        f"Causal effect of {predictor} on {targets} using {method_name}"
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)
    ax.legend()
    return ax
