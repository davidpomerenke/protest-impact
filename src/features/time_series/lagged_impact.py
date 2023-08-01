import matplotlib.pyplot as plt
import pandas as pd


def plot_lagged_impact(
    results: pd.DataFrame,
    predictor: str,
    targets: str = "protest",
    ax: plt.Axes = None,
    gap=0,
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
            & (results["lag"] == gap)
        ]
        ax.plot(r["step"] + gap, r["coef"], label=target, linewidth=1.5)
        if "ci_lower" in r.columns and "ci_upper" in r.columns:
            ax.fill_between(r["step"] + gap, r["ci_upper"], r["ci_lower"], alpha=0.2)
    # ax.set_xticks(range(-10, 11, 1))
    ax.set_xlabel("Shift (days)")
    ax.set_ylabel("Coefficient")
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)
    ax.legend()
    return ax
