import matplotlib.pyplot as plt
import pandas as pd


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
            targets = [
                "media_online_protest",
                "media_online_not_protest",
                "media_print_protest",
                "media_print_not_protest",
            ]
        case "goals":
            targets = [
                "media_online_goal",
                "media_online_subsidiary_goal",
                "media_online_framing",
                "media_print_goal",
                "media_print_subsidiary_goal",
                "media_print_framing",
            ]
    for target in targets:
        r = results[
            (results["target"] == target)
            & (results["predictor"] == predictor)
            & (results["lag"] == 0)
        ]
        ax.plot(r["step"], r["coef"], label=target, linewidth=1.5)
        if "ci_lower" in r.columns and "ci_upper" in r.columns:
            ax.fill_between(r["step"], r["ci_upper"], r["ci_lower"], alpha=0.2)
    # ax.set_xticks(range(-10, 11, 1))
    ax.set_xlabel("Shift (days)")
    ax.set_ylabel("Coefficient")
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)
    ax.legend()
    return ax
