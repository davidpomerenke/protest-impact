import altair as alt
import matplotlib.pyplot as plt
import pandas as pd

from src.cache import cache
from src.models.synthetic_control import synthetic_control
from src.models.time_series import (
    disambiguate_target,
    doubly_robust,
    propensity_weighting,
    regression,
)


def plot_impact_ts(
    results: pd.DataFrame,
    predictor: str,
    targets: str = "protest",
    ax: plt.Axes = None,
    ci: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot results from `model.time_series.apply_method` for multiple steps and targets, for a single predictor;
    especially for the causal effects of overall or specific protest occurrence.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    for target in disambiguate_target(targets):
        r = results[(results["target"] == target) & (results["predictor"] == predictor)]
        ax.plot(r["step"], r["coef"], label=target, linewidth=1.5)
        if ci and "ci_lower" in r.columns and "ci_upper" in r.columns:
            ax.fill_between(r["step"], r["ci_upper"], r["ci_lower"], alpha=0.2)
    s = results["step"]
    ax.set_xticks(range(s.min(), s.max() + 1, 2))
    ax.set_xlabel("Day of outcome")
    ax.set_ylabel("ATT estimate (#articles)")
    ax.axhline(0, color="black", linewidth=1)
    if s.min() < 0:
        ax.axvline(0, color="black", linewidth=1)
    # show grid
    ax.grid(axis="y", alpha=0.3)
    return ax


_methods = dict(
    regression=regression,
    synthetic_control=synthetic_control,
    # propensity_weighting=propensity_weighting,
    # doubly_robust=doubly_robust,
)


def plot_trends(
    cumulative: bool = False,
    lags=range(-1, 1),
    steps=range(0, 15),
    random_treatment_regional=None,
    random_treatment_global=None,
    n_jobs=4,
):
    fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharey=True, sharex=True)
    treatment = "occ_protest"
    targets = ["media_combined_protest", "media_combined_not_protest"]
    params = dict(
        target=targets,
        treatment=treatment,
        lags=lags,
        steps=steps,
        cumulative=cumulative,
        ignore_group=True,
        ignore_medium=True,
        positive_queries=False,
        random_treatment_regional=random_treatment_regional,
        random_treatment_global=random_treatment_global,
        add_features=["size", "ewm"],
        n_jobs=n_jobs,
    )
    fig_params = dict(predictor=treatment, targets=targets)
    for i, (mname, m) in enumerate(_methods.items()):
        ax = axes[i]
        plot_impact_ts(m(**params), ax=ax, **fig_params)
        ax.set_title(mname)
    handles, labels = ax.get_legend_handles_labels()
    labels = [
        "articles mentioning climate change AND protest",
        "articles mentioning climate change AND NOT protest",
    ]
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1))
    return fig, axes


@cache
def compute_groups(
    methods, step, random_treatment_regional=None, random_treatment_global=None
):
    targets = [
        "media_combined_protest",
        "media_combined_not_protest",
        "media_combined_framing",
        "media_combined_goal",
        "media_combined_subsidiary_goal",
    ]
    treatments = [
        # "occ_FFF",
        # "occ_FFFX",
        # "occ_ALG",
        # "occ_XR",
        # "occ_EG",
        # "occ_GP",
        "occ_protest",
    ]
    cumulative = True
    data = []
    for treatment in treatments:
        params = dict(
            target=targets,
            treatment=treatment,
            lags=range(-14, 1),
            steps=[step],
            cumulative=cumulative,
            ignore_group=treatment == "occ_protest",
            ignore_medium=True,
            positive_queries=False,
            random_treatment_regional=random_treatment_regional,
            random_treatment_global=random_treatment_global,
        )
        for mname, m in _methods.items():
            if mname not in methods:
                continue
            results = m(**params)
            results["method"] = mname
            results["treatment"] = treatment
            data.append(results)
    results = pd.concat(data)
    return results


def plot_groups(
    step, kind, random_treatment_regional=None, random_treatment_global=None
):
    match kind:
        case "groups":
            groups = [
                "occ_FFF",
                # "occ_FFFX",
                "occ_ALG",
                "occ_XR",
                "occ_EG",
                "occ_GP",
            ]
            methods = ["synthetic_control"]
        case "methods":
            groups = ["occ_protest"]
            methods = list(_methods.keys())
    results = compute_groups(
        methods,
        step,
        random_treatment_regional=random_treatment_regional,
        random_treatment_global=random_treatment_global,
    )
    results = results[results["treatment"].isin(groups)]
    results["target"] = results["target"].str.replace("media_combined_", "")
    x = alt.X(
        "target:N",
        title="",
        sort=["protest", "not_protest", "framing", "goal", "subsidiary_goal"],
    )
    yname = "ATT estimate (#articles)"
    bars = (
        alt.Chart()
        .mark_bar()
        .encode(
            x=x,
            y=alt.Y("coef:Q", title=yname, scale=alt.Scale(domain=[-10, 10])),
            color=alt.Color(
                "target:N",
                title="",
                sort=["protest", "not_protest", "framing", "goal", "subsidiary_goal"],
            ),
        )
        .properties(
            width=100,
            height=150,
        )
    )

    error_bars = (
        alt.Chart()
        .mark_errorbar()
        .encode(
            alt.Y("ci_lower:Q", title=yname),
            alt.Y2("ci_upper:Q", title=yname),
            x=x,
        )
    )
    match kind:
        case "groups":
            return alt.layer(bars, error_bars, data=results).facet(
                column=alt.Column("treatment:N", title="", sort=groups),
            )
        case "methods":
            return alt.layer(bars, error_bars, data=results).facet(
                column=alt.Column("method:N", title="", sort=list(_methods.keys())),
            )


plot_trends(
    cumulative=False,
    lags=range(-7, 1),
    steps=range(-15, 15),
    random_treatment_global=56,
)
