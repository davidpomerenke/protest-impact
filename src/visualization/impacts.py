from functools import partial

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd

from src.cache import cache
from src.models.synthetic_control import synthetic_control
from src.models.time_series import (
    disambiguate_target,
    doubly_robust,
    instrumental_variable_liml,
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
    for i, target in enumerate(disambiguate_target(targets)):
        r = results[(results["target"] == target) & (results["predictor"] == predictor)]
        ax.plot(
            r["step"], r["coef"], label=target, linewidth=1.5, marker=".", color=f"C{i}"
        )
        if ci and "ci_lower" in r.columns and "ci_upper" in r.columns:
            ax.fill_between(
                r["step"], r["ci_upper"], r["ci_lower"], alpha=0.2, color=f"C{i}"
            )
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


_correlation = partial(regression, lags=[0], no_controls=True)

_regression = partial(regression, lags=[-1, 0], add_features=["diff"])

_synthetic_control = partial(synthetic_control, lags=range(-180, 1))

_instrumental_variable_liml = partial(
    instrumental_variable_liml,
    instruments="pc_weather_covid_season",
    iv_instruments=["pc_resid_9"],
    lags=range(-1, 1),
    add_features=["diff"],
)

_propensity_weighting = partial(
    propensity_weighting,
    lags=range(-4, 1),
    add_features=["diff", "size", "ewm"],
    standardize=True,
)

_doubly_robust = partial(
    doubly_robust,
    lags=range(-4, 1),
    add_features=["diff", "size", "ewm"],
    standardize=True,
)

_methods = dict(
    correlation=_correlation,
    propensity_weighting=_propensity_weighting,
    regression=_regression,
    synthetic_control=_synthetic_control,
    instrumental_variable_liml=_instrumental_variable_liml,
    # doubly_robust=_doubly_robust,
)


def plot_trends(
    cumulative: bool = False,
    steps=range(0, 15),
    random_treatment_regional=None,
    random_treatment_global=None,
    protest_source="acled",
    n_jobs=4,
    show_progress=True,
):
    fig, axes = plt.subplots(1, 5, figsize=(15, 4), sharey=False, sharex=True)
    treatment = "occ_protest"
    targets = ["media_combined_protest", "media_combined_not_protest"]
    params = dict(
        target=targets,
        treatment=treatment,
        steps=steps,
        cumulative=cumulative,
        ignore_group=True,
        ignore_medium=True,
        positive_queries=False,
        random_treatment_regional=random_treatment_regional,
        random_treatment_global=random_treatment_global,
        add_features=["size", "ewm"],
        n_jobs=n_jobs,
        show_progress=show_progress,
        protest_source=protest_source,
    )
    fig_params = dict(predictor=treatment, targets=targets)
    for i, (mname, m) in enumerate(_methods.items()):
        if show_progress:
            print(mname)
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
    methods,
    step,
    random_treatment_regional=None,
    random_treatment_global=None,
    treatments="occ_protest",
    protest_sources=["acled"],
):
    targets = [
        "media_combined_protest",
        "media_combined_not_protest",
        "media_combined_framing",
        "media_combined_goal",
        "media_combined_subsidiary_goal",
    ]
    cumulative = True
    data = []
    for treatment in treatments:
        for protest_source in protest_sources:
            params = dict(
                target=targets,
                treatment=treatment,
                steps=[step],
                cumulative=cumulative,
                ignore_group=treatment == "occ_protest",
                ignore_medium=True,
                positive_queries=False,
                random_treatment_regional=random_treatment_regional,
                random_treatment_global=random_treatment_global,
                protest_source=protest_source,
            )
            for mname, m in _methods.items():
                if mname not in methods:
                    continue
                results = m(**params)
                results["method"] = mname
                results["treatment"] = treatment
                results["source"] = protest_source
                data.append(results)
    results = pd.concat(data)
    return results


def plot_groups(
    step,
    kind,
    random_treatment_regional=None,
    random_treatment_global=None,
    protest_source="acled",
):
    match kind:
        case "groups":
            groups = [
                "occ_FFF",
                # "occ_FFFX",
                *(["occ_ALG"] if protest_source != "gpreg" else []),
                "occ_XR",
                *(["occ_EG"] if protest_source != "gpreg" else []),
                "occ_GP",
                # "occ_OTHER_CLIMATE_ORG",
            ]
            methods = ["synthetic_control"]
            sources = ["acled"]
        case "methods":
            groups = ["occ_protest"]
            methods = list(_methods.keys())
            sources = ["acled"]
        case "sources":
            groups = ["occ_protest"]
            methods = ["synthetic_control"]
            sources = ["acled", "gpreg", "gprep"]
    results = compute_groups(
        methods,
        step,
        random_treatment_regional=random_treatment_regional,
        random_treatment_global=random_treatment_global,
        treatments=groups,
        protest_sources=sources,
    )
    results["source"] = (
        results["source"]
        .str.replace("gpreg", "Registrations data (GPReg)")
        .str.replace("gprep", "Reports data automated (GPRep)")
        .str.replace("acled", "Reports data curated (ACLED)")
    )
    results = results[results["treatment"].isin(groups)]
    results["treatment"] = (
        results["treatment"]
        .str.replace("occ_", "")
        .str.replace("OTHER_CLIMATE_ORG", "other")
    )
    results["target"] = results["target"].str.replace("media_combined_", "")
    dimensions = ["protest", "not_protest", "framing", "goal", "subsidiary_goal"]
    x = alt.X(
        "target:N",
        title="",
        sort=dimensions,
    )
    yname = "ATT estimate (#articles)"
    bars = (
        alt.Chart(results)
        .mark_bar()
        .encode(
            x=x,
            y=alt.Y(
                "coef:Q",
                title=yname,
                scale=alt.Scale(
                    zero=True,  # does not seem to work
                ),
            ),
            color=alt.Color(
                "target:N",
                title="",
                sort=dimensions,
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
            return (
                alt.layer(bars, error_bars, data=results)
                .facet(
                    column=alt.Column("method:N", title="", sort=list(_methods.keys())),
                )
                .resolve_scale(y="independent")
            )
        case "sources":
            return alt.layer(bars, error_bars, data=results).facet(
                column=alt.Column("source:N", title=""),
            )


# plot_groups(kind="methods", step=1)
