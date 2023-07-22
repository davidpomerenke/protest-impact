import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm.auto import tqdm

from src.cache import cache
from src.data import german_regions
from src.data.news import get_regional_count_df
from src.data.news.coverage_filter import filter_protests
from src.data.news.dereko.dereko import get_scraped_entries
from src.data.protests import get_climate_protests, get_climate_queries
from src.data.protests.random import get_random_events
from src.models.synthetic_control.models import AutoRegressor
from src.models.synthetic_control.util import train_test_split
from src.util.functools import function_name


@cache(ignore=["query_func", "dereko_entries"])
def get_regional_counts_for_protest(
    region,
    event_date,
    query_str=None,
    query_func=None,
    source="mediacloud",
    dereko_entries=None,
    n_days_train=7 * 4 * 6,
    n_days_predict=7 * 4,
):
    """
    Gets article counts around the protest event, both for the region of the protest
    and for other regions where no protests happened.

        protest: a single protest event, such as a row from ACLED
        protest_df: a dataframe of protests, such as ACLED
        co_terms: additional terms to include as predictors (that is, only for the control regions)
        source: the source to get word count histories from
        n_days: the number of days before and after the protest to get word counts for

    Returns a dataframe with columns:
        date: the date of the word count
        count: the word count
        region: the region of the word count
        is_protest_region: whether the region is the region of the protest
    """

    start_date = (
        None
        if n_days_train is None
        else event_date.date() - relativedelta(days=n_days_train)
    )
    end_date = (
        None
        if n_days_predict is None
        else event_date.date() + relativedelta(days=n_days_predict - 1)
    )
    kwargs = {}
    if source == "dereko_scrape":
        assert dereko_entries is not None
        kwargs["entries"] = dereko_entries
    regions = [a["name"] for a in german_regions]
    _dfs = []
    for _region in regions:
        is_protest_region = _region == region
        df = get_regional_count_df(
            query_string=query_str,
            query_func=query_func,
            region=_region,
            start_date=start_date,
            end_date=end_date,
            source=source,
            **kwargs,
        )
        if df is None:
            if is_protest_region:
                return None
            else:
                continue
        if start_date is not None:
            df = df[df["date"] >= pd.Timestamp(start_date)]
        if end_date is not None:
            df = df[df["date"] <= pd.Timestamp(end_date)]
        if df["count"].mean() == 0:
            if is_protest_region:
                return None
            else:
                continue
        df["region"] = _region
        df["is_protest_region"] = is_protest_region
        _dfs.append(df)
    df = pd.concat(_dfs)
    return pivotize(df, event_date)


def filter_regions(
    df,
    region,
    event_date,
    reference_events,
    n_days_protest_free_pre=0,
    n_days_protest_free_post=0,
    min_control_regions=3,
    min_count=0,
):
    """
    df: df in pivot form, with each column representing a region
    """

    def is_protest_free(region, except_protest_date=False):
        return not any(
            (reference_events["admin1"] == region)
            & (
                reference_events["event_date"]
                > event_date - relativedelta(days=n_days_protest_free_pre)
            )
            & (
                reference_events["event_date"]
                < event_date + relativedelta(days=n_days_protest_free_post)
            )
            & (
                reference_events["event_date"] != event_date
                if except_protest_date
                else True
            )
        )

    if not is_protest_free(region, except_protest_date=True):
        return None

    control_regions = []
    for _region in df.columns:
        if region == _region:
            continue
        if not is_protest_free(_region):
            continue
        if df[_region].mean() < min_count:
            continue
        control_regions.append(_region)

    if len(control_regions) < min_control_regions:
        return None

    df = df[[region, *control_regions]]
    return df


def pivotize(df, event_date):
    pivot_df = (
        df.pivot(index="date", columns=["region"], values="count")
        .reset_index()
        .fillna(0)
    )
    pivot_df.index = (pivot_df["date"] - pd.Timestamp(event_date)).dt.days
    pivot_df.index = pd.RangeIndex(pivot_df.index[0], pivot_df.index[-1] + 1)
    pivot_df = pivot_df.drop(columns=["date"])
    return pivot_df


def add_weekday_interactions(weekdays, X, pivot_df):
    # TODO: refactor
    if weekdays in ["dummies", "interactions", "interactions_only"]:
        pivot_df["weekday"] = pivot_df["date"].dt.weekday
        pivot_df = pd.get_dummies(pivot_df, columns=["weekday"])

    def weekday_interactions(Z):
        extra_columns = []
        for region in Z.columns:
            for weekday in range(7):
                extra_columns.append(
                    pd.Series(
                        Z[region] * Z[f"weekday_{weekday}"],
                        name=f"{region}_weekday_{weekday}",
                    )
                )
        return pd.concat(extra_columns, axis=1)

    if weekdays == "interactions":
        X = pd.concat([X, weekday_interactions(X)], axis=1)
    if weekdays == "interactions_only":
        X = weekday_interactions(X)


def get_standard_time_series(event_type, n):
    reference_events = get_climate_protests()
    region_weights = reference_events["admin1"].value_counts(normalize=True)
    if event_type == "protest":
        events = reference_events.sample(n)
    elif event_type == "random":
        events = get_random_events(n, region_weights=region_weights)
    query = get_climate_queries()["topic_focused"]
    time_series = []
    for p in events.to_dict("records"):
        counts = get_regional_counts_for_protest(
            query,
            p,
            reference_events,
            n_days_protest_free_pre=1,
            n_days_protest_free_post=1,
        )
        if counts is not None:
            X, Y = pivotize(counts, p)
            time_series.append((X, Y, p))
    return time_series
