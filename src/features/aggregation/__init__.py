from itertools import chain

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.utils.timeseries_generation import generate_index
from munch import Munch
from tqdm.auto import tqdm

from src import end, start
from src.cache import cache
from src.data import german_regions
from src.data.news import counts_for_region
from src.data.protests import load_climate_protests_with_labels
from src.data.protests.keywords import climate_queries
from src.data.weather import (
    get_weather_history,
    impute_weather_history,
    interpolate_weather_histories,
)
from src.features.time_series.holidays import get_holidays


@cache
def outcome(
    region: str, start: pd.Timestamp = start, end: pd.Timestamp = end
) -> TimeSeries:
    aspects = dict()
    for qname, query in climate_queries().items():
        df = counts_for_region(region, query)
        if df is None:
            return None
        df = df.set_index("date")
        aspects[f"media_{qname}"] = df["count"]
    df = pd.concat(aspects, axis=1)
    df = df[(df.index >= start) & (df.index <= end)]
    return df


def treatment_unaggregated(source: str) -> pd.DataFrame:
    df = load_climate_protests_with_labels()
    if source == "acled":
        df = df[df["source"] == "acled"]
        df["size"] = df["size_post"]
        return df.drop(columns=["size_pre", "size_post", "source"])
    elif source == "gpreg":
        df = df[df["source"] == "gpreg"]
        df["size"] = df["size_pre"]
        return df.drop(columns=["size_pre", "size_post", "source"])
    elif source == "mean":
        # only for the synthetic control method
        # where we want to avoid false control regions
        df["size"] = df[["size_pre", "size_post"]].mean(axis=1)
        return df.drop(columns=["size_pre", "size_post", "source"])
    elif source == "difference":
        # only for the instrumental variable method
        df = df[
            (df["source"] == "gpreg") & (df["size_pre"] > 0) & (df["size_post"] > 0)
        ]
        return df.drop(columns=["source"])
    return df


def agg_notes(notes: list[str]) -> str:
    notes = [note for note in notes if not pd.isna(note)]
    if len(notes) == 0:
        return np.nan
    else:
        return "; ".join(notes)


@cache
def treatment(region: str, source: str = "acled") -> pd.DataFrame:
    df = treatment_unaggregated(source)
    for actor in df["actor"].unique():
        df["occ_" + actor] = df["actor"].str.contains(actor)
        df["size_" + actor] = df.apply(
            lambda x: x["size"] if x["actor"] == actor else 0, axis=1
        )
    df = df[df.region == region].copy()
    df = df.groupby("date").agg(
        dict(
            notes=agg_notes,
            moderate="any",
            radical="any",
            **{col: "any" for col in df.columns if col.startswith("occ_")},
            **{col: "sum" for col in df.columns if col.startswith("size_")},
        )
    )
    bool_cols = [col for col in df.columns if col.startswith("occ_")] + [
        "moderate",
        "radical",
    ]
    for col in bool_cols:
        df[col] = df[col].astype(int)
    # create index for the whole time range and fill missing values with 0
    index = generate_index(start=start, end=end, freq="D")
    df = df.reindex(index, fill_value=0)
    return df


def location_weights(region: str, source: str = "acled") -> dict[str, float]:
    df = treatment_unaggregated(source)
    df = df[df.region == region].copy()
    weights = dict(df.location.value_counts(sort=True)[:10])
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    return weights


def controls(
    region: str, start: pd.Timestamp = start, end: pd.Timestamp = end
) -> pd.DataFrame:
    date_range = pd.date_range(start=start, end=end, freq="D")
    df = pd.Series(date_range.day_name(), index=date_range)
    df = pd.get_dummies(df, prefix="weekday", drop_first=True)
    df["is_holiday"] = get_holidays(df.index, region).astype(int)
    return df


def instruments(region: str, source: str = "acled") -> pd.DataFrame:
    dfs = []
    weights = []
    for city, weight in location_weights(region, source).items():
        df = get_weather_history(city, "Germany")
        df = df.drop(columns=["wdir"])
        df = df[(df.index >= start) & (df.index <= end)]
        dfs.append(df)
        weights.append(weight)
    df = interpolate_weather_histories(dfs, weights)
    if df.isna().any().any():
        df = impute_weather_history(df)
    return df


def regions(source: str = "acled"):
    df = treatment_unaggregated(source)
    return df.region.unique


def actors(source: str = "acled"):
    df = treatment_unaggregated(source)
    actors = set(chain(*df.actor.dropna().str.split("; "))) - {"OTHER_CLIMATE_ORG"}
    return actors


def region_actor_combinations(source: str = "acled", min_protest_days: int = 0):
    df = treatment_unaggregated(source)
    actors_ = actors(source)
    combinations = []
    for actor in actors_:
        regions_ = df[df.actor == actor].region.unique()
        for region in regions_:
            df_ = df[(df.actor == actor) & (df.region == region)]
            if len(df_) >= min_protest_days:
                combinations.append((region, actor))
    return combinations


def naive_one_region(
    region: str, source: str = "acled"
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    df_y = outcome(region)
    if df_y is None:
        return None, None
    df_w = treatment(region, source)
    df_x = controls(region)
    df = pd.concat([df_y, df_w, df_x], axis=1)
    y = list(df_y.columns)
    w = [ww for ww in df_w.columns if ww.startswith("occ_")]
    x = list(df_x.columns)
    future = w + [xx for xx in x if not xx.startswith("weekday_")]
    future_only = [xx for xx in x if xx.startswith("weekday_")]
    all_cols = y + w + x
    df = df[all_cols]
    vars = Munch(y=y, w=w, x=x, future=future, future_only=future_only)
    return df, vars


def naive_all_regions(source: str = "acled"):
    data = [naive_one_region(region["name"], source) for region in tqdm(german_regions)]
    dfs, vars = zip(*[(d, v) for d, v in data if d is not None])
    vars = vars[0]
    for df in dfs:
        df[vars.y] = df[vars.y] / df[vars.y].mean()
    return dfs, vars
