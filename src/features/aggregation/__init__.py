from itertools import chain

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.utils.timeseries_generation import generate_index
from tqdm.auto import tqdm

from src import end, start
from src.cache import cache
from src.data import german_regions
from src.data.covid_restrictions import load_covid
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
    for source_name, source in [("online", "mediacloud"), ("print", "dereko")]:
        for qname, query in climate_queries(short=True).items():
            df = counts_for_region(qname, region, source)
            if df is None:
                return None
            aspects[f"media_{source_name}_{qname}"] = df["count"]
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


@cache
def controls(
    region: str, start: pd.Timestamp = start, end: pd.Timestamp = end
) -> pd.DataFrame:
    date_range = pd.date_range(start=start, end=end, freq="D")
    df = pd.Series(date_range.day_name(), index=date_range)
    df = pd.get_dummies(df, prefix="weekday", drop_first=True)
    df["holiday"] = get_holidays(df.index, region).astype(int)
    return df


@cache
def weather(region: str, source: str = "acled") -> pd.DataFrame:
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
    df = df.add_prefix("weather_")
    return df


def instruments(region: str, source: str = "acled") -> pd.DataFrame:
    df = pd.concat([weather(region, source), load_covid()], axis=1)
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


def one_region(
    region: str,
    include_instruments: bool = False,
    ignore_group: bool = False,
    protest_source: str = "acled",
) -> pd.DataFrame | None:
    df_y = outcome(region)
    if df_y is None:
        return None
    df_w = treatment(region, protest_source)
    df_w = df_w[[c for c in df_w.columns if c.startswith("occ_")]]
    if ignore_group:
        df_w = df_w.any(axis=1).astype(int).to_frame("occ_protest")
    df_x = controls(region)
    if include_instruments:
        df_z = instruments(region, protest_source)
        df = pd.concat([df_y, df_w, df_x, df_z], axis=1)
    else:
        df = pd.concat([df_y, df_w, df_x], axis=1)
    return df


@cache
def all_regions(
    include_instruments: bool = False,
    ignore_group: bool = False,
    region_dummies: bool = False,
    protest_source: str = "acled",
) -> list[pd.DataFrame]:
    dfs = [
        (
            region.name,
            one_region(region.name, include_instruments, ignore_group, protest_source),
        )
        for region in tqdm(german_regions)
    ]
    names, dfs = zip(*[(name, df) for name, df in dfs if df is not None])
    if region_dummies:
        region_dummies = pd.get_dummies(names, prefix="region", drop_first=True)
        for i, df in enumerate(dfs):
            df[region_dummies.columns] = region_dummies.iloc[i]
    return list(zip(names, dfs))
