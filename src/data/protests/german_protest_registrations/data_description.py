import os

import geocoder
import pandas as pd
from dotenv import load_dotenv

from src.cache import cache
from src.data import german_regions
from src.paths import interim_data

load_dotenv()


def is_capital(city, region):
    return any([a for a in german_regions if a.name == region and a.capital == city])


def get_population(city, region):
    # use geocoder to get the number of inhabitants for each city
    g = geocoder.geonames(f"{city}, {region}", key=os.environ["GEONAMES_USERNAME"])
    return g.population


@cache
def overview_table():
    path = interim_data / "german-protest-registrations/all-protests.csv"
    df = pd.read_csv(path, parse_dates=["Datum"])
    df = df.rename(columns={"Datum": "date", "Stadt": "city", "Bundesland": "region"})
    df["year"] = df["date"].dt.year.astype(str).apply(lambda x: x[:-2])
    agg_df = (
        df.groupby(["region", "city", "year"]).size().unstack().fillna(0).astype(int)
    )
    agg_df = agg_df.drop(columns=["2023", "n"])
    # add column to agg_df whether or not the "Teilnehmer" column is available
    agg_df["registrations"] = df.groupby(["region", "city"]).apply(
        lambda x: x["Teilnehmer"].mean() > 10
    )
    agg_df["observations"] = df.groupby(["region", "city"]).apply(
        lambda x: x["Teilnehmer (tatsächlich)"].mean() > 10
    )
    # add column whether or not the datasubset is included in the analysis
    agg_df["incl?"] = agg_df["registrations"] & (agg_df["2020"] > 0)

    agg_df["capital"] = agg_df.apply(lambda x: is_capital(x.name[1], x.name[0]), axis=1)
    agg_df["kpop"] = agg_df.apply(
        lambda x: get_population(x.name[1], x.name[0]), axis=1
    )
    return agg_df


@cache
def pretty_overview_table(symbol="✓"):
    # symbol: "✓" or "x"
    df = overview_table()
    # calculate sums row (add later)
    sums = df.sum(numeric_only=False)
    sums.name = ("", "sum", "")
    # insert empty rows for regions that are not in the data
    for region in german_regions:
        if region["name"] not in df.index:
            df.loc[(region["name"], "–"), :] = ""
            df.loc[(region["name"], "–"), "kpop"] = "–"
    # sort by region name
    df = df.sort_index()
    # add sum row at the bottom
    df = pd.concat([df, sums.to_frame().T])
    df["#reg?"] = df["registrations"].replace({0: " ", 1: symbol})
    df["#obs?"] = df["observations"].replace({0: " ", 1: symbol})
    df["cap?"] = df["capital"].replace({0: " ", 1: symbol})
    df["incl?"] = df["incl?"].replace({0: " ", 1: symbol})
    df["kpop"] = df["kpop"].apply(
        lambda x: f"{int(x/1000):,}" if x not in ["–", ""] else x
    )
    years = ["2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022"]
    for col in years:
        df[col] = df[col].apply(lambda x: f"{int(x):,}" if x != "" else "")
    df = df.replace(0, "")
    df = df.replace("0", "")
    # reorder columns
    df = df[["kpop", "cap?", "#reg?", "#obs?", *years]]  # "incl?",
    df = df.rename(index={"Mecklenburg-Vorpommern": "Meck.-Vorpommern"})
    return df
