import json

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.cache import cache
from src.data import _get_name
from src.paths import external_data


@cache
def press_releases(cutoff: int | str | None = 500) -> pd.Series:
    title_only = isinstance(cutoff, str) and cutoff == "title"
    data = []
    for file in tqdm(list(external_data.glob("nexis/climate/json/**/*.json"))):
        with open(file) as f:
            item = json.load(f)
            # parse date field
            item["date"] = pd.to_datetime(item["date"])
            data.append(item)
    df = pd.DataFrame(data)
    df["text"] = df["text"].str.removeprefix(")").str.strip()
    if title_only:
        df["text"] = df["title"]
    else:
        # group texts and titles for each date together
        df["text"] = (
            df["title"]
            + "\n\n"
            + df["location"].fillna("")
            + " "
            + df["text"].fillna("").str[:cutoff]
        )
    df["date"] = df["date"].dt.date
    s = df.groupby("date").agg({"text": "\n\n".join})["text"]
    # fill missing dates with empty strings
    s = s.reindex(
        pd.date_range(
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2022-12-31"),
        ),
        fill_value="",
    )
    s.name = "press"
    return s


@cache
def _read_fulltexts(cutoff: int | str | None = None):
    title_only = isinstance(cutoff, str) and cutoff == "title"
    articles = []
    article = None
    title = None
    for file in tqdm(
        list((external_data / "ids-dereko/climate-articles").glob("*.jsonl"))
    ):
        with open(file) as f:
            for line in f:
                # every article is split into multiple lines
                item = json.loads(line)
                if item["title"] == title:
                    if not title_only:
                        # append to ongoing article
                        article["text"] += "\n...\n" + item["text"]
                else:
                    # start new article
                    if article:
                        article["text"] = article["text"][
                            : cutoff if isinstance(cutoff, int) else None
                        ]
                        articles.append(article)
                    article = item
                    if item["title"] is not None:
                        article["text"] = item["title"]
                        if not title_only:
                            article["text"] += "\n\n" + item["text"]
                    title = item["title"]
    corpora = pd.read_csv(external_data / "ids-dereko/corpora/corpora.csv", sep=";")
    regions = {
        row["Sigle"]: row["Region"]
        for _, row in corpora.iterrows()
        if row["Region"] not in ["AT", "CH", np.nan]
    }
    articles = [a for a in articles if a["corpus"].lower() in regions]
    for article in tqdm(articles):
        article["region"] = _get_name(regions[article["corpus"].lower()])
        article["date"] = pd.to_datetime(article["date"]).date()
    return articles


@cache
def fulltexts(region: str, cutoff: int | None = None) -> pd.Series:
    articles = _read_fulltexts(cutoff=cutoff)
    articles = [a for a in articles if a["region"] == region]
    df = pd.DataFrame(
        dict(
            date=[a["date"] for a in articles],
            text=[a["text"] for a in articles],
        )
    )
    s = df.groupby("date").agg({"text": "\n\n---\n\n".join})["text"]
    # fill missing dates with empty strings
    s = s.reindex(
        pd.date_range(
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2022-12-31"),
        ),
        fill_value="",
    )
    return s
