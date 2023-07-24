import pandas as pd

from src import end, start
from src.cache import cache
from src.data import german_regions
from src.data.protests.keywords import climate_queries
from src.paths import external_data

path = external_data / "ids-dereko"


@cache
def counts_for_region(
    query_key: str,
    region: str,
    start: pd.Timestamp = start,
    end: pd.Timestamp = end,
) -> pd.DataFrame | None:
    assert query_key in climate_queries().keys()
    codes = [a.code for a in german_regions if a.name == region]
    newspapers = pd.read_csv(path / "corpora/corpora.csv", sep=";")
    newspapers = newspapers[
        newspapers["Region"].isin(codes)
        & (newspapers["von"] <= start.year)
        & (newspapers["bis"] >= end.year)
    ]
    dfs = []
    for corpus, sigle in zip(newspapers["Corpus"], newspapers["Sigle"]):
        df = pd.read_csv(path / "counts" / corpus / sigle / f"{query_key}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        dfs.append(df)
    if len(dfs) == 0:
        return None
    df = pd.concat(dfs).groupby("date").sum()
    if len(df) > 0:
        range_ = pd.date_range(start or df.index.min(), end or df.index.max(), freq="D")
        df = df.reindex(range_, fill_value=0)
    if start is not None:
        df = df[df.index >= start]
    if end is not None:
        df = df[df.index <= end]
    return df[["text_count"]].rename(columns={"text_count": "count"})
