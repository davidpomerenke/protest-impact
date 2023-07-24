import re
from os import environ

import pandas as pd
from dotenv import load_dotenv

from src import end, kill_umlauts_without_mercy, start
from src.cache import get_cached
from src.data.news.mediacloud.newspaper_collections import region_tags

"""
Documentation:
 - https://github.com/mediacloud/backend/blob/master/doc/api_2_0_spec/api_2_0_spec.md
 - https://mediacloud.org/support/query-guide/
Cost: free
"""

load_dotenv()


def counts_for_region(
    q: str, region: str, start_date: pd.Timestamp = start, end_date: pd.Timestamp = end
) -> pd.DataFrame:
    region_query = "tags_id_media:" + str(to_mediacloud_region(region))
    return counts(q, region_query, start_date, end_date)


def counts(
    q: str, fq: str, start: pd.Timestamp = start, end: pd.Timestamp = end
) -> pd.DataFrame:
    _start_date = pd.Timestamp("2015-01-01")
    _end_date = pd.Timestamp("2023-06-30")
    assert start is None or start >= _start_date, "Start date "
    assert end is None or end <= _end_date, "Mediacloud only supports 2015-2023"
    df = _counts_for_general_query(q, fq, _start_date, _end_date)
    if df is None:
        return None
    if start is not None:
        df = df[df["date"] >= start]
    if end is not None:
        df = df[df["date"] <= end]
    return df


def _counts_for_general_query(
    q: str, fq: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    # query may contain leading *, or special characters;
    # removing them for Mediacloud:
    q = re.sub(r"\*(\w)", r"\1", q)
    q = (
        q.replace("ä", "a")
        .replace("ö", "o")
        .replace("ü", "u")
        .replace("ß", "ss")
        .replace("Ä", "A")
        .replace("Ö", "O")
        .replace("Ü", "U")
    )
    counts = get_cached(
        "https://api.mediacloud.org/api/v2/stories_public/count/",
        params={
            "q": q,
            "fq": f"{fq} and publish_date:[{start.isoformat()}Z TO {end.isoformat()}Z]",
            "split": True,
            "split_period": "day",
            "key": environ["MEDIACLOUD_API_KEY"],
        },
    )
    counts.raise_for_status()
    df = pd.DataFrame(counts.json()["counts"])
    if len(df) == 0:
        return None
    df["date"] = pd.to_datetime(df["date"])
    # WORKAROUND: sum counts by day
    df = df.groupby("date").sum().reset_index()
    # set missing dates to 0 (without making date the index)
    df = (
        df.set_index("date")
        .reindex(pd.date_range(start, end, freq="D"), fill_value=0)
        .reset_index()
    )
    df = df.rename(columns={"index": "date"})
    return df


def to_mediacloud_region(region: str) -> int:
    region_tags_ = {
        kill_umlauts_without_mercy(k.lower()): v for k, v in region_tags.items()
    }
    return region_tags_[kill_umlauts_without_mercy(region.lower())]
