import re
from datetime import date
from os import environ

import pandas as pd
from dotenv import load_dotenv

from protest_impact.data.news.mediacloud.newspaper_collections import region_tags
from protest_impact.util import (
    cache,
    get_cached,
    kill_umlauts_without_mercy,
    project_root,
)

"""
Documentation:
 - https://github.com/mediacloud/backend/blob/master/doc/api_2_0_spec/api_2_0_spec.md
 - https://mediacloud.org/support/query-guide/
Cost: free
"""

load_dotenv()


def counts_for_region(q, region, start_date=None, end_date=None) -> pd.DataFrame:
    region_query = "tags_id_media:" + str(to_mediacloud_region(region))
    return counts(q, region_query, start_date, end_date)


def counts(q: str, fq: str, start_date: date, end_date: date) -> pd.DataFrame:
    _start_date = date(2015, 1, 1)
    _end_date = date(2023, 6, 30)
    if start_date is not None and start_date < _start_date:
        print("Warning: start_date is before 2019-01-01, using 2019-01-01")
    if end_date is not None and end_date > _end_date:
        print("Warning: end_date is after 2022-01-01, using 2022-01-01")
    df = _counts_for_general_query(q, fq, _start_date, _end_date)
    if start_date is not None:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df[df["date"] <= pd.Timestamp(end_date)]
    return df


def _counts_for_general_query(
    q: str, fq: str, start_date: date, end_date: date
) -> pd.DataFrame:
    q_ = re.sub(r"\*(\w)", r"\1", q)
    q_ = (
        q_.replace("ä", "a")
        .replace("ö", "o")
        .replace("ü", "u")
        .replace("ß", "ss")
        .replace("Ä", "A")
        .replace("Ö", "O")
        .replace("Ü", "U")
    )
    if q_ != q:
        print(
            f'Warning: query "{q[:10]}..." contains leading *, or special characters; removing them for Mediacloud.'
        )
        q = q_
    counts = get_cached(
        "https://api.mediacloud.org/api/v2/stories_public/count/",
        params={
            "q": q,
            "fq": f"{fq} and publish_date:[{start_date}T00:00:00Z TO {end_date}T00:00:00Z]",
            "split": True,
            "split_period": "day",
            "key": environ["MEDIACLOUD_API_KEY"],
        },
    )
    counts.raise_for_status()
    df = pd.DataFrame(counts.json()["counts"])
    if len(df) == 0:
        raise ValueError("No results found for query")
    df["date"] = pd.to_datetime(df["date"])
    # WORKAROUND: sum counts by day
    df = df.groupby("date").sum().reset_index()
    # set missing dates to 0 (without making date the index)
    df = (
        df.set_index("date")
        .reindex(pd.date_range(start_date, end_date, freq="D"), fill_value=0)
        .reset_index()
    )
    df = df.rename(columns={"index": "date"})
    return df


def to_mediacloud_region(region: str):
    region_tags_ = {
        kill_umlauts_without_mercy(k.lower()): v for k, v in region_tags.items()
    }
    return region_tags_[kill_umlauts_without_mercy(region.lower())]
