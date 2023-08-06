import pandas as pd

from src import end, start
from src.data.news.dereko import counts_for_region as dereko
from src.data.news.mediacloud.word_counts import counts_for_region as mediacloud
from src.data.protests.keywords import climate_queries


def counts_for_region(
    query_key: str,
    region: str,
    source: str = "all",
    start: pd.Timestamp = start,
    end: pd.Timestamp = end,
) -> pd.DataFrame | None:
    assert source in ["dereko", "mediacloud"]
    queries = climate_queries()
    if source == "mediacloud":
        return mediacloud(q=queries[query_key], region=region, start=start, end=end)
    elif source == "dereko":
        return dereko(query_key=query_key, region=region, start=start, end=end)
