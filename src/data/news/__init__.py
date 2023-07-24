import pandas as pd

# from src.data.news.dereko.dereko import (
#     get_scraped_regional_count_df as dereko_get_scraped_regional_count_df,
# )
from src.data.news.mediacloud.word_counts import counts_for_region as mediacloud


def counts_for_region(
    region: str, query: str, start: pd.Timestamp = None, end: pd.Timestamp = None
) -> pd.DataFrame:
    """
    start_date:
    end_date:
    source: one of "dereko", "mediacloud"
    """
    # if source == "mediacloud":::::
    return mediacloud(q=query, region=region, start_date=start, end_date=end)
