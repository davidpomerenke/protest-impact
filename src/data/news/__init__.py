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
    return mediacloud(q=query, region=region, start_date=start, end_date=end)


# # TODO:
# def proportions_for_region(q, region, start_date=None, end_date=None) -> pd.DataFrame:
#     # the #articles containing "und" is an approximation of the total #articles
#     q1, q2 = q, "und"
#     df1 = counts(q1, region, start_date, end_date)
#     df2 = counts(q2, region, start_date, end_date).rolling(28, center=True).mean()
#     df1["count"] = df1["count"] / df2["count"]
#     return df1
