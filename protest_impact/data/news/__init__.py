import pandas as pd

# from protest_impact.data.news.dereko.dereko import (
#     get_scraped_regional_count_df as dereko_get_scraped_regional_count_df,
# )
from protest_impact.data.news.mediacloud.word_counts import (
    counts_for_region as mediacloud_regional_count_df,
)


def counts_for_region(
    region,
    query=None,
    source=None,
    start_date=None,
    end_date=None,
    **kwargs,
) -> pd.DataFrame:
    """
    start_date:
    end_date:
    source: one of "dereko_api", "dereko_scrape", "mediacloud", "mediacloud_stable"
    """

    # if source == "dereko_api":
    #     return dereko_get_regional_count_df(
    #         query_string=query_string,
    #        # region=region,
    #         start_date=start_date,
    #         end_date=end_date,
    #         **kwargs,
    #     )
    # elif source == "dereko_scrape":
    #     return dereko_get_scraped_regional_count_df(
    #         query_func=query_func,
    #         #region=region,
    #         start_date=start_date,
    #         end_date=end_date,
    #         **kwargs,
    #     )
    if source == "mediacloud":
        return mediacloud_regional_count_df(
            q=query,
            region=region,
            start_date=start_date,
            end_date=end_date,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Invalid source {source}. Must be one of 'dereko_api', 'dereko_scrape', 'mediacloud', or 'mediacloud_stable'."
        )


# # TODO:
# def proportions_for_region(q, region, start_date=None, end_date=None) -> pd.DataFrame:
#     # the #articles containing "und" is an approximation of the total #articles
#     q1, q2 = q, "und"
#     df1 = counts(q1, region, start_date, end_date)
#     df2 = counts(q2, region, start_date, end_date).rolling(28, center=True).mean()
#     df1["count"] = df1["count"] / df2["count"]
#     return df1
