import pandas as pd
from dateutil.relativedelta import relativedelta

from protest_impact.util import project_root

count_df = pd.read_csv(
    project_root / "data" / "protest" / "protest_and_topic_counts.csv"
)
count_df["date"] = pd.to_datetime(count_df["date"])

topic_count_df = count_df.copy()[count_df["type"] == "general"]
topic_count_df["count"] = topic_count_df.groupby(["date", "media_id", "name"])[
    "count"
].transform(lambda x: x / x.sum())


def get_topic_counts(args):
    keyword, media_id, inner_distance, outer_distance = args
    # retrieve topic of keyword
    keyword_rows = count_df[count_df["keyword"] == keyword]["topic"]
    if len(keyword_rows) == 0:
        return None
    topic = keyword_rows.iloc[0]
    protest_counts = count_df[
        (count_df["type"] == "protest")
        & (count_df["topic"] == topic)
        & (count_df.date.dt.year >= 2014)
        & (count_df.date.dt.year <= 2021)
    ].copy()
    topic_count_df_ = topic_count_df[
        (topic_count_df.media_id == media_id) & (topic_count_df.keyword == keyword)
    ].copy()
    protest_counts["past_start"] = protest_counts["date"].apply(
        lambda x: x - relativedelta(days=outer_distance)
    )
    protest_counts["past_end"] = protest_counts["date"].apply(
        lambda x: x - relativedelta(days=inner_distance)
    )
    protest_counts["future_start"] = protest_counts["date"].apply(
        lambda x: x + relativedelta(days=inner_distance)
    )
    protest_counts["future_end"] = protest_counts["date"].apply(
        lambda x: x + relativedelta(days=outer_distance)
    )
    protest_counts["past_count"] = protest_counts.apply(
        lambda x: topic_count_df_[
            (topic_count_df_.date >= x["past_start"])
            & (topic_count_df_.date <= x["past_end"])
        ]["count"].sum(),
        axis=1,
    )
    protest_counts["future_count"] = protest_counts.apply(
        lambda x: topic_count_df_[
            (topic_count_df_.date >= x["future_start"])
            & (topic_count_df_.date <= x["future_end"])
        ]["count"].sum(),
        axis=1,
    )
    protest_counts["keyword"] = keyword
    protest_counts["topic"] = topic
    return protest_counts
