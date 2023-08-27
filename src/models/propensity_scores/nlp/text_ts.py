import re

import holidays
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from src.cache import cache
from src.data import german_regions
from src.data.protests.keywords import climate_queries
from src.features.aggregation import one_region, treatment_unaggregated


@cache
def region_headlines(region):
    df = one_region(region, include_press=True, include_texts=True, text_cutoff="title")
    queries = climate_queries(mode="raw", short=True)
    queries = [
        a.replace("?", "").replace("*", "").replace("\\", "")
        for b in queries.values()
        for a in b
    ]

    def get_climate_headlines(row):
        text = row["text"] + "\n\n" + row["press"]
        text = [
            line
            for line in text.split("\n\n")
            if any([q in line.lower() for q in queries])
        ]
        text = "\n".join(text)
        text = re.sub(r"(POLITIK|ROUNDUP|WMD|WDH|WMO)/?(: )?", "", text)
        text = (
            text.replace("&quot;", '"').replace("&#8220;", '"').replace("&#8221;", '"')
        )
        return text

    return df.apply(get_climate_headlines, axis=1)


def get_numbers(region):
    # df = one_region(
    #     # region, add_features=["ewm", "diff"], ignore_group=True, ignore_medium=True
    # )
    # values = dict(
    #     media_combined_all= "climate",
    #     media_combined_protest= "climate protest",
    #     media_combined_all_diff1= "1day change",
    #     media_combined_all_diff7= "7day change",
    #     media_combined_all_diff28= "28day change",
    #     occ_protest_ewm28= "28day ewm",
    #     occ_protest_ewm224= "224day ewm",
    # )
    # df = df[values.keys()]
    # row = df.iloc[-30]
    # for c in df.columns:
    #     if "ewm" in c:
    #         print(f"{c}: {row[c]:.2f}")
    #     else:
    #         print(f"{c}: {row[c]:.0f}")
    # return df
    pass


def get_text_for_dates(start, end, df, region):
    region_code = [a.code for a in german_regions if a.name == region][0]
    df = df[df["region"] == region]
    holi = holidays.Germany(years=range(2018, 2023), subdiv=region_code)
    items = []
    for date in pd.date_range(start, end):
        text = ""
        text += f"{region}, "
        protests = df[df["date"] == date]
        date_info = (
            f"{date.strftime('%A')}, {date.day}. {date.month_name()} {date.year}"
        )
        if date in holi:
            date_info += f", {holi[date]}"
        days_ago = (end - date).days
        if days_ago != 0:
            date_info += f" ({days_ago} days ago)"
        elif days_ago == 0:
            date_info += " (today)"
        text += date_info + "\n"
        text += "has protests: " + ("yes" if len(protests) > 0 else "no") + "\n"
        if len(protests) > 0:
            text += "number of protests: " + str(len(protests)) + "\nProtests:\n"
            for _, protest in protests.iterrows():
                text += f"{protest['actor']}: {protest['notes']}\n"
        text += "Headlines:\n"
        text += region_headlines(region)[date]
        text += "\n"
        items.append(text)
    return items


@cache
def process(df, region):
    print(region)
    X = []
    y = []
    d = []
    for date in tqdm(pd.date_range("2020-02-01", "2022-12-31")):
        start = date - pd.Timedelta(days=30)
        items = get_text_for_dates(start, date, df, region)
        i = items[-1].index("has protests: ") + len("has protests: ")
        X.append("\n".join(items[:-1] + [items[-1][:i]]))
        y.append(items[-1][i : i + 3].strip())
        d.append(date)
    return X, y, d


@cache
def text_timeseries():
    df = treatment_unaggregated("acled")
    df = df[df.country == "Germany"]
    regions = [
        a
        for a in df.region.unique()
        if a
        not in [
            "Baden-Württemberg",
            "Mecklenburg-Vorpommern",
            "Bremen",
        ]
    ]
    X, y, d = zip(
        *Parallel(n_jobs=-1)(delayed(process)(df, region) for region in tqdm(regions))
    )
    X = [a for b in X for a in b]
    y = [a for b in y for a in b]
    d = [a for b in d for a in b]
    df = pd.DataFrame({"text": X, "label": y, "date": d})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop(columns=["date"])
    return df
