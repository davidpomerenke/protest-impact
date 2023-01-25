import os
import sys
from datetime import date
from time import sleep
from random import shuffle
from itertools import product
from multiprocessing import Pool, freeze_support

from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import pandas as pd

from protest_impact.data.news.config import (
    media_ids,
    filter_words,
    start_year,
    end_year,
)
from protest_impact.data.news.scraping import download_fulltext
from protest_impact.data.news.sources.mediacloud import search as mediacloud_search
from protest_impact.data.news.sources.google import search as google_search
from protest_impact.util import website_name, fulltext_path
from protest_impact.data.news.config import (
    all_newspapers_with_id,
    complete_newspapers_with_id,
    start_year,
    end_year,
)


def get_all_metadata():
    data = []
    for newspaper in complete_newspapers_with_id:
        for engine in ["google", "mediacloud"]:
            for year in tqdm(range(start_year, end_year)):
                for month in range(1, 13):
                    articles = get_monthly_metadata(newspaper, engine, year, month)
                    for article in articles:
                        data.append(
                            {
                                "newspaper": newspaper,
                                "engine": engine,
                                "path": fulltext_path(article),
                            }
                        )
    df_sources = pd.DataFrame(data)
    df_sources.head()


def get_monthly_metadata(
    website: str, engine_name: str, year: int, month: int, threshold: int = None
):
    engine = {"google": google_search, "mediacloud": mediacloud_search}[engine_name]
    start_date = date(year=year, month=month, day=1)
    end_date = start_date + relativedelta(months=1)
    results = engine(
        None,
        date=start_date,
        end_date=end_date,
        threshold=threshold,
        newspaper=(website, media_ids[website]),
    )
    articles = sorted(results, key=lambda r: r.date)
    return articles


def get_halfmonthly_metadata(website: str, engine_name: str, year: int, month: int):
    engine = {"google": google_search, "mediacloud": mediacloud_search}[engine_name]
    site_args = (
        dict(media_id=media_ids[website])
        if engine_name == "mediacloud"
        else dict(site=website)
    )
    start_date = date(year=year, month=month, day=1)
    mid_date = start_date + relativedelta(days=15)
    end_date = start_date + relativedelta(months=1)
    results = engine(None, date=start_date, end_date=mid_date, **site_args)
    results += engine(None, date=mid_date, end_date=end_date, **site_args)
    articles = sorted(results, key=lambda r: r.date)
    return articles


def get_weekly_metadata(website: str, engine_name: str, year: int, month: int):
    engine = {"google": google_search, "mediacloud": mediacloud_search}[engine_name]
    site_args = (
        dict(media_id=media_ids[website])
        if engine_name == "mediacloud"
        else dict(site=website)
    )
    start_date = date(year=year, month=month, day=1)
    mid_date_1 = start_date + relativedelta(days=7)
    mid_date_2 = start_date + relativedelta(days=14)
    mid_date_3 = start_date + relativedelta(days=21)
    end_date = start_date + relativedelta(months=1)
    results = engine(None, date=start_date, end_date=mid_date_1, **site_args)
    results += engine(None, date=mid_date_1, end_date=mid_date_2, **site_args)
    results += engine(None, date=mid_date_2, end_date=mid_date_3, **site_args)
    results += engine(None, date=mid_date_3, end_date=end_date, **site_args)
    articles = sorted(results, key=lambda r: r.date)
    return articles


def download_manually(website: str, engine_name: str, year: int, month: int):
    print(f"{engine_name} {website} {year}-{month:02d}")
    for article in tqdm(get_weekly_metadata(website, engine_name, year, month)):
        if any(w in article.url for w in filter_words):
            continue
        if website_name(article.url) not in all_newspapers_with_id.keys():
            continue
        download_fulltext(article)


def download_all(website: str):
    for engine_name in ["google"]:  # , "mediacloud"]:
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                download_manually(website, engine_name, year, month)


def download_randomly(*args):
    configs = list(
        product(
            ["google"],  # "mediacloud"],
            media_ids.keys(),
            range(start_year, end_year + 1),
            range(1, 13),
        )
    )
    shuffle(configs)
    for engine_name, website, year, month in configs:
        try:
            download_manually(website, engine_name, year, month)
        except:
            print("ERROR")
            print(f"Failed to download {engine_name} {website} {year}-{month:02d}")
            sleep(5)


def mute():
    sys.stdout = open(os.devnull, "w")


if __name__ == "__main__":
    freeze_support()
    if len(sys.argv) < 2:
        print("Usage: python download.py random")
        print("Usage: python download.py google|mediacloud <website>")
        sys.exit(1)
    if sys.argv[1] == "random":
        download_randomly()
    elif sys.argv[1] == "random_parallel":
        n = int(sys.argv[2])
        with Pool(n, initargs=mute) as p:
            try:
                p.map(download_randomly, range(n))
            except KeyboardInterrupt:
                print("Caught KeyboardInterrupt, terminating workers")
                p.terminate()
                p.join()
    elif sys.argv[1] == "parallel":
        n = int(sys.argv[2])
        websites = list(media_ids.keys())
        with Pool(n) as p:
            try:
                p.map(download_all, websites)
            except KeyboardInterrupt:
                print("Caught KeyboardInterrupt, terminating workers")
                p.terminate()
                p.join()
    else:
        engine_name = sys.argv[1]
        website = sys.argv[2]
        start_year_ = int(sys.argv[3]) if len(sys.argv) > 3 else start_year
        for year in range(start_year_, end_year + 1):
            for month in range(1, 13):
                download_manually(website, engine_name, year, month)
