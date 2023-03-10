import json
import os
import sys
from datetime import date
from multiprocessing import Pool, freeze_support
from os import environ
from time import sleep, time

import pandas as pd
from dateutil import parser
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from tqdm import tqdm

from protest_impact.data.news.config import (
    all_newspapers_with_id,
    complete_newspapers_with_id,
    end_year,
    filter_words,
    newspapers,
    start_year,
)
from protest_impact.data.news.scraping import download_fulltext
from protest_impact.data.news.sources.google import search as google_search
from protest_impact.data.news.sources.mediacloud import search as mediacloud_search
from protest_impact.data.protests.config import search_string
from protest_impact.types import NewsItem
from protest_impact.util import fulltext_path, project_root, website_name
from protest_impact.util.cache import get

load_dotenv()


def get_protest_article_metadata(
    media_id, start_time=None, last_processed_stories_id=0, verbose=False
):
    if start_time is None:
        start_time = time()
    num_rows = 1000
    response = get(
        "https://api.mediacloud.org/api/v2/stories_public/list/",
        params={
            "q": search_string[1:],
            "fq": f"media_id:{media_id}",
            "rows": num_rows,  # max 1000
            "last_processed_stories_id": last_processed_stories_id,
            "key": environ["MEDIACLOUD_API_KEY"],
        },
        headers={"Accept": "application/json"},
    )
    response.raise_for_status()
    result = response.json()
    end_time = time()
    if verbose:
        print(len(result), end_time - start_time)
    print(len(result))
    if len(result) > 0.9 * num_rows:
        last_processed_stories_id = result[-1]["processed_stories_id"]
        result += get_protest_article_metadata(
            media_id, end_time, last_processed_stories_id, verbose
        )
    return result


def download_protest_articles(newspaper):
    media_id = newspaper["media_id"]
    with open("started.txt", "a") as f:
        f.write(f"{media_id} {newspaper['name']}\n")
    articles = get_protest_article_metadata(media_id)
    for i, article in enumerate(articles):
        print(i, len(articles), article["url"])
        start_time = time()
        if article["publish_date"] is None:
            continue
        if any(word in article["url"] for word in filter_words):
            continue
        download_fulltext(
            NewsItem(
                url=article["url"],
                title=article["title"],
                date=parser.parse(article["publish_date"]),
                content="",
            )
        )
    with open("finished.txt", "a") as f:
        f.write(f"{media_id} {newspaper['name']}\n")


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
        newspaper=(website, newspapers[website]),
    )
    articles = sorted(results, key=lambda r: r.date)
    return articles


def get_counts(
    get_metadata=get_monthly_metadata, newspapers_with_id=complete_newspapers_with_id
):
    data = []
    for newspaper in tqdm(newspapers_with_id):
        for engine in ["google"]:  # , "mediacloud"]:
            print(newspaper, engine)
            for year in range(start_year, end_year):
                for month in range(1, 13):
                    articles = get_metadata(newspaper, engine, year, month)
                    data.append(
                        {
                            "newspaper": newspaper,
                            "engine": engine,
                            "date": date(year=year, month=month, day=1),
                            "count": len(articles),
                        }
                    )
    return pd.DataFrame(data)


def get_halfmonthly_metadata(website: str, engine_name: str, year: int, month: int):
    engine = {"google": google_search, "mediacloud": mediacloud_search}[engine_name]
    start_date = date(year=year, month=month, day=1)
    mid_date = start_date + relativedelta(days=15)
    end_date = start_date + relativedelta(months=1)
    results = engine(
        None,
        date=start_date,
        end_date=mid_date,
        newspaper=(website, newspapers[website]),
    )
    results += engine(
        None, date=mid_date, end_date=end_date, newspaper=(website, newspapers[website])
    )
    articles = sorted(results, key=lambda r: r.date)
    return articles


def get_weekly_metadata(website: str, engine_name: str, year: int, month: int):
    engine = {"google": google_search, "mediacloud": mediacloud_search}[engine_name]
    start_date = date(year=year, month=month, day=1)
    mid_date_1 = start_date + relativedelta(days=7)
    mid_date_2 = start_date + relativedelta(days=14)
    mid_date_3 = start_date + relativedelta(days=21)
    end_date = start_date + relativedelta(months=1)
    results = engine(
        None,
        date=start_date,
        end_date=mid_date_1,
        newspaper=(website, newspapers[website]),
    )
    results += engine(
        None,
        date=mid_date_1,
        end_date=mid_date_2,
        newspaper=(website, newspapers[website]),
    )
    results += engine(
        None,
        date=mid_date_2,
        end_date=mid_date_3,
        newspaper=(website, newspapers[website]),
    )
    results += engine(
        None,
        date=mid_date_3,
        end_date=end_date,
        newspaper=(website, newspapers[website]),
    )
    articles = sorted(results, key=lambda r: r.date)
    return articles


def download_manually(website: str, engine_name: str, year: int, month: int):
    print(f"{engine_name} {website} {year}-{month:02d}")
    for article in tqdm(get_monthly_metadata(website, engine_name, year, month)):
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


def mute():
    sys.stdout = open(os.devnull, "w")


if __name__ == "__main__":
    freeze_support()
    if len(sys.argv) < 2:
        print("Usage: python download.py random")
        print("Usage: python download.py google|mediacloud <website>")
        sys.exit(1)
    elif sys.argv[1] == "parallel":
        n = int(sys.argv[2])
        websites = list(newspapers.keys())
        with Pool(n) as p:
            try:
                p.map(download_all, websites)
            except KeyboardInterrupt:
                print("Caught KeyboardInterrupt, terminating workers")
                p.terminate()
                p.join()
    elif sys.argv[1] == "protests":
        media_id = int(sys.argv[2])
        download_protest_articles(media_id)
    elif sys.argv[1] == "protests_parallel":
        with open(
            project_root / "data" / "news" / "scrapable_mediacloud_newspapers.json"
        ) as f:
            newspapers = json.load(f)
        n = int(sys.argv[2])
        with Pool(n) as p:
            try:
                p.map(download_protest_articles, newspapers)
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
