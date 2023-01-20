import sys
from datetime import date
from time import sleep
from random import shuffle
from itertools import product

from dateutil.relativedelta import relativedelta
from tqdm import tqdm

from protest_impact.data.news.config import (
    media_ids,
    filter_words,
    start_year,
    end_year,
)
from protest_impact.data.news.scraping import download_fulltext
from protest_impact.data.news.sources.mediacloud import search as mediacloud_search
from protest_impact.data.news.sources.google import search as google_search


def download_manually(website: str, engine_name: str, year: int, month: int):
    print(f"{engine_name} {website} {year}-{month:02d}")
    engine = {"google": google_search, "mediacloud": mediacloud_search}[engine_name]
    start_date = date(year=year, month=month, day=1)
    end_date = start_date + relativedelta(months=1)
    site_args = (
        dict(media_id=media_ids[website])
        if engine_name == "mediacloud"
        else dict(site=website)
    )
    results = engine(None, date=start_date, end_date=end_date, **site_args)
    articles = sorted(results, key=lambda r: r.date)
    for article in tqdm(articles):
        if any(w in article.url for w in filter_words):
            continue
        # print(article.url)
        download_fulltext(article)


if len(sys.argv) < 2:
    print("Usage: python download.py random")
    print("Usage: python download.py google|mediacloud <website>")
    sys.exit(1)
if sys.argv[1] == "random":
    configs = list(
        product(
            ["google", "mediacloud"],
            media_ids.keys(),
            range(start_year, end_year),
            range(1, 13),
        )
    )
    shuffle(configs)
    for engine_name, website, year, month in configs:
        try:
            download_manually(website, engine_name, year, month)
        except:
            sleep(5)
else:
    engine_name = sys.argv[1]
    website = sys.argv[2]
    for year in range(start_year, end_year):
        for month in range(1, 13):
            download_manually(website, engine_name)
