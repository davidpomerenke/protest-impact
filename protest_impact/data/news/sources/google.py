from datetime import date, datetime, timedelta
from itertools import chain
from math import ceil
from os import environ
from time import sleep

import requests
from dateutil import parser
from dotenv import load_dotenv

from protest_impact.types import NewsItem
from protest_impact.util import get

"""
Documentation: https://serpapi.com/search-api
Cost: 50$/month for 5000 searches -> ~0.01$/search
"""

load_dotenv()


def search(
    query: str | None,
    date: date,
    end_date: date = None,
    newspaper: str = None,
    offset: int = 0,
    threshold: int = None,
) -> NewsItem:
    results_per_page = 100
    query_ = query or ""
    site_ = f"site:{newspaper}" if newspaper else ""
    end_date_ = end_date or (date + timedelta(days=1))
    response = get(
        "https://api.scaleserp.com/search",
        headers={},
        params={
            "search_type": "news",
            "q": f"{query_} {site_}",
            "location": "Germany",
            "google_domain": "google.de",
            "gl": "de",
            "hl": "de",
            "time_period": "custom",
            "time_period_min": date.strftime("%m-%d-%Y"),
            "time_period_max": end_date_.strftime("%m-%d-%Y"),
            "num": results_per_page,
            "page": 1 + offset,
            "api_key": environ["SCALE_SERP_API_KEY"],
        },
    )
    response.raise_for_status()
    json = response.json()
    if "news_results" not in json:
        return []
    json = json["news_results"]
    if type(json[0]) == list:
        json = chain(json)
    results = [
        NewsItem(
            date=parser.parse(item["date_utc"]).date(),
            url=item["link"],
            title=item["title"],
            content=item["snippet"] if "snippet" in item else "",
        )
        for item in json
    ]
    if len(results) == results_per_page:
        # print(f"Page number {offset + 1}")
        results += search(query, date, end_date, newspaper, offset + 1)
    if threshold and len(results) >= threshold:
        mid_date = date + (end_date - date) // 2
        results = search(query, date, mid_date, newspaper, offset, threshold)
        results += search(query, mid_date, end_date, newspaper, offset, threshold)
    return list(set(results))
