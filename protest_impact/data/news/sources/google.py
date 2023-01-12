from datetime import date, timedelta
from os import environ
from time import sleep

import requests
from dotenv import load_dotenv

from protest_impact.types import NewsItem
from protest_impact.util import get

"""
Documentation: https://serpapi.com/search-api
Cost: 50$/month for 5000 searches -> ~0.01$/search
"""

load_dotenv()


def search(query: str, date: date, offset=0) -> NewsItem:
    results_per_page = 1000
    response = get(
        "https://serpapi.com/search",
        headers={},
        params={
            "q": f"{query} after:{date.isoformat()} before:{(date + timedelta(days=1)).isoformat()}",
            "gl": "de",
            "hl": "de",
            "lr": "lang_de",
            "google_domain": "google.de",
            "engine": "google",
            "tbm": "nws",
            "api_key": environ["SERPAPI_KEY"],
            "num": results_per_page,
            "start": offset,
        },
    )
    response.raise_for_status()
    json = response.json()
    if "news_results" not in json:
        return []
    json = json["news_results"]
    return [
        NewsItem(
            date=date, url=item["link"], title=item["title"], content=item["snippet"]
        )
        for item in json
    ]
