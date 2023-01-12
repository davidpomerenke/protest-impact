from datetime import date
from os import environ
from time import sleep

from dotenv import load_dotenv

from protest_impact.types import NewsItem
from protest_impact.util import get

"""
Documentation: https://docs.microsoft.com/en-us/bing/search-apis/bing-web-search/overview
Cost: 4$/1000 searches -> 0.004$/search
"""

load_dotenv()


def search(query: str, date: date, offset=0) -> NewsItem:
    results_per_page = 100
    response = get(
        "https://api.bing.microsoft.com/v7.0/search",
        headers={"Ocp-Apim-Subscription-Key": environ["BING_API_KEY"]},
        params={
            "q": f"{query} language:de loc:de",
            "mkt": "de-DE",
            "responseFilter": "webPages",
            "freshness": date.isoformat(),
            "count": results_per_page,
            "offset": offset,
        },
    )
    response.raise_for_status()
    json = response.json()
    if not "webPages" in json:
        return []
    json = json["webPages"]
    results = [
        NewsItem(
            date=date, url=item["url"], title=item["name"], content=item["snippet"]
        )
        for item in json["value"]
    ]
    print(offset, json["totalEstimatedMatches"])
    if json["totalEstimatedMatches"] > offset + results_per_page:
        if offset >= 1000:
            raise Exception("too many results")
        sleep(0.5)
        results += search(query, date, offset + results_per_page)
    return list(set(results))
