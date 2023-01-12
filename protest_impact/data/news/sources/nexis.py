from datetime import date, timedelta
from os import environ
from time import sleep

import requests
from dotenv import load_dotenv

from protest_impact.types import NewsItem
from protest_impact.util import get

"""
Documentation: https://help-lexisnexis-com.ezproxy.ub.unimaas.nl/Flare/nexisuni/US/nl_NL/Content/field/searchwithtermstips.htm
Cost: free via university library
"""

load_dotenv()


def search(query: str, date: date, offset=0) -> NewsItem:
    results_per_page = 1000
    response = get(
        "http://advance.lexis.com.ezproxy.ub.unimaas.nl/api/search",
        headers={},
        params={
            "q": f"{query}",
            "collection": "cases",
            "qlang": "bool",
            "context": "1516831",
        },
    )
    # response.raise_for_status()
    return response.text
    json = response.json()
    return [
        NewsItem(
            date=date, url=item["link"], title=item["title"], content=item["snippet"]
        )
        for item in json
    ]
