import re
from datetime import date
from os import environ
from urllib.parse import quote

from protest_impact.types import NewsItem
from protest_impact.util import get, html2text


def get_story(url: str, date: date) -> NewsItem | None:
    response = get(url)
    if response.status_code == 200:
        html = response.text
    else:
        response = get(f"http://archive.org/wayback/available?url={quote(url)}")
        response.raise_for_status()
        result = response.json()["archived_snapshots"]
        if not "closest" in result:
            return None
        if not result["closest"]["available"]:
            return None
        response = get(result["closest"]["url"])
        response.raise_for_status()
        html = response.text
    title, text = html2text(html)
    return NewsItem(
        url=url,
        date=date,
        title=title,
        content=text,
    )
