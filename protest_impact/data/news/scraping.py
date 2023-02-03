import re
from datetime import date
from os import environ
from time import sleep
from urllib.parse import quote

import requests
from tqdm import tqdm

from protest_impact.types import NewsItem
from protest_impact.util import fulltext_path, html2text


def get_fulltext(metadata: NewsItem) -> NewsItem:
    """Returns the full text of a news article. If not cached, downloads and saves it."""
    path = fulltext_path(metadata)
    if path.exists():
        with open(path) as f:
            return NewsItem.from_str(f.read())
    return _download_and_save_fulltext(metadata)


def download_fulltext(metadata: NewsItem) -> None:
    """Downloads and saves the full text of a news article (if not cached)."""
    path = fulltext_path(metadata)
    if not path.exists():
        _download_and_save_fulltext(metadata)


def _download_and_save_fulltext(metadata: NewsItem) -> None:
    """Downloads and saves and returns the full text of a news article."""
    result = _download_fulltext(metadata)
    match result:
        case NewsItem(_):
            story = result
        case Exception():
            story = metadata
    path = fulltext_path(story)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(str(story))
    return story


def _download_fulltext(metadata: NewsItem) -> NewsItem | Exception:
    """Downloads and returns the full text of a news article."""
    sleep(0.2)
    try:
        response = requests.get(metadata.url)
    except Exception as e:
        print(e)
        response = requests.get(metadata.url, allow_redirects=False)
    if response.status_code == 200:
        html = response.text
    else:
        # return Exception(metadata.url)
        response = requests.get(
            f"http://archive.org/wayback/available?url={quote(metadata.url)}"
        )
        if response.status_code != 200:
            return Exception(metadata.url)
        response.raise_for_status()
        result = response.json()["archived_snapshots"]
        if not "closest" in result or not result["closest"]["available"]:
            return Exception(metadata.url)
        response = requests.get(result["closest"]["url"])
        tqdm.write("😴")
        sleep(4)
        if response.status_code != 200:
            return Exception(metadata.url)
        html = response.text
    title, text = html2text(html)
    return NewsItem(
        url=metadata.url,
        date=metadata.date,
        title=title,
        content=text,
    )
