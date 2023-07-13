from pathlib import Path

import joblib

from protest_impact.types import NewsItem
from protest_impact.util.html import website_name

project_root = Path(__file__).parent.parent.parent


def fulltext_path(story: NewsItem) -> Path:
    return (
        project_root
        / "data"
        / "news"
        / "fulltext"
        / website_name(story.url)
        / f"{story.date.year}-{story.date.month:02d}-{story.date.day:02d}"
        / f"{joblib.hash(story.url)}.txt"
    )
