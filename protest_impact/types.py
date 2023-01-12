from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class NewsItem:
    date: date
    url: str
    title: str
    content: str
