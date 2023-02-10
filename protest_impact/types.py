from dataclasses import dataclass
from datetime import date
from typing import Optional

from dateutil import parser


@dataclass(frozen=True)
class NewsItem:
    date: date
    url: str
    title: str
    content: Optional[str] = None

    def __str__(self):
        return f"{self.title}\n\n{self.date.isoformat()}\n\n{self.content}\n\n{self.url}\n\n"

    @staticmethod
    def from_str(s: str) -> "NewsItem | Exception":
        parts = s.split("\n\n")
        if len(parts) < 4:
            return Exception(s)
        return NewsItem(
            date=parser.parse(parts[1], yearfirst=True, dayfirst=False),
            # date=date.fromisoformat(parts[1]),
            url=parts[-2],
            title=parts[0],
            content="\n\n".join(parts[2:-2]),
        )

    @staticmethod
    def from_dict(d: dict) -> "NewsItem | Exception":
        return NewsItem(
            date=parser.parse(d["publish_date"], yearfirst=True, dayfirst=False),
            url=d["url"],
            title=d["title"],
            content="",
        )

    def to_dict(self) -> dict:
        return {
            "publish_date": self.date.isoformat(),
            "url": self.url,
            "title": self.title,
            "content": self.content,
        }
