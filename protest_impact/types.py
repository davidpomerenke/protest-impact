from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class NewsItem:
    date: date
    url: str
    title: str
    content: str

    def __str__(self):
        return f"{self.title}\n\n{self.date.isoformat()}\n\n{self.content}\n\n{self.url}\n\n"

    @staticmethod
    def from_str(s: str) -> "NewsItem | Exception":
        parts = s.split("\n\n")
        if len(parts) < 4:
            return Exception(s)
        return NewsItem(
            date=date.fromisoformat(parts[1]),
            url=parts[-2],
            title=parts[0],
            content="\n\n".join(parts[2:-2]),
        )
