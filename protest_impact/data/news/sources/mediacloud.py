from datetime import date, timedelta
from os import environ

from dotenv import load_dotenv

from protest_impact.types import NewsItem
from protest_impact.util import get

"""
Documentation:
 - https://github.com/mediacloud/backend/blob/master/doc/api_2_0_spec/api_2_0_spec.md
 - https://mediacloud.org/support/query-guide/
Cost: free
"""

load_dotenv()


def search(query: str, date: date, offset=0) -> NewsItem:
    results_per_page = 100
    response = get(
        "https://api.mediacloud.org/api/v2/stories_public/list/",
        params={
            "last_processed_stories_id": offset,
            "rows": results_per_page,
            "q": query,
            "fq": [
                "tags_id_media:34412409",
                f"publish_date:[{date.isoformat()}T00:00:00Z TO {(date + timedelta(days=1)).isoformat()}T00:00:00Z]",
            ],
            "key": environ["MEDIACLOUD_API_KEY"],
        },
        headers={"Accept": "application/json"},
    )
    response.raise_for_status()
    json = response.json()
    return [
        NewsItem(date=date, url=item["url"], title=item["title"], content="")
        for item in json
    ]
