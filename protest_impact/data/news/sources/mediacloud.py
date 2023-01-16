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


def search(
    query: str, date: date, media_id: int = None, last_processed_stories_id: int = 0
) -> NewsItem:
    results_per_page = 100
    response = get(
        "https://api.mediacloud.org/api/v2/stories_public/list/",
        params={
            "last_processed_stories_id": last_processed_stories_id,
            "rows": results_per_page,
            "q": query,
            "fq": [
                f"media_id:{media_id}" if media_id else "",
                # "tags_id_media:34412409",
                f"publish_date:[{date.isoformat()}T00:00:00Z TO {(date + timedelta(days=1)).isoformat()}T00:00:00Z]",
            ],
            "key": environ["MEDIACLOUD_API_KEY"],
        },
        headers={"Accept": "application/json"},
    )
    response.raise_for_status()
    json = response.json()
    results = [
        NewsItem(date=date, url=item["url"], title=item["title"], content="")
        for item in json
    ]
    if len(results) == results_per_page:
        last_processed_stories_id = json[-1]["processed_stories_id"]
        results += search(query, date, media_id, last_processed_stories_id)
    return list(set(results))
