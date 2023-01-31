from datetime import date, timedelta
from os import environ
from time import sleep

from dateutil import parser
from dotenv import load_dotenv

from protest_impact.data.news.config import media_ids
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
    query: str,
    date: date,
    end_date: date = None,
    newspaper: str = None,
    last_processed_stories_id: int = 0,
    threshold: int = None,
) -> NewsItem:
    media_id = media_ids[newspaper]
    end_date_ = end_date or (date + timedelta(days=1))
    results_per_page = 1000
    print(
        get._get_argument_hash(
            "https://api.mediacloud.org/api/v2/stories_public/list/",
            params={
                "last_processed_stories_id": last_processed_stories_id,
                "rows": results_per_page,
                "q": query,
                "fq": [
                    f"media_id:{media_id}" if media_id else "",
                    # "tags_id_media:34412409",
                    f"publish_date:[{date.isoformat()}T00:00:00Z TO {end_date_.isoformat()}T00:00:00Z]",
                ],
                "key": environ["MEDIACLOUD_API_KEY"],
            },
            headers={"Accept": "application/json"},
        )
    )
    # sleep(1)
    response = get(
        "https://api.mediacloud.org/api/v2/stories_public/list/",
        params={
            "last_processed_stories_id": last_processed_stories_id,
            "rows": results_per_page,
            "q": query,
            "fq": [
                f"media_id:{media_id}" if media_id else "",
                # "tags_id_media:34412409",
                f"publish_date:[{date.isoformat()}T00:00:00Z TO {end_date_.isoformat()}T00:00:00Z]",
            ],
            "key": environ["MEDIACLOUD_API_KEY"],
        },
        headers={"Accept": "application/json"},
    )
    if response.status_code != 200:
        print(response.text)
    response.raise_for_status()
    json = response.json()
    results = [
        NewsItem(
            date=parser.parse(item["publish_date"]).date(),
            url=item["url"],
            title=item["title"],
            content="",
        )
        for item in json
        if item["publish_date"] is not None
    ]
    if len(results) >= 0.9 * results_per_page:
        last_processed_stories_id = json[-1]["processed_stories_id"]
        # print(f"last_processed_stories_id: {last_processed_stories_id}")
        results += search(query, date, end_date, newspaper, last_processed_stories_id)
    return list(set(results))
