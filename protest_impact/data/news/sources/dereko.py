from datetime import date
from os import environ

import pandas as pd
from dotenv import load_dotenv

from protest_impact.types import NewsItem
from protest_impact.util import get, html2text, log
from protest_impact.util import project_root

"""
Documentation: https://korap.ids-mannheim.de/doc/api#page-top
Cost: free
Authentication HowTo: https://github.com/KorAP/Kustvakt/issues/492
"""

load_dotenv()


def load_dereko_corpora():
    # returns a dict for every year with a list of all corpora availabele for that year
    df = pd.read_csv(project_root / "datasets" / "dereko_corpora.csv", sep=";")
    df = df[df["Leitmedium"] == 1]
    corpora = {}
    for year in range(1950, 2030):
        corpora[year] = df[(df["von"] <= year) & (df["bis"] >= year)]["Sigle"].tolist()
    return corpora


def search(
    query: str, date: date, end_date: date = None, corpora=None, offset=0
) -> NewsItem:
    end_date_ = end_date or date
    if corpora is None:
        all_corpora = load_dereko_corpora()
        corpora = [
            all_corpora[date.year] for date in range(date.year, end_date_.year + 1)
        ]
    corpora = [sigle.upper() + str(date.year)[-2:] for sigle in corpora]  # TODO
    results_per_page = 100
    if len(corpora) == 0:
        return []
    corpora_query = " | ".join([f"corpusSigle={corpus}" for corpus in corpora])
    date_query = f"creationDate since {date.isoformat()} & creationDate until {end_date_.isoformat()}"
    print(query, date_query, corpora_query)
    res = get(
        url="https://korap.ids-mannheim.de/api/v1.0/search",
        headers={
            "Authorization": "Bearer " + environ["DEREKO_ACCESS_TOKEN"],
        },
        params={
            "q": query,
            "ql": "poliqarp",
            "context": "500-token,500-token",  # more tokens are not possible
            "cq": f"({corpora_query}) & {date_query}",
            "page": 1,
        },
    )
    res.raise_for_status()
    json = res.json()
    return [
        NewsItem(
            date=date,
            url=item["pubPlace"].replace("URL:", ""),
            title=item["title"],
            content=html2text(item["snippet"])[1],
        )
        for item in json["matches"]
    ]
