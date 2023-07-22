import holidays
import pandas as pd
from protest_impact.data import german_regions
from protest_impact.util import cache


@cache
def _all_holidays():
    german_holidays = dict()
    for region in german_regions:
        german_holidays[region["name"]] = [
            pd.to_datetime(d)
            for d in holidays.Germany(
                years=range(2020, 2023), subdiv=region["code"]
            ).keys()
        ]
    return german_holidays


def get_holidays(idx: pd.DatetimeIndex, region: str) -> pd.Series:
    return idx.to_series().isin(_all_holidays()[region])
