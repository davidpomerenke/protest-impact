import numpy as np
import pandas as pd

from protest_impact.data import german_regions


def get_placebo_events(
    n, rng, start="2020-04-01", end="2022-08-31", region_weights=None
):
    random_events = []
    for _ in range(n):
        random_date = pd.to_datetime(rng.choice(pd.date_range(start, end)))
        _region_weights = []
        for region in german_regions:
            if region_weights is not None and region["name"] in region_weights:
                _region_weights.append(region_weights[region["name"]])
            else:
                _region_weights.append(0)
        _region_weights = np.array(_region_weights)
        _region_weights = _region_weights / _region_weights.sum()
        random_region = rng.choice(german_regions, p=_region_weights)["name"]
        random_events.append(
            {
                "event_date": random_date,
                "admin1": random_region,
            }
        )
    return pd.DataFrame(random_events)
