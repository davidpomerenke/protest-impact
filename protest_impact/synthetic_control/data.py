import json
from datetime import date
from itertools import chain, product

import numpy as np
import pandas as pd
from joblib import Memory, Parallel, delayed
from joblib.hashing import hash
from munch import Munch, munchify
from tqdm.auto import tqdm

from protest_impact.data.news.dereko.dereko import get_scraped_entries
from protest_impact.data.protests import (
    aggregate_protests,
    get_climate_protests,
    get_climate_queries,
)
from protest_impact.synthetic_control import (
    filter_regions,
    get_regional_counts_for_protest,
)
from protest_impact.util import cache, project_root

SEED = 20230429
rng = np.random.default_rng(SEED)

dereko_entries = dict()
for t in ["climate", "climate_and_protest", "climate_not_protest"]:
    # dereko_entries[t] = get_scraped_entries(discourse_type=t)
    dereko_entries[t] = None

queries = get_climate_queries()


@cache
def get_data(parameters):
    """
    version = 0
    """
    p = munchify(parameters)
    protests = get_climate_protests(
        start_date=p.start_date, end_date=p.end_date, groups=p.protest_groups
    )
    if p.aggregate_protests:
        protests = aggregate_protests(protests)
    query = queries[p.discourse_type]
    metadata = []
    for event in tqdm(protests.to_dict(orient="records")):
        args = Munch(**parameters, **event)
        df = get_regional_counts_for_protest(
            query_str=query[0],
            query_func=query[1],
            region=args.admin1,
            event_date=args.event_date,
            source=p.source,
            n_days_train=None,
            n_days_predict=p.prediction_interval,
            dereko_entries=dereko_entries[p.discourse_type],
        )
        if df is None:
            continue
        df = filter_regions(
            df=df,
            region=args.admin1,
            event_date=args.event_date,
            reference_events=protests,
            n_days_protest_free_pre=p.n_days_protest_free_pre,
            n_days_protest_free_post=p.n_days_protest_free_post,
            min_control_regions=0,
            min_count=0,
        )
        if df is None:
            continue
        args.df = df
        metadata.append(args)
    return metadata


data_config_space = dict(
    start_date=[date(2020, 1, 1)],
    end_date=[date(2022, 12, 31)],
    protest_groups=[["fff", "alg", "xr", "eg"]],
    source=["mediacloud"],  # "dereko_scrape"],
    discourse_type=["climate", "climate_not_protest", "climate_and_protest"],
    prediction_interval=[28],
    n_days_protest_free_pre=[1],
    n_days_protest_free_post=[1],
    aggregate_protests=[True],
)


def get_all_data(n_jobs=1):
    combinations = product(*data_config_space.values())
    combinations = [dict(zip(data_config_space.keys(), c)) for c in combinations]
    # use joblib to parallelize
    data = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(get_data)(combination) for combination in tqdm(combinations)
    )
    return list(chain(*data))


def get_data_parts(parts, p, n_jobs=1):
    """
    parts: list of strings, one of ["top", "mid", "bot", "all"]
    p: float, proportion of data to include in each part
    """
    overview = get_all_data(n_jobs=n_jobs)
    by_size = [d for d in overview if d["size"] > 0]
    by_size = sorted(by_size, key=lambda x: x["size"], reverse=True)
    l = len(by_size)
    parts_dict = dict(
        top=by_size[: round(l * p)],
        mid=by_size[round(l * (0.5 - p / 2)) : round(l * (0.5 + p / 2))],
        bot=by_size[round(l * (1 - p)) :],
        all=overview,
    )
    return list(chain(*[parts_dict[p] for p in parts]))
