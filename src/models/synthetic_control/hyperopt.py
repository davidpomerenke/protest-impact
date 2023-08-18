import json
from itertools import product

import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from src.cache import cache
from src.models.synthetic_control import synthetic_control
from src.paths import models


@cache
def objective(params):
    rmses = []
    maes = []
    coefs = []
    for i in range(5):
        result = synthetic_control(
            target="media_combined_all",
            treatment="occ_protest",
            lags=params["lags"],
            steps=[6],
            cumulative=True,
            ignore_group=True,
            ignore_medium=True,
            positive_queries=True,
            add_features=["size", "ewm"],
            n_jobs=4,
            random_treatment_global=i,
        ).iloc[0]
        rmses.append(result.rmse)
        maes.append(result.mae)
        coefs.append(result.coef)
    params["rmse"] = np.mean(rmses)
    params["rmse_std"] = np.std(rmses)
    params["mae"] = np.mean(maes)
    params["mae_std"] = np.std(maes)
    params["bias"] = np.mean(coefs)
    params["bias_std"] = np.std(coefs)
    return params


def hyperopt(n_jobs=4):
    # lags = [7, 28, 3 * 28, 6 * 28, 365]
    # lags = [7, 14, 21, 28, 2 * 28, 3 * 28]
    # lags = [21, 24, 28, 35, 42, 49]
    # lags = range(20, 41)
    lags = range(20, 25)
    params = dict(
        lags=[[-i] for i in lags],
    )
    combinations = list(product(*params.values()))
    results = Parallel(n_jobs=n_jobs)(
        delayed(objective)(dict(zip(params.keys(), combination)))
        for combination in tqdm(combinations)
    )
    with open(models / "synthetic_control" / "params.json", "w") as f:
        json.dump(results, f, indent=2)
    best_result = min(results, key=lambda r: r["rmse"])
    return best_result
