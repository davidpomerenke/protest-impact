import json
from itertools import product

import matplotlib.pyplot as plt
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
    for i in range(4):
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


@cache
def hyperopt(n_jobs=4, show_progress=True):
    lags = [1, 7, 30, 90, 180, 270, 360]
    # lags = range(84, 169, 28)
    # lags = range(112, 133, 7)
    # lags = range(126, 142, 3)
    # lags = [136, 137]
    # lags = [138]
    # lags = [136, 137, 138, 139]
    # lags = [138, 140, 144]
    # lags = [120, 130, 140, 150, 160]
    params = dict(
        lags=[[-i] for i in lags],
    )
    combinations = list(product(*params.values()))
    results = Parallel(n_jobs=n_jobs)(
        delayed(objective)(dict(zip(params.keys(), combination)))
        for combination in tqdm(combinations, disable=not show_progress)
    )
    with open(models / "synthetic_control" / "params.json", "w") as f:
        json.dump(results, f, indent=2)
    best_result = min(results, key=lambda r: np.abs(r["bias"]))
    return results, best_result
