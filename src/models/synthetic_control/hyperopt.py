import sys
import warnings
from datetime import datetime
from functools import partial
from itertools import chain, product
from time import time

import numpy as np
import pandas as pd
from joblib import Memory, Parallel, delayed
from munch import munchify
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler
from tqdm.auto import tqdm

from src import project_root
from src.cache import cache
from src.data import neighbor_regions
from src.models.synthetic_control.data import get_data_parts
from src.models.synthetic_control.model_configs import get_model
from src.models.synthetic_control.models import MeanScaler, SleepingScaler

warnings.filterwarnings("error")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

SEED = 20230413
rng = np.random.default_rng(SEED)


@cache
def objective(config, return_time_series=False):
    config = munchify(config)
    config.seed = SEED
    config.pruned = False
    df = config.df
    # df.index = pd.RangeIndex(df.index[0], df.index[-1] + 1)

    if config.source == "dereko_scrape":
        config.pruned = True
        return config
        # HACK: filter protest data for dereko, but more scraping would be better
        if pd.Timestamp(config.event_date) < pd.Timestamp("2020-07-01"):
            config.pruned = True
            return config

    # 1. data preprocessing

    if config.ignore_neighbor_regions:
        nr = set(neighbor_regions[config.admin1]).intersection(df.columns)
        df = df.drop(columns=nr)
    if len(df.columns) < 4:
        config.pruned = True
        return config

    config.n_theoretical_control_regions = len(df.columns) - 1

    if config.agg_weekly:
        for col in df.columns[1:]:
            df[f"{col}_weekly"] = df[col].rolling(7).mean()

    # normalization to make the different time series more comparable
    normalizer = dict(
        none=SleepingScaler(),  # no normalization, keeps the same (positive) values
        mean=MeanScaler(),  # mean normalization, keeps the values positive
        standard=StandardScaler(),  # z-scores, introduces negative values
        robust=RobustScaler(),  # robust z-scores, introduces negative values
        power=PowerTransformer(),  # makes the distribution more normal, introduces negative values
    )[config.normalize]

    # 2. Model selection and hyperparameter optimization

    # initialize the respective model and set its hyperparameters
    model, config = get_model(config)

    # 3. training

    cv = TimeSeriesSplit(
        n_splits=5,
        max_train_size=config.training_interval,
        test_size=config.prediction_interval,
    )

    df = df.rename(columns={df.columns[0]: "y"})

    maes, rmses, rrmses = [], [], []
    for split, (train_index, test_index) in enumerate(reversed(list(cv.split(df)))):
        df_train = df.iloc[train_index]
        df_test = df.iloc[test_index]
        X_train = df_train.drop(columns=["y"])
        y_train = df_train["y"]
        X_test = df_test.drop(columns=["y"])
        y_test = df_test["y"]

        if split == 0:
            assert X_train.iloc[0].name == -config.training_interval
            assert X_train.iloc[-1].name == -1
            assert X_test.iloc[0].name == 0
            assert X_test.iloc[-1].name == config.prediction_interval - 1

        # remove control regions with too little data
        X_train = X_train.loc[:, X_train.median() >= 1]
        X_test = X_test[X_train.columns]
        assert X_train.shape[1] == X_test.shape[1]

        if X_train.shape[1] == 0 or y_train.median() < 1:
            config.pruned = True
            return config

        if config.method in ["distance", "sociodemographic"]:
            _model = model(control_regions=X_train.columns)
        else:
            _model = model

        normalizer.fit(X_train.values)
        X_train = normalizer.transform(X_train.values)
        X_test = normalizer.transform(X_test.values)
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)

        tt = TransformedTargetRegressor(regressor=_model, transformer=normalizer)
        tt.fit(X_train, y_train.values)
        y_pred = tt.predict(X_test)
        y_test = y_test.values

        if config.normalize == "power":
            # inverse power transform may create nans
            y_pred = np.nan_to_num(y_pred)

        # 4. evaluation

        if split == 0:  # we are in the test split
            # calculate impact for each of the 4 weeks after the protest
            difference = y_test - y_pred
            n_prediction_weeks = config.prediction_interval // 7
            for w in range(n_prediction_weeks):
                impact = difference[w * 7 : (w + 1) * 7].sum()
                mean = y_pred[w * 7 : (w + 1) * 7].sum()
                config[f"impact_week_{w}"] = impact
                relative_impact = impact / mean if mean > 0 else np.nan
                config[f"relative_impact_week_{w}"] = relative_impact
            for d in range(0, config.prediction_interval):
                config[f"impact_day_{d}"] = difference[d]
                relative_impact = difference[d] / y_pred[d] if y_pred[d] > 0 else np.nan
                config[f"relative_impact_day_{d}"] = relative_impact
            if return_time_series:
                # create a time-series dataframe
                # storing this dataframe for every parameter combination is too much
                # so we only store it for the best parameter combination
                X_all = np.concatenate([X_train, X_test])
                X_sym = X_all[-config.prediction_interval * 2 :]
                y_pred_sym = tt.predict(X_sym)
                y_true_all = np.concatenate([y_train, y_test])
                y_true_sym = y_true_all[-config.prediction_interval * 2 :]
                comparison_df = pd.DataFrame(
                    {"true": y_true_sym, "predicted": y_pred_sym},
                    index=df.iloc[-len(X_sym) :].index,
                )
                config.comparison_df = comparison_df.to_json()
                comparison_df_normalized = pd.DataFrame(
                    {
                        "true": tt.transformer_.transform(
                            y_true_sym.reshape(-1, 1)
                        ).squeeze(),
                        "predicted": tt.transformer_.transform(
                            y_pred_sym.reshape(-1, 1)
                        ).squeeze(),
                    },
                    index=df.iloc[-len(X_sym) :].index,
                )
                config.comparison_df_normalized = comparison_df_normalized.to_json()
            config.n_control_regions = X_train.shape[1]
        else:  # we are in one of the validation splits
            # # the metrics are computed on a weekly basis
            # # because there is too much randomness on a daily basis
            # y_pred = y_pred.reshape(-1, 7).sum(axis=1)
            # y_test = y_test.reshape(-1, 7).sum(axis=1)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mean = y_test.mean()
            rrmse = rmse / mean if mean > 0 else np.nan
            maes.append(mae)
            rmses.append(rmse)
            rrmses.append(rrmse)
    config.mae = np.mean(maes)
    config.rmse = np.mean(rmses)
    config.rrmse = np.mean(rrmses)
    return config


common_grounded_params = dict(
    ignore_neighbor_regions=[True],
    agg_weekly=[False],
    normalize=["standard"],
    training_interval=[28 * 3],
)
common_params = dict(
    ignore_neighbor_regions=[True],
    agg_weekly=[False],
    normalize=["standard"],
    training_interval=[28, 28 * 3, 28 * 12],
)
autoregressor_params = dict(
    use_autoregressor=[False],
    lags=[14],
)
search_spaces = [
    dict(
        method=["mean"],
        **common_grounded_params,
        disable=[],
    ),
    dict(
        method=["distance"],
        distance__inverse=[False, True],
        **common_grounded_params,
        disable=[],
    ),
    dict(
        method=["sociodemographic"],
        sociodemographic__method=["nmf"],
        sociodemographic__sum_to_one=[True],
        **common_grounded_params,
        # disable=[],
    ),
    dict(  # TODO: the commented parameters still need to be evaluated
        method=["lasso"],  # lasso, ridge
        ignore_neighbor_regions=[True],
        interpretable=[True],
        alpha=[0.1],  # 0.01, 0.1, 0.5
        agg_weekly=[False],  # True, False
        normalize=["standard"],  # mean, standard
        training_interval=[28 * 6],  # 28, 28 * 3, 28 * 12
        **autoregressor_params,
    ),
    dict(
        method=["random_forest"],
        n_estimators=[10, 50, 100, 200],
        max_features=[1.0],
        max_depth=[None, 5, 20],
        min_samples_split=[2],
        min_samples_leaf=[1],
        use_autoregressor=[False],
        lags=[14],
        **common_params,
        disable=[],
    ),
    dict(
        method=["gradient_boosting"],
        n_estimators=[10, 50, 100, 200],
        max_features=[1.0],
        max_depth=[None, 5, 20],
        min_samples_split=[2],
        min_samples_leaf=[1],
        use_autoregressor=[False],
        lags=[14],
        **common_params,
        disable=[],
    ),
    dict(
        method=["gam"],
        n_splines=[5, 10],
        spline_order=[1, 2, 3],
        lam=[1e-4, 1e-2, 1],
        **common_params,
        disable=[],
    ),
    dict(
        normalize=["none"],
        method=["bayesian_structural_time_series"],
        fit_method=["vi"],  # vi, hmc
        agg_weekly=[False],
        training_interval=[28 * 3],
        disable=[],
    ),
    dict(
        normalize=["none"],
        method=["bayesian_structural_time_series"],
        fit_method=["vi"],  # vi, hmc
        agg_weekly=[True],
        training_interval=[28 * 12],
        disable=[],
    ),
    dict(
        method=["neural"],
        neural__method=["NBEATSx"],  # NHITS, NBEATS
        neural__max_steps=[500],
        agg_weekly=[True],
        normalize=["none"],
        training_interval=[28 * 12],
        disable=[],
    ),
]

memory = Memory(location=".cache", verbose=0)


def search_space(search_space, metadata):
    results = []
    for combi in product(*search_space.values()):
        combi = dict(zip(search_space.keys(), combi))
        config = dict(combi, **metadata)
        start_ts = time()
        result = objective(config)
        result.date = datetime.now().date().isoformat()
        result.start_ts = start_ts
        result.end_ts = time()
        result.duration = result.end_ts - start_ts
        results.append(result)
    return results


def search(search_spaces, metadata):
    results = [search_space(s, metadata) for s in search_spaces]
    return list(chain(*results))


def search_parallel(search_spaces, metadata_list, n_jobs=1):
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(search)(search_spaces, metadata) for metadata in tqdm(metadata_list)
    )
    return list(chain(*results))


if __name__ == "__main__":
    n = 1
    if len(sys.argv) > 1:
        if sys.argv[1] == "clear":
            search.clear()
            search_space.clear()
            objective.clear()
            exit()
        elif sys.argv[1] == "n":
            n = int(sys.argv[2])
    metadata = get_data_parts(["all"], 1, n_jobs=n)
    # metadata = get_data_parts(["bot", "mid", "top"], 0.05, n_parallel=n)
    results = search_parallel(search_spaces, metadata, n_jobs=n)
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        project_root / "hyperopt" / "results.csv", index=False, compression="gzip"
    )
