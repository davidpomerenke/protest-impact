from functools import partial

from protest_impact.synthetic_control.models import (
    GAM,
    AutoRegressor,
    BayesianStructuralTimeSeries,
    DistanceWeightsEstimator,
    MeanEstimator,
    NeuralEstimator,
    SociodemographicWeightsEstimator,
)
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge


def get_model(config):
    if config.method == "mean":
        model = MeanEstimator()
    elif config.method == "distance":
        model = partial(
            DistanceWeightsEstimator,
            event_region=config.admin1,
            inverse=config.distance__inverse,
        )
    elif config.method == "sociodemographic":
        model = partial(
            SociodemographicWeightsEstimator,
            method=config.sociodemographic__method,
            event_region=config.admin1,
            sum_to_one=config.sociodemographic__sum_to_one,
        )
    elif config.method in ["linear_regression", "lasso", "ridge"]:
        args = {}
        # learn interpretable coefficients (factor model)
        if config.interpretable:
            args["fit_intercept"] = True
            args["positive"] = True
        if config.method == "linear_regression":
            model = LinearRegression(**args)
        elif config.method == "ridge":
            model = Ridge(**args)
        elif config.method == "lasso":
            model = Lasso(alpha=config["alpha"], **args)
    elif config.method in ["random_forest", "gradient_boosting"]:
        args = dict(
            n_estimators=config.n_estimators,
            max_features=config.max_features,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
        )
        if config.method == "random_forest":
            model = RandomForestRegressor(**args)
        elif config.method == "gradient_boosting":
            model = GradientBoostingRegressor(**args)
    elif config.method == "gam":
        model = GAM(
            n_splines=config.n_splines, spline_order=config.spline_order, lam=config.lam
        )
    elif config.method == "bayesian_structural_time_series":
        model = BayesianStructuralTimeSeries(fit_method=config.fit_method)
    elif config.method == "neural":
        assert config.prediction_interval is not None
        model = NeuralEstimator(
            horizon=config.prediction_interval,
            agg_weekly=config.agg_weekly,
            model_name=config.neural__method,
            max_steps=config.neural__max_steps,
        )

    if config.method in [
        "linear_regression",
        "ridge",
        "lasso",
        "random_forest",
        "gradient_boosting",
    ]:
        if config.use_autoregressor:
            if config.agg_weekly or "interpretable" in config and config.interpretable:
                # a) would keep the weights for the lags all positive
                # with b) there are not necessarily enough previous steps
                config.pruned = True
                return config
            model = AutoRegressor(regressor=model, lags=config.lags)

    # log the category of the model
    if config.method in ["mean", "distance", "sociodemographic"]:
        method_group = "grounded"
    if config.method in [
        "random_forest",
        "gradient_boosting",
        "gam",
        "bayesian_structural_time_series",
        "neural",
    ]:
        method_group = "nonlinear"
    if config.method in ["linear_regression", "ridge", "lasso"]:
        if not config["use_autoregressor"] and config["interpretable"]:
            method_group = "linear_weighted_sum"
        elif not config["use_autoregressor"]:
            method_group = "linear"
        elif config["use_autoregressor"]:
            method_group = "linear_autoregressive"
    config.method_group = method_group

    return model, config
