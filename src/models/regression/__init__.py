import pandas as pd
import statsmodels.api as sm


def _regression(
    target: str, lagged_df: pd.DataFrame, treatment=None, no_controls: bool = False
):
    """
    For use with src/models/time_series.py.
    """
    y = lagged_df[[target]]
    X = lagged_df.drop(columns=[target])
    if no_controls:
        X = X[[c for c in X.columns if c.startswith("occ_")]]
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit(cov_type="HC3")
    coefs = decode_param_names(results.params, X.columns, data_type="coef")
    conf_int = decode_param_names(results.conf_int(), X.columns, data_type="conf_int")
    coefs = coefs.merge(conf_int, on=["predictor", "lag"])
    coefs = coefs[(coefs.lag == 0) & (coefs.predictor == treatment)]
    coefs = coefs.drop(columns=["lag"])
    return coefs


def decode_param_names(
    coefficients: list, feature_names: list[str], data_type: str
) -> pd.DataFrame:
    feature_names = [fn + "_lag0" if not "_lag" in fn else fn for fn in feature_names]
    if data_type == "conf_int":
        df = pd.DataFrame(
            {
                "feature_name": feature_names,
                "ci_lower": coefficients[0],
                "ci_upper": coefficients[1],
            }
        )
    elif data_type == "coef":
        df = pd.DataFrame({"feature_name": feature_names, "coef": coefficients})
    name_parts = df["feature_name"].str.rsplit("_lag", n=1)
    df["predictor"] = name_parts.str[0]
    df["lag"] = name_parts.str[1].astype(int)
    df = df.drop(columns=["feature_name"])
    return df
