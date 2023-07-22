from itertools import product
from time import time

import numpy as np
import pandas as pd
import statsmodels.api as sm
from dateutil import relativedelta
from linearmodels import IV2SLS, IVGMM, IVLIML
from linearmodels.iv.model import _IVModelBase
from linearmodels.iv.results import IVResults
from munch import Munch
from protest_impact.data.news import get_regional_count_df
from protest_impact.data.protests import (
    aggregate_protests,
    get_climate_protests,
    get_climate_queries,
)
from protest_impact.data.weather import get_longterm_weather, get_weather
from protest_impact.instrumental_variables.auto import AutoRandomForestRegressor
from protest_impact.util import cache, project_root
from sklearn.ensemble import RandomForestRegressor


@cache
def get_covid_data_(fill_values=True) -> pd.DataFrame:
    df = pd.read_csv(
        project_root / "datasets/covid/changes-visitors-covid.csv", parse_dates=["Day"]
    )
    df = df.query("Entity == 'Germany'")
    df = df.drop(columns=["Code", "Entity"])

    if fill_values:
        # fill days at the start with zeros
        # add row for 2020-01-01:
        df = df.append({"Day": pd.to_datetime("2020-01-01")}, ignore_index=True)
        df = df.fillna(0)
        df = df.set_index("Day").asfreq("D", fill_value=0).reset_index().fillna(0)
        # fill days at the end with last value
        # add row for 2022-12-31:
        df = df.append({"Day": pd.to_datetime("2023-01-01")}, ignore_index=True)
        df = df.set_index("Day").asfreq("D", method="ffill").reset_index().fillna(0)
        # drop the last row again
    df = df.iloc[:-1]
    df = df.rename(columns={"Day": "date"})
    df = df.set_index("date")
    return df


@cache
def get_covid_data(protests) -> pd.DataFrame:
    df = get_covid_data_()
    cols = df.columns
    df = protests.join(df, on="event_date", lsuffix="xyz")
    df = df[cols].fillna(0)
    return df


@cache
def get_weather_data(protests: pd.DataFrame) -> pd.DataFrame:
    df = protests.apply(lambda x: get_weather(x["location"], x["event_date"]), axis=1)
    return df


@cache
def get_climate_data(protests: pd.DataFrame) -> pd.DataFrame:
    df = protests.apply(
        lambda x: get_longterm_weather(x["location"], x["event_date"]), axis=1
    )
    df.columns = [f"longterm_{col}" for col in df.columns]
    return df


def _get_discourse(row, query, day_n) -> int:
    ts = get_regional_count_df(
        region=row["admin1"], query_string=query, source="mediacloud"
    )
    date_ = row["event_date"] + pd.Timedelta(days=day_n)
    pre_week = ts[(ts["date"] >= date_ - pd.Timedelta(days=7)) & (ts["date"] < date_)]
    c = pre_week["count"].mean()
    post = ts[ts["date"] == date_]
    return post["count"].values[0]


@cache
def get_discourse(protests, query, day_n) -> pd.Series:
    df = protests.apply(lambda x: _get_discourse(x, query, day_n), axis=1)
    df.name = "discourse_change"
    return df


class OLS(_IVModelBase):
    def __init__(self, dependent, exog, endog, instruments):
        self.dependent = dependent
        self.exog = exog
        self.endog = endog
        self.instruments = instruments

    def fit(self):
        X2 = pd.concat([self.endog, self.exog], axis=1)
        X2 = X2.rename(columns={"size": "size_predicted"})
        X2 = X2.rename(columns={"log10(size)": "size_predicted"})
        X2 = X2.rename(columns={"log(log(size))": "size_predicted"})
        X2 = sm.add_constant(X2)
        y2 = self.dependent
        model2 = sm.OLS(y2, X2)
        results2 = model2.fit()
        impact_pred = results2.predict(X2)
        return Munch(
            second_stage=Munch(
                results=results2, model=model2, X=X2, y=y2, y_hat=impact_pred
            ),
        )


class Manual_2SLS(_IVModelBase):
    def __init__(self, dependent, exog, endog, instruments):
        self.dependent = dependent
        self.exog = exog
        self.endog = endog
        self.instruments = instruments

    def fit(self):
        # stage 1
        X1 = pd.concat([self.instruments, self.exog], axis=1)
        y1 = self.endog
        X1 = sm.add_constant(X1)
        model1 = sm.OLS(y1, X1)
        results1 = model1.fit()
        endog_pred = results1.predict(X1)
        endog_pred.name = "size_predicted"

        # stage 2
        X2 = pd.concat([endog_pred, self.exog], axis=1)
        X2 = sm.add_constant(X2)
        y2 = self.dependent
        model2 = sm.OLS(y2, X2)
        results2 = model2.fit()
        impact_pred = results2.predict(X2)
        return Munch(
            first_stage=Munch(
                results=results1, model=model1, X=X1, y=y1, y_hat=endog_pred
            ),
            second_stage=Munch(
                results=results2, model=model2, X=X2, y=y2, y_hat=impact_pred
            ),
        )


class RFLR(_IVModelBase):
    def __init__(self, dependent, exog, endog, instruments):
        self.dependent = dependent
        self.exog = exog
        self.endog = endog
        self.instruments = instruments

    def fit(self):
        # stage 1
        X1 = pd.concat([self.instruments, self.exog], axis=1)
        y1 = self.endog
        X1 = sm.add_constant(X1)
        model1 = RandomForestRegressor().fit(X1, y1)
        endog_pred = pd.Series(model1.predict(X1), name="size_predicted")

        # stage 2
        X2 = pd.concat([endog_pred, self.exog], axis=1)
        y2 = self.dependent
        X2 = sm.add_constant(X2)
        model2 = sm.OLS(y2, X2)
        results2 = model2.fit()
        impact_pred = results2.predict(X2)

        return Munch(
            first_stage=Munch(model=model1, X=X1, y=y1, y_hat=endog_pred),
            second_stage=Munch(model=model2, X=X2, y=y2, y_hat=impact_pred),
        )


class RFRF(_IVModelBase):
    def __init__(self, dependent, exog, endog, instruments):
        self.dependent = dependent
        self.exog = exog
        self.endog = endog
        self.instruments = instruments

    def fit(self):
        # stage 1
        X1 = pd.concat([self.instruments, self.exog], axis=1)
        y1 = self.endog
        X1 = sm.add_constant(X1)
        model1 = RandomForestRegressor().fit(X1, y1)
        endog_pred = pd.Series(model1.predict(X1), name="size_predicted")

        # stage 2
        X2 = pd.concat([endog_pred, self.exog], axis=1)
        y2 = self.dependent
        X2 = sm.add_constant(X2)
        model2 = RandomForestRegressor().fit(X2, y2)
        impact_pred = model2.predict(X2)

        return Munch(
            first_stage=Munch(model=model1, X=X1, y=y1, y_hat=endog_pred),
            second_stage=Munch(model=model2, X=X2, y=y2, y_hat=impact_pred),
        )


class AutoRFRF(_IVModelBase):
    def __init__(self, dependent, exog, endog, instruments):
        self.dependent = dependent
        self.exog = exog
        self.endog = endog
        self.instruments = instruments

    def fit(self):
        # stage 1
        X1 = pd.concat([self.instruments, self.exog], axis=1)
        y1 = self.endog
        X1 = sm.add_constant(X1)
        model1 = AutoRandomForestRegressor().fit(X1, y1)
        endog_pred = pd.Series(model1.predict(X1), name="size_predicted")

        # stage 2
        X2 = pd.concat([endog_pred, self.exog], axis=1)
        y2 = self.dependent
        X2 = sm.add_constant(X2)
        model2 = AutoRandomForestRegressor().fit(X2, y2)
        impact_pred = model2.predict(X2)

        return Munch(
            first_stage=Munch(model=model1, X=X1, y=y1, y_hat=endog_pred),
            second_stage=Munch(model=model2, X=X2, y=y2, y_hat=impact_pred),
        )


def ols(dependent, exog, endog, instruments):
    return IV2SLS(dependent, pd.concat([exog, endog], axis=1), None, None)


def first_stage(dependent, exog, endog, instruments):
    return IV2SLS(endog, pd.concat([exog, instruments], axis=1), None, None)


@cache
def evaluate(p, protests):
    discourse_change = get_discourse(protests, p.query, p.day_n)
    dep = discourse_change
    endog = protests[p.endog]
    iv = p.method(dependent=dep, exog=p.exog, endog=endog, instruments=p.instr)
    res = iv.fit()
    return res


size_pairs = [(10**i, 10 ** (i + 1)) for i in range(1, 4)]
size_labels = [f"{m}<=size<{n}" for m, n in size_pairs]

space = dict(
    endog=[
        ["size"],
        ["log10(size)"],
        ["sqrt(size)"],
        ["size", "log10(size)", "sqrt(size)"],
        [*size_labels],
    ],
    instr=["weather_and_covid"],
    use_exog=[True],
    discourse=["climate_and_protest"],  # "climate_not_protest", "climate"
    day_n=[0],
    random_discourse=[False],
    method=["OLS", "IV2SLS", "IVGMM"],  # "IVLIML"
)


@cache
def get_instruments(protests):
    weather = get_weather_data(protests)
    climate = get_climate_data(protests)
    covid = get_covid_data(protests)
    instrs = pd.concat([weather, climate, covid], axis=1)
    return instrs


def instrumental_variables(protests: pd.DataFrame, space=space):
    protests = protests.query("size > 0").copy()
    protests = protests.reset_index()
    instrs = get_instruments(protests)
    exog = protests[
        [
            "sub_event_type",
            "assoc_actor_1",
            "actor2",
            "admin1",
            "weekday",
            # "n_protests",  # not sure if this is exogenous
        ]
    ]
    # turn all categorical variables into dummies
    for col in exog.columns:
        if exog[col].dtype == "object":
            dummies = pd.get_dummies(exog[col], prefix=col, drop_first=True)
            exog = pd.concat([exog, dummies], axis=1)
            exog = exog.drop(columns=[col])
    exog = sm.add_constant(exog)
    protests["log10(size)"] = np.log10(protests["size"])
    protests["log(log(size))"] = np.log(np.log((protests["size"])))
    protests["log2(size)"] = np.log2(protests["size"])
    protests["sqrt(size)"] = np.sqrt(protests["size"])
    for label, (m, n) in zip(size_labels, size_pairs):
        protests[label] = (protests["size"] >= m) & (protests["size"] < n)
    queries = get_climate_queries()

    results = dict()
    for vals in list(product(*space.values())):
        p = Munch(zip(space.keys(), vals))
        name = f"{p.discourse}\n{p.endog} ~ {p.instr}\n{'with exog' if p.use_exog else ''}\nday {p.day_n}\n{p.method}"
        p.query = queries[p.discourse][0]
        p.exog = exog if p.use_exog else pd.Series(np.ones(len(exog)), name="const")
        p.instr = instrs[p.instr]
        p.method = dict(
            OLS=OLS,
            Manual_2SLS=Manual_2SLS,
            IV2SLS=IV2SLS,
            IVGMM=IVGMM,
            IVLIML=IVLIML,
            RFLR=RFLR,
            RFRF=RFRF,
            AutoRFRF=AutoRFRF,
            first_stage=first_stage,
        )[p.method]
        result = evaluate(p, protests)
        results[name] = result

    # TODO: validation
    # res.wooldridge_regression
    # res.wooldridge_overid
    # res.sargan
    return results
