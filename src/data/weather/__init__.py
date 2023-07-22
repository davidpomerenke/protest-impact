from datetime import datetime

import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from meteostat import Daily, Monthly, Point
from protest_impact.util import cache
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

geolocator = Nominatim(user_agent="protest-impact")
Daily.cache_dir = ".cache/meteostat"


@cache
def _get_weather_history(location, accessor, country):
    coordinates = geolocator.geocode(location + ", " + country)
    if coordinates is None:
        coordinates = geolocator.geocode(location)
    if coordinates is None:
        raise ValueError(f"Could not find coordinates for {location}")
    point = Point(coordinates.latitude, coordinates.longitude)
    df = accessor(point, start=datetime(2005, 1, 1), end=datetime(2023, 5, 31)).fetch()
    return df


@cache
def interpolate_weather_histories(
    dfs: list[pd.DataFrame], weights: list[float]
) -> pd.DataFrame:
    masked_dfs = np.ma.masked_invalid(np.array(dfs))
    mean = np.ma.average(masked_dfs, weights=weights, axis=0)
    df = pd.DataFrame(mean, columns=dfs[0].columns, index=dfs[0].index)
    return df


def get_weather_history(location: str, country: str) -> pd.DataFrame:
    return _get_weather_history(location, Daily, country)


@cache
def impute_weather_history(df: pd.DataFrame) -> pd.DataFrame:
    imp = IterativeImputer(
        max_iter=10, random_state=0, estimator=RandomForestRegressor()
    )
    imp.fit(df)
    df = pd.DataFrame(imp.transform(df), columns=df.columns, index=df.index)
    return df


def get_monthly_weather_history(location: str, country="Germany") -> pd.DataFrame:
    return _get_weather_history(location, Monthly, country)


def get_weather(location: str, date: pd.Timestamp, country="Germany") -> pd.DataFrame:
    return get_weather_history(location, country).loc[date]


def get_monthly_weather(location: str, date: pd.Timestamp) -> pd.DataFrame:
    return get_monthly_weather_history(location).loc[date]


@cache
def get_longterm_weather(  # it's called climate, but that's confusing
    location: str, date: pd.Timestamp, country: str = "Germany"
) -> pd.DataFrame:
    # Get average of the last 10 years for each day
    df = get_weather_history(location, country)
    df = df[(df.index.year >= date.year - 10) & (df.index.year < date.year)]
    df = df[df.index.month == date.month]
    df = df[df.index.day == date.day]
    return df.mean()
