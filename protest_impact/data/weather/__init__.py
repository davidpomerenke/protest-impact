from datetime import datetime

import pandas as pd
from geopy.geocoders import Nominatim
from meteostat import Daily, Monthly, Point

from protest_impact.util import cache

geolocator = Nominatim(user_agent="protest-impact")
Daily.cache_dir = ".cache/meteostat"


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
def get_weather_history(location: str, country) -> pd.DataFrame:
    return _get_weather_history(location, Daily, country)


@cache
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
