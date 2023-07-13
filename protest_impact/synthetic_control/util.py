import math
from datetime import date, datetime

import pandas as pd


def _split(df, protest_date):
    if isinstance(protest_date, (str, date, datetime)):
        protest_date = pd.Timestamp(protest_date)
    return df[df["date"] == protest_date].index[0]


def train_test_split(df, protest_column, protest_date):
    """
    df: df in pivoted form, with regions as columns and dates as index
    """
    df.index = (df["date"] - pd.Timestamp(protest_date)).dt.days
    df.index = pd.RangeIndex(df.index[0], df.index[-1] + 1)
    df = df.drop(columns=["date"])
    train = df.loc[:-1]
    test = df.loc[0:]
    X_train = train.drop(columns=[protest_column])
    Y_train = train[protest_column]
    X_test = test.drop(columns=[protest_column])
    Y_test = test[protest_column]
    return X_train, Y_train, X_test, Y_test
