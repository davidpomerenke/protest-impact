import pandas as pd
from protest_impact.util import project_root


def load_mobility():
    df = pd.read_csv(
        project_root
        / "data/our-world-in-data/mobility-trends/changes-visitors-covid.csv",
        parse_dates=["Day"],
    )
    df = df.rename(columns={"Day": "date"})
    df = df.query("Entity == 'Germany'")
    df = df.drop(columns=["Code", "Entity"])
    # fill days at the start with zeros
    # add row for 2020-01-01:
    df = df.append({"date": pd.to_datetime("2020-01-01")}, ignore_index=True)
    df = df.set_index("date").asfreq("D", fill_value=0).reset_index().fillna(0)
    df = df.fillna(0)
    # fill days at the end with last value
    # add row for 2022-12-31:
    df = df.append({"date": pd.to_datetime("2023-01-01")}, ignore_index=True)
    df = df.set_index("date").asfreq("D", method="ffill").reset_index().fillna(0)
    # drop the last row again
    df = df.iloc[:-1]
    return df


def load_stringency():
    df = pd.read_csv(
        project_root / "data/our-world-in-data/stringency-index/owid-covid-data.csv",
        parse_dates=["date"],
    )
    df = df.query("location == 'Germany'")
    df = df[["date", "stringency_index"]]
    # fill days at the start with zeros
    df = df.append({"date": pd.to_datetime("2020-01-01")}, ignore_index=True)
    df = df.set_index("date").asfreq("D", fill_value=0).reset_index().fillna(0)
    return df
