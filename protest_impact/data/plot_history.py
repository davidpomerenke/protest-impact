from datetime import datetime

import holidays
import matplotlib.pyplot as plt
import pandas as pd
from dateutil.relativedelta import relativedelta


def plot_history(df: pd.DataFrame, year: int = None, newspaper: str = None):
    german_holidays = holidays.Germany(years=range(2013, 2023))
    assert year is None or newspaper is None
    if newspaper is not None:
        newspapers = [newspaper]
        years = [datetime(year, 1, 1) for year in df["date"].dt.year.unique()]
    if year is not None:
        years = [datetime(year, 1, 1)]
        newspapers = df["newspaper"].unique()
    # plot the data for each newspaper (with matplotlib)
    # use the object oriented interface of matplotlib
    plt.close()
    fig, axs = plt.subplots(max(len(newspapers), len(years)), figsize=(20, 20))
    ax_nr = 0
    for newspaper in newspapers:
        for year in years:
            ax = axs[ax_nr]
            ax_nr += 1
            # plot the data for the newspaper (different hue for each engine)
            for engine in ["mediacloud", "google"]:
                df_part = df[
                    (df["newspaper"] == newspaper)
                    & (df["engine"] == engine)
                    & (df["date"].dt.year == year.year)
                ]
                ax.plot(df_part["date"], df_part["count"], label=engine)
            ax.set_title(f"{newspaper} {year.year}")
            # highlight weekends, holidays, and month boundaries
            for date in pd.date_range(year, year + relativedelta(years=1), freq="D"):
                if date.dayofweek >= 5:
                    ax.axvspan(
                        date, date + pd.Timedelta(days=1), color="orange", alpha=0.1
                    )
                if date in german_holidays:
                    ax.axvspan(
                        date, date + pd.Timedelta(days=1), color="red", alpha=0.1
                    )
                if date.day == 1:
                    ax.axvspan(
                        date, date + pd.Timedelta(days=1), color="black", alpha=0.2
                    )
            ax.set_xlim(year, year + relativedelta(years=1))
    return fig
