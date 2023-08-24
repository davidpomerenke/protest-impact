import pandas as pd

from src.paths import interim_data


def load_german_protest_reports() -> pd.DataFrame:
    df = pd.read_csv(
        interim_data / "german-protest-reports/protests.csv",
    )
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    return df
