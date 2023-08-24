import pandas as pd

from src.paths import interim_data


def load_german_protest_reports() -> pd.DataFrame:
    df = pd.read_csv(
        interim_data / "german-protest-reports/protests.csv",
        parse_dates=["protest_date"],
    )
    return df
