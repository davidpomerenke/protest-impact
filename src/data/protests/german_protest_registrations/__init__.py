import pandas as pd
from protest_impact.util import cache, project_root


@cache
def load_german_protest_registrations() -> pd.DataFrame:
    """
    Load the German protest registrations dataset.
    Note that the covered years vary between cities (with 2022 being the most covered year).
    """

    df = pd.read_csv(
        project_root
        / "protest_impact/data/protests/german_protest_registrations/all-protests.csv",
        parse_dates=["Datum"],
    )
    df = df.rename(
        columns={
            "Datum": "date",
            "Stadt": "location",
            "Veranstalter": "actor",
            "Teilnehmer": "size_pre",
            "Teilnehmer (tatsächlich)": "size_post",
            "Thema": "notes",
            "Bundesland": "region",
        }
    )
    df["type"] = pd.NA
    df["country"] = "Germany"
    return df[
        [
            "date",
            "type",
            "actor",
            "country",
            "region",
            "location",
            "notes",
            "size_pre",
            "size_post",
        ]
    ]
