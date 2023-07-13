import numpy as np
import pandas as pd

from protest_impact.data.protests.acled import load_acled_protests
from protest_impact.data.protests.german_protest_registrations import (
    load_german_protest_registrations,
)
from protest_impact.data.protests.keywords import all_keywords, movement_keywords
from protest_impact.util import cache, kill_umlauts_without_mercy, project_root


@cache
def aggregate_protests(protests: pd.DataFrame) -> pd.DataFrame:
    agg = protests.groupby(["admin1", "event_date"])
    largest_protests = agg.apply(
        lambda x: x.sort_values(["size"], ascending=False).head(1)
    )
    largest_protests["size"] = agg["size"].sum()
    largest_protests["n_protests"] = agg.size()
    largest_protests["largest_protest_size"] = agg["size"].max()
    largest_protests["has_mixed_groups"] = agg["assoc_actor_1"].nunique() > 1
    largest_protests = largest_protests.reset_index(drop=True)
    return largest_protests


@cache
def get_organization_labels(df: pd.DataFrame, topic: str) -> pd.Series:
    kws = all_keywords(topic)
    kws = [kw.lower().replace(" ", "") for kw in kws]

    org_names = movement_keywords[topic]["organizations"].items()
    org_names = [
        (code, [alias.lower().replace("*", "").replace(" ", "") for alias in aliases])
        for code, aliases in org_names
    ]

    def org(x):
        is_relevant = False
        orgs = set()
        for content in x.values:
            if not isinstance(content, str):
                continue
            content = content.lower().replace(" ", "")
            if any([kw in content for kw in kws]):
                is_relevant = True
            for code, aliases in org_names:
                if any([alias in content for alias in aliases]):
                    orgs.add(code)
        if not is_relevant:
            return np.nan
        elif len(orgs) == 0:
            return f"other_{topic}_org"
        else:
            return "; ".join(orgs)

    orgs = df.apply(org, axis=1)
    orgs.name = "actor"
    return orgs


def load_protests():
    acled = load_acled_protests()
    acled["source"] = "acled"
    gpreg = load_german_protest_registrations()
    gpreg["source"] = "gpreg"
    gpreg.location = gpreg.location.apply(kill_umlauts_without_mercy)
    protests = pd.concat([acled, gpreg])
    protests = protests.reset_index(drop=True)
    return protests


def load_climate_protests_with_labels():
    df = load_protests()
    df["actor"] = get_organization_labels(df, "climate").str.upper()
    df = df.dropna(subset=["actor"])
    return df
