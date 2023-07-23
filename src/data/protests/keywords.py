from itertools import chain

import pandas as pd
import yaml

from src.paths import _root

abbreviations = dict(
    FFF="Fridays for Future",
    FFFX="Fridays for Future + X",
    ALG="Letzte Generation",
    XR="Extinction Rebellion",
    EG="Ende Gelände",
    EFO="End Fossil Occupy",
    GP="Greenpeace",
    OTHER_CLIMATE_ORG="Other",
)

with open(_root / "src/data/protests/keywords.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

treatment_keywords = config["treatment_keywords"]
movement_keywords = config["movement_keywords"]


def all_keywords(topic: str) -> list[str]:
    kws = (
        (
            list(chain(*movement_keywords[topic]["organizations"].values()))
            if "organizations" in movement_keywords[topic]
            else []
        )
        + movement_keywords[topic]["movement"]
        + movement_keywords[topic]["topic"]
        + (
            movement_keywords[topic]["framing"]
            if "framing" in movement_keywords[topic]
            else []
        )
        + (
            movement_keywords[topic]["goal"]
            if "goal" in movement_keywords[topic]
            else []
        )
    )
    kws = [kw.replace("?", "") for kw in kws]
    return kws


def join(kws: list[str], mode="default", short=False) -> str:
    assert mode in ["default", "dereko"]
    if short:
        kws = [kw for kw in kws if "?" not in kw]
    kws = [kw.replace("?", "") for kw in kws]
    if mode == "default":
        q = " OR ".join([f'"{a}"' if len(a.split()) > 1 else a for a in kws])
    if mode == "dereko":
        q = " OR ".join([f"({a})" if len(a.split()) > 1 else a for a in kws])
    q = q.replace("\\", "")
    return q


def climate_queries(mode="default", short=False):
    protest_keywords = (
        treatment_keywords["de"]
        + movement_keywords["climate"]["movement"]
        + list(chain(*movement_keywords["climate"]["organizations"].values()))
    )
    if short:
        topic_keywords = movement_keywords["climate"]["topic"]
    else:
        topic_keywords = list(
            chain(
                *[movement_keywords["climate"][a] for a in ["topic", "framing", "goal"]]
            )
        )

    framing_keywords = movement_keywords["climate"]["framing"]
    goal_keywords = movement_keywords["climate"]["goal"]
    subsidiary_goal = movement_keywords["climate"]["subsidiary_goal"]

    if mode == "raw":
        return dict(
            topic=topic_keywords,
            protest=protest_keywords,
            framing=framing_keywords,
            goal=goal_keywords,
            subsidiary_goal=subsidiary_goal,
        )

    protest = join(protest_keywords, mode=mode, short=short)
    topic = join(topic_keywords, mode=mode, short=short)
    framing = join(framing_keywords, mode=mode, short=short)
    goal = join(goal_keywords, mode=mode, short=short)
    subsidiary_goal = join(subsidiary_goal, mode=mode, short=short)

    and_not = "NOT" if mode == "dereko" else "AND NOT"

    return dict(
        protest=f"({topic}) AND ({protest})",
        not_protest=f"({topic}) {and_not} ({protest})",
        framing=f"({topic}) AND ({framing})",
        goal=f"({topic}) AND ({goal})",
        subsidiary_goal=f"({topic}) AND ({subsidiary_goal})",
    )


def climate_query_table():
    qs = climate_queries(mode="raw")
    qs = {
        k: [v.replace("\\", "") for v in vs if not v.startswith("?")]
        for k, vs in qs.items()
    }
    maxlen = max([len(v) for v in qs.values()])
    df = pd.DataFrame({k: v + [""] * (maxlen - len(v)) for k, v in qs.items()})
    return df
