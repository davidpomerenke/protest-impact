import re
from itertools import chain

import yaml

from protest_impact.util import project_root

with open(project_root / "protest_impact" / "data" / "protests" / "keywords.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

treatment_keywords = config["treatment_keywords"]
movement_keywords = config["movement_keywords"]


def as_simple_search_string(or_keywords: list[str]) -> str:
    # TODO add quotation marks
    return " OR ".join(f'"{a}"' if len(a.split()) > 1 else a for a in or_keywords)


def as_regex(or_keywords: list[str]) -> re.Pattern:
    r = "|".join(or_keywords)
    r = re.sub(r"\*", ".*", r)
    r = re.compile(r, re.IGNORECASE)
    return r


def all_keywords(topic: str) -> list[str]:
    kws = (
        (
            list(chain(*movement_keywords[topic]["organizations"].values()))
            if "organizations" in movement_keywords[topic]
            else []
        )
        + movement_keywords[topic]["movement"]
        + movement_keywords[topic]["topic"]
        + movement_keywords[topic]["framing"]
        if "framing" in movement_keywords[topic]
        else [] + movement_keywords[topic]["goal"]
        if "goal" in movement_keywords[topic]
        else []
    )
    return kws


def join_with_quotes(kws: list[str]) -> str:
    q = " OR ".join([f'"{a}"' if len(a.split()) > 1 else a for a in kws])
    q = q.replace("\\", "")
    return q


def climate_queries():
    protest_keywords = (
        treatment_keywords["de"]
        + movement_keywords["climate"]["movement"]
        + list(chain(*movement_keywords["climate"]["organizations"].values()))
    )
    topic_keywords = list(
        chain(*[movement_keywords["climate"][a] for a in ["topic", "framing", "goal"]])
    )
    framing_keywords = movement_keywords["climate"]["framing"]
    goal_keywords = movement_keywords["climate"]["goal"]
    subsidiary_goal = movement_keywords["climate"]["subsidiary_goal"]

    protest = join_with_quotes(protest_keywords)
    topic = join_with_quotes(topic_keywords)
    framing = join_with_quotes(framing_keywords)
    goal = join_with_quotes(goal_keywords)
    subsidiary_goal = join_with_quotes(subsidiary_goal)

    return dict(
        topic=topic,
        protest=f"({topic}) AND ({protest})",
        not_protest=f"({topic}) AND NOT ({protest})",
        framing=f"({topic}) AND ({framing})",
        goal=f"({topic}) AND ({goal})",
        subsidiary_goal=f"({topic}) AND ({subsidiary_goal})",
    )
