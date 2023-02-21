import re

import yaml

from protest_impact.util import project_root

with open(project_root / "protest_impact" / "data" / "protests" / "config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

treatment_keywords = config["treatment_keywords"]
movement_keywords = config["movement_keywords"]

# TODO add quotation marks
search_string = " OR ".join(
    f'"{a}"' if len(a.split()) > 1 else a for a in treatment_keywords["de"]
)

_search_regex = re.sub(r"\*", ".*", search_string)
_search_regex = re.sub(r" OR ", "|", _search_regex)
_search_regex = re.sub(r'"', "", _search_regex)
search_regex = re.compile(_search_regex, re.IGNORECASE)
