import re

import yaml

with open("config.yaml") as f:
    config = yaml.load(f)

treatment_keywords = config["treatment_keywords"]

search_string = " OR ".join(treatment_keywords["de"])

_search_regex = re.sub(r"\*", ".*", search_string)
_search_regex = re.sub(r" OR ", "|", _search_regex)
_search_regex = re.sub(r'"', "", _search_regex)
search_regex = re.compile(_search_regex, re.IGNORECASE)
