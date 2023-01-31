import re

treatment_keywords = {
    "en": [
        "protest",
        "protester",
        "demonstration",
        "rally",
        "march",
        "parade",
        "strike",
        "picket",
        "sit-in",
        "rebellion",
        "uprising",
        "resistance",
        "unrest",
        "activism",
        "activist",
        "civil disobedience",
    ],
    "de": [
        "protest",
        "demonstration",
        "demo",
        "demonstrant",
        "demonstrantin",
        "marsch",
        "aufmarsch",
        "parade",
        "kundgebung",
        "streik",
        "mahnwache",
        "sitzblockade",
        "aufstand",
        "rebellion",
        "widerstand",
        "unruhen",
        "aktivismus",
        "aktivist",
        "aktivistin",
        "ziviler ungehorsam",
    ],
}

movement_keywords = {
    "climate": [
        "fridays for future",
        "fridaysforfuture",
        "fridays4future",
        "extinction rebellion",
        "just stop oil",
        "letzte generation",
        "ultima generazione",
        "ende gelände",
        "klimabewegung",
        "klimaaktivist",
        "klimaaktivistin",
        "klimastreik",
        "klimastreikende",
        "klimaschutz",
        "klimagerechtigkeit",
    ],
    "racism": [
        "black lives matter",
        "blacklivesmatter",
        "rassismus",
        "rassistisch",
        "antirassismus",
        "antirassistisch",
        "antirassist",
        "antirassistin",
    ],
    "feminism": [
        "feminismus",
        "feminist",
        "feministin",
        "feministisch",
        "frauenstreik",
        "sexismus",
        "equal pay",
        "gleichberechtigung",
        "gleichstellung",
        "geschlechtergerechtigkeit",
    ],
    "covid": [
        "querdenker",
        "querdenken",
        "hygienedemo",
    ],
    "antifascism": [
        "antifa",
        "antifaschismus",
        "antifaschist",
        "antifaschistin",
        "antifaschistisch",
    ],
}

# the exact search string was:
search_string = '*protest* OR Versammlung* OR demonstr* OR Kundgebung* OR Kampagne* OR "Soziale Bewegung*" OR Hausbesetzung* OR Streik* OR Unterschriftensammlung* OR Hasskriminalität* OR Unruhen* OR Aufruhr* OR Aufstand* OR Boykott* OR Riot* OR Aktivis* OR Widerstand* OR Mobilisierung* OR Bürgerinitiative* OR Bürgerbegehren*'  # from Wiedemann et al. 2022

_search_regex = re.sub(r"\*", ".*", search_string)
_search_regex = re.sub(r" OR ", "|", _search_regex)
_search_regex = re.sub(r'"', "", _search_regex)
search_regex = re.compile(_search_regex, re.IGNORECASE)
