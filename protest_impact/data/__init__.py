german_regions = [
    {"name": "Baden-Württemberg", "code": "BW", "capital": "Stuttgart"},
    {"name": "Bayern", "code": "BY", "capital": "München"},
    {"name": "Berlin", "code": "BE", "capital": "Berlin"},
    {"name": "Brandenburg", "code": "BB", "capital": "Potsdam"},
    {"name": "Bremen", "code": "HB", "capital": "Bremen"},
    {"name": "Hamburg", "code": "HH", "capital": "Hamburg"},
    {"name": "Hessen", "code": "HE", "capital": "Wiesbaden"},
    {"name": "Mecklenburg-Vorpommern", "code": "MV", "capital": "Schwerin"},
    {"name": "Niedersachsen", "code": "NI", "capital": "Hannover"},
    {"name": "Nordrhein-Westfalen", "code": "NW", "capital": "Düsseldorf"},
    {"name": "Rheinland-Pfalz", "code": "RP", "capital": "Mainz"},
    {"name": "Saarland", "code": "SL", "capital": "Saarbrücken"},
    {"name": "Sachsen", "code": "SN", "capital": "Dresden"},
    {"name": "Sachsen-Anhalt", "code": "ST", "capital": "Magdeburg"},
    {"name": "Schleswig-Holstein", "code": "SH", "capital": "Kiel"},
    {"name": "Thüringen", "code": "TH", "capital": "Erfurt"},
]


_neighbor_regions = dict(
    BW=["BY", "HE", "RP"],
    BY=["BW", "HE", "SN", "TH"],
    BE=["BB"],
    BB=["BE", "MV", "NI", "SN", "ST"],
    HB=["NI"],
    HH=["SH", "NI"],
    HE=["BW", "BY", "NW", "NI", "RP", "TH"],
    MV=["BB", "NI", "SH"],
    NI=["BB", "HB", "HE", "MV", "NW", "SH", "ST"],
    NW=["HE", "NI", "RP"],
    RP=["BW", "HE", "NW", "SL"],
    SL=["RP"],
    SN=["BB", "BY", "ST", "TH"],
    ST=["BB", "NI", "SN", "TH"],
    SH=["HH", "MV", "NI"],
    TH=["BY", "HE", "NI", "SN", "ST"],
)


def _get_name(code):
    return [r["name"] for r in german_regions if r["code"] == code][0]


neighbor_regions = {
    _get_name(k): [_get_name(c) for c in v] for k, v in _neighbor_regions.items()
}
