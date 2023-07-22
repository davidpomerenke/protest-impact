from munch import Munch

german_regions = [
    Munch(name="Baden-Württemberg", code="BW", capital="Stuttgart"),
    Munch(name="Bayern", code="BY", capital="München"),
    Munch(name="Berlin", code="BE", capital="Berlin"),
    Munch(name="Brandenburg", code="BB", capital="Potsdam"),
    Munch(name="Bremen", code="HB", capital="Bremen"),
    Munch(name="Hamburg", code="HH", capital="Hamburg"),
    Munch(name="Hessen", code="HE", capital="Wiesbaden"),
    Munch(name="Mecklenburg-Vorpommern", code="MV", capital="Schwerin"),
    Munch(name="Niedersachsen", code="NI", capital="Hannover"),
    Munch(name="Nordrhein-Westfalen", code="NW", capital="Düsseldorf"),
    Munch(name="Rheinland-Pfalz", code="RP", capital="Mainz"),
    Munch(name="Saarland", code="SL", capital="Saarbrücken"),
    Munch(name="Sachsen", code="SN", capital="Dresden"),
    Munch(name="Sachsen-Anhalt", code="ST", capital="Magdeburg"),
    Munch(name="Schleswig-Holstein", code="SH", capital="Kiel"),
    Munch(name="Thüringen", code="TH", capital="Erfurt"),
]

german_region_names = [r.name for r in german_regions]


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
    return [r.name for r in german_regions if r.code == code][0]


neighbor_regions = {
    _get_name(k): [_get_name(c) for c in v] for k, v in _neighbor_regions.items()
}
