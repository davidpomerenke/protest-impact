from itertools import chain

newspapers_of_record = [
    "spiegel.de",
    "sueddeutsche.de",
    "faz.net",
    "zeit.de",
]  # cf. https://de.wikipedia.org/wiki/Leitmedium, https://en.wikipedia.org/wiki/Newspaper%20of%20record

popular_newspapers = [
    "t-online.de",
    "focus.de",
    "n-tv.de",
    "bild.de",
    "welt.de",
    "spiegel.de",
    "faz.net",
    "stern.de",
    "rnd.de",
    "tagesspiegel.de",
]  # top 10 news by unique users 2021, https://www.agof.de/newsroom/#factsandfigures (via Statista)

local_quality_newspapers = [
    "tagesspiegel.de",
    "haz.de",
    "noz.de",
    "augsburger-allgemeine.de",
    "stuttgarter-zeitung.de",
]  # https://www.die-zeitungen.de/aktuelles/news/article/news/qualitaets-ranking-deutscher-zeitungen.html#:~:text=Die%20Zeit%2C%20S%C3%BCddeutsche%20Zeitung%20und,den%20Bl%C3%A4ttern%20beste%20journalistische%20Qualit%C3%A4t.

diverse_local_newspapers = [
    "lvz.de",
    "saechsische.de",
    "stuttgarter-zeitung.de",
    "weser-kurier.de",
]  # cf. Wiedemann et al. 2022

local_newspapers_ = {
    "schleswig-holstein": [
        "shz.de",
    ],
    "mecklenburg-vorpommern": [
        "ostsee-zeitung.de",
    ],
    "hamburg": [
        "abendblatt.de",
    ],
    "bremen": [
        "weser-kurier.de",
    ],
    "niedersachsen": [
        "haz.de",
        "noz.de",
    ],
    "brandenburg": [
        "maz-online.de",
    ],
    "berlin": [
        "bz-berlin.de",
        "tagesspiegel.de",
    ],
    "sachsen-anhalt": [
        "mz-web.de",
    ],
    "nordrhein-westfalen": [
        "waz.de",
        "rp-online.de",
        "rundschau-online.de",
    ],
    "sachsen": [
        "freiepresse.de",
        "saechsische.de",
        "lvz.de",
    ],
    "thüringen": [
        "thueringer-allgemeine.de",
    ],
    "hessen": [
        "fr.de",
    ],
    "rheinland-pfalz": [
        "rheinpfalz.de",
    ],
    "saarland": [
        "saarbruecker-zeitung.de",
    ],
    "baden-württemberg": [
        "stuttgarter-zeitung.de",
        "schwaebische.de",
    ],
    "bayern": [
        "nordbayern.de",
        "augsburger-allgemeine.de",
        "merkur.de",
    ],
}  # https://www.meedia.de/publishing/die-auflagen-bilanz-der-groessten-82-regionalzeitungen-deutliche-verluste-fuer-die-grossen-in-nrw-express-und-mopo-im-freien-fall-137638686e0416e91a6373b6be6d5810

local_newspapers = list(chain(*local_newspapers_.values()))

# bringing in some diversity
other = [
    "taz.de",
    "jungewelt.de",
    "jungefreiheit.de",
    "nzz.ch",
    "derstandard.at",
    "tagesschau.de",
    "zdf.de",
    "rtl.de",
    "sat1.de",
    "emma.de",
    "heise.de",
]

media_ids = {
    "taz.de": 20001,
    "jungewelt.de": 21854,
    "jungefreiheit.de": 310655,
    "nzz.ch": 39892,
    "derstandard.at": 39179,
    "tagesschau.de": 21043,
    "zdf.de": 40752,
    "rtl.de": 179736,
    "sat1.de": 42040,
    "emma.de": 301029,
    "heise.de": 5847,
    "shz.de": 265072,
    "ostsee-zeitung.de": 39627,
    "abendblatt.de": 20569,
    "maz-online.de": 193693,
    "mz-web.de": 71406,
    "waz.de": 367278,
    "freiepresse.de": 258653,
    "thueringer-allgemeine.de": 20467,
    "fr.de": 367784,
    "rheinpfalz.de": 277936,
    "saarbruecker-zeitung.de": 40889,
    "schwaebische.de": 41407,
    "nordbayern.de": 300747,
    "merkur.de": 306164,
    "augsburger-allgemeine.de": 813687,
    "bild.de": 22009,
    "bz-berlin.de": 144263,
    "faz.net": 19404,
    "focus.de": 19972,
    "haz.de": 21547,
    "lvz.de": 518104,
    "n-tv.de": 23538,
    "noz.de": 70220,
    "rnd.de": 1324242,
    "rp-online.de": 40776,
    "rundschau-online.de": 119201,
    "saechsische.de": 1129453,
    "spiegel.de": 19831,
    "stern.de": 20244,
    "stuttgarter-zeitung.de": 39696,
    "sueddeutsche.de": 21895,
    "t-online.de": 40762,
    "tagesspiegel.de": 39206,
    "welt.de": 20453,
    "weser-kurier.de": 119186,
    "zeit.de": 22119,
}

incomplete_newspapers = [
    "merkur.de",
    "freiepresse.de",
    "shz.de",
    "ostsee-zeitung.de",
    "maz-online.de",
    "mz-web.de",
    "fr.de",
    "saarbruecker-zeitung.de",
    "schwaebische.de",
    "nordbayern.de",
]

all_newspapers_with_id = {
    name: media_ids[name]
    for name in sorted(
        set(
            [
                *newspapers_of_record,
                *popular_newspapers,
                *local_newspapers,
                *other,
            ]
        )
    )
}

complete_newspapers_with_id = {
    name: media_ids[name]
    for name in sorted(
        set(
            [
                *newspapers_of_record,
                *popular_newspapers,
                *local_newspapers,
            ]
        )
        - set(incomplete_newspapers)
    )
}

# parts of urls not to be scraped
filter_words = [
    "quiz",
    "sportal.spiegel.de",
    "tagesspiegel.feedsportal.com",
    "neukunden",
    "zip6.zeit.de",
    "hosting.de",
    "tarife-und-produkte.t-online.de",
    "gewinnspiele.t-online.de",
    "m.welt.de",
    "images.zeit.de",
    "stadtkind-stuttgart.de",
    "altavist.com",
    "planestream.de",
    "advent.spiegel.de",
    "feedsportal.com",
    "blog.lvz-online.de",
    "sportschau.de",
]

start_year = 2013
end_year = 2022
