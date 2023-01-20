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

wiedemann_newspapers = [
    "lvz.de",
    "saechsische.de",
    "stuttgarter-zeitung.de",
    "weser-kurier.de",
]  # cf. Wiedemann et al. 2022

local_newspapers = [
    "augsburger-allgemeine.de",
    "bz-berlin.de",
    "haz.de",
    "noz.de",
    "rp-online.de",
    "rundschau-online.de",
    *wiedemann_newspapers,
]

media_ids = {
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

all_newspapers_with_id = [
    (name, media_ids[name])
    for name in sorted(
        set([*newspapers_of_record, *popular_newspapers, *local_newspapers])
    )
]

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
]

start_year = 2013
end_year = 2022
