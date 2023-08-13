import re

from src.cache import cache


@cache
def kwic(text: str, regex: re.Pattern, n=0) -> str:
    from src.models.propensity_scores.nlp.spacy_ import nlp

    sents = list(nlp(text).sents)
    kwics_nrs = set()
    for i, sent in enumerate(sents):
        if regex.search(sent.text_with_ws):
            kwics_nrs.add(i)
            for j in range(1, n + 1):
                kwics_nrs.add(i - j)
                kwics_nrs.add(i + j)
    kwic_text = ""
    for kwic_nr in sorted(list(kwics_nrs)):
        if kwic_nr >= 0 and kwic_nr < len(sents):
            if kwic_nr - 1 not in kwics_nrs:
                kwic_text += "\n...\n"
            kwic_text += sents[kwic_nr].text_with_ws
    return kwic_text
