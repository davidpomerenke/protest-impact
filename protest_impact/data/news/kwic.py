import spacy

from protest_impact.data.protests.config import search_regex

nlp = spacy.load("de_core_news_sm", disable=["parser", "tagger", "ner", "tokenizer"])
nlp.add_pipe("sentencizer")


def kwic(text, n=0):
    sents = list(nlp(text).sents)
    kwics_nrs = set()
    for i, sent in enumerate(sents):
        if search_regex.search(sent.text_with_ws):
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

def kwic_dataset(dataset, n=0):
    return dataset.map(
        lambda x: {
            "text": x["meta"]["title"]
            + "\n\n"
            + kwic("\n".join(list(x["text"].split("\n"))[1:]), n=n)
        }
    )
