import spacy

if __name__ == "__main__":
    spacy.cli.download("de_core_news_sm")

nlp = spacy.load("de_core_news_sm", disable=["parser", "tagger", "ner", "tokenizer"])
nlp.add_pipe("sentencizer")
