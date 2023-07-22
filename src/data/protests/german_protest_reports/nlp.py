import spacy

nlp = spacy.load("de_core_news_sm", disable=["parser", "tagger", "ner", "tokenizer"])
nlp.add_pipe("sentencizer")
