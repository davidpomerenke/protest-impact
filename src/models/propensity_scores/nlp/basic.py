import pandas as pd
from munch import Munch
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import (
    KFold,
    TimeSeriesSplit,
    cross_val_predict,
    cross_val_score,
)
from sklearn.naive_bayes import ComplementNB

from src.cache import cache
from src.features.time_series import get_lagged_df


@cache
def get_data(cutoff: int | None = None):
    df = get_lagged_df(
        "occ_protest",
        lags=range(-7, 1),
        ignore_group=True,
        text_cutoff=cutoff,
        region_dummies=True,
        include_texts=True,
    )
    y = df.occ_protest
    X_ts = df.drop(columns=["occ_protest", "occ_protest_lag0"])
    X_ts = X_ts[[c for c in X_ts.columns if not c.startswith("text_")]]
    text_cols = [c for c in df.columns if c.startswith("text_")]
    X_text = df[text_cols]
    tscv = TimeSeriesSplit(n_splits=20)
    clf = ComplementNB()
    return Munch(
        X_ts=X_ts, _X_text=X_text, y=y, tscv=tscv, clf=clf, text_cols=text_cols
    )


def f1_ts():
    d = get_data()
    cvs = cross_val_score(d.clf, d.X_ts, d.y, cv=d.tscv, scoring="f1")
    print(f"Cross-validated F1 score: {cvs.mean():.3f} +/- {cvs.std():.3f}")


@cache
def get_text_vector(cutoff: int | None = None):
    d = get_data(cutoff=cutoff)
    lags = [int(c.split("_lag")[1]) for c in d.text_cols]
    vec = CountVectorizer(
        stop_words=stopwords.words("german"),
        ngram_range=(1, 2),
        max_features=1000,
        min_df=5,
        max_df=0.8,
    )
    # caution: this performs preprocessing across cv splits, not suitable for evaluation
    same_day = d._X_text["text_lag0"]  # source is print newspapers, so no leaking here
    week_before = (
        d._X_text["text_lag-7"]
        + d._X_text["text_lag-6"]
        + d._X_text["text_lag-5"]
        + d._X_text["text_lag-4"]
        + d._X_text["text_lag-3"]
        + d._X_text["text_lag-2"]
        + d._X_text["text_lag-1"]
    )
    vec = vec.fit(same_day)
    # X_text = hstack([vec.transform(same_day), vec.transform(week_before)])
    X_text = vec.transform(same_day)
    return X_text, vec


def f1_text(cutoff: int | None = None):
    d = get_data(cutoff=cutoff)
    X_text, vec = get_text_vector(cutoff=cutoff)
    cvs = cross_val_score(d.clf, X_text, d.y, cv=d.tscv, scoring="f1")
    print(f"Cross-validated F1 score: {cvs.mean():.3f} +/- {cvs.std():.3f}")


def f1_combi(cutoff: int | None = None):
    d = get_data(cutoff=cutoff)
    X_text, vec = get_text_vector(cutoff=cutoff)
    X_ts_sparse = csr_matrix(d.X_ts.values)
    X_combi = hstack([X_ts_sparse, X_text])
    cvs = cross_val_score(d.clf, X_combi, d.y, cv=d.tscv, scoring="f1")
    print(f"Cross-validated F1 score: {cvs.mean():.3f} +/- {cvs.std():.3f}")


def get_text_probas(cutoff: int | None = None):
    d = get_data(cutoff=cutoff)
    X_text, vec = get_text_vector(cutoff=cutoff)
    cv = KFold(n_splits=5, shuffle=False)
    X_text_proba = cross_val_predict(
        ComplementNB(), X_text, d.y, cv=cv, method="predict_proba", n_jobs=1
    )
    return X_text_proba


def f1_augmented(cutoff: int | None = None):
    d = get_data(cutoff=cutoff)
    X_text_proba = get_text_probas(cutoff=cutoff)
    X_text_proba = pd.Series(X_text_proba[:, 1], name="text_proba")
    X_augmented = pd.concat([d.X_ts, X_text_proba], axis=1)
    cvs = cross_val_score(ComplementNB(), X_augmented, d.y, cv=d.tscv, scoring="f1")
    print(f"Cross-validated F1 score: {cvs.mean():.3f} +/- {cvs.std():.3f}")


if __name__ == "__main__":
    if False:
        import nltk

        nltk.download("stopwords")
