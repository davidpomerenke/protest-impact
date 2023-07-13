# investigating the impact of short-term changes in the weather on protest size

from collections import defaultdict
from functools import partial

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

from protest_impact.data.protests import load_official_protests
from protest_impact.data.weather import get_weather_history
from protest_impact.util import cache


@cache
def load_useful_protests(topic_clusters=True):
    df = load_official_protests(all=True)
    df = df[(df["registered"] > 0) & (df["actual"] > 0)]
    if topic_clusters:
        cluster_terms, labels = get_topic_clusters(df["topic"], k=30)
        df["topic_cluster"] = labels
        topic_dummies = pd.get_dummies(
            df["topic_cluster"], prefix="topic", drop_first=True
        )
    df = df[["event_date", "location", "registered", "actual", "topic"]]
    if topic_clusters:
        df = pd.concat([df, topic_dummies], axis=1)
    return (cluster_terms, df) if topic_clusters else df


def remove_outliers(df, var, std=3):
    # remove outliers (with size_diff more than 3 standard deviations away from the mean)
    return df[
        df[var].between(
            df[var].mean() - std * df[var].std(),
            df[var].mean() + std * df[var].std(),
        )
    ]


@cache
def get_topic_clusters(series, k=20):
    nltk.download("stopwords", quiet=True)
    stopwords = (
        nltk.corpus.stopwords.words("german")
        + nltk.corpus.stopwords.words("english")
        + "heute jeweils täglich protest demonstration mahnwache kundgebung mo di mi do fr sa so 2019 2020 2021 2022".split()
        + [str(a) for a in range(32) if a != 19]
        + ["0" + str(i) for i in range(10)]
    )
    vectorizer = TfidfVectorizer(
        stop_words=stopwords, ngram_range=(1, 5), max_features=1000, sublinear_tf=True
    )
    vectorizer.fit(series.unique())
    X = vectorizer.transform(series)
    X = pd.DataFrame.sparse.from_spmatrix(X)
    X.columns = vectorizer.get_feature_names_out()

    model = KMeans(n_clusters=k, init="k-means++", max_iter=100, n_init=1).fit(X)

    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    cluster_terms = defaultdict(str)
    for i in range(k):
        cluster_terms[i] = ", ".join([terms[ind] for ind in order_centroids[i, :5]])

    return dict(cluster_terms), model.labels_


weather_histories = dict(
    Magdeburg=get_weather_history("Magdeburg", "Germany"),
    Berlin=get_weather_history("Berlin", "Germany"),
)


@cache
def get_weather_histories(protest, start, stop, window):
    b = pd.Timedelta(days=start - window)
    a = pd.Timedelta(days=stop)
    d = protest["event_date"]
    weather = weather_histories[protest["location"]].loc[d + b : d + a, "prcp"]
    weather.index = (weather.index - d).days.astype(str)
    return weather


@cache
def get_weather_change(protests, start, stop, window=7, x=0):
    # protests: should include columns "event_date" and "location"
    gwh = partial(get_weather_histories, start=start, stop=stop, window=window)
    weather = protests.apply(gwh, axis=1)
    protests = pd.concat([protests, weather], axis=1)
    previous = protests[[str(a) for a in range(x - window, x - 2)]].mean(axis=1)
    current = protests[str(x)]
    change = current - previous
    return change


def get_weather_changes(df, start, stop, window, n_cores=1):
    # df: should include columns "event_date" and "location"
    df = Parallel(n_jobs=n_cores)(
        delayed(get_weather_change)(df, start, stop, window=window, x=day)
        for day in tqdm(range(start, stop))
    )
    df = pd.concat(df, axis=1)
    df.columns = [f"prcp_change_{day}" for day in range(start, stop)]
    return df


def get_metrics(df, day, weights=None, model_type=1, return_results=False):
    name = f"prcp_change_{day}"
    if model_type == 1:
        # suitable because it is a better model
        # (linear relationship between the change in weather and the change in size)
        X = pd.concat([np.arcsinh(df[name]), np.log10(df["registered"])], axis=1)
        X = sm.add_constant(X)
        y = np.arcsinh(df["actual"] - df["registered"])
    elif model_type == 2:
        # suitable because it predicts what we want (actual size)
        X = pd.concat([np.arcsinh(df[name]), np.log10(df["registered"])], axis=1)
        X = sm.add_constant(X)
        y = np.log10(df["actual"])
    if weights is not None:
        model = sm.WLS(y, X, weights=weights)
    else:
        model = sm.OLS(y, X)
    results = model.fit(cov_type="HC3")
    if return_results:
        return results
    conf_int = results.conf_int().loc[name].tolist()
    return dict(
        day=day,
        coef=results.params[name],
        ci_lower=conf_int[0],
        ci_upper=conf_int[1],
        pvalue=results.pvalues[name],
    )


def normal_overlay(series):
    mu, sigma = series.mean(), series.std()
    x = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 100)
    plt.plot(
        x, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    )


def plot_size_diff(df):
    plt.scatter(df["registered"], df["actual"], s=1, alpha=0.5)
    plt.loglog()
    plt.xlabel("Registered")
    plt.ylabel("Actual")
    plt.plot([0, 100_000], [0, 100_000], color="black", linestyle="--", linewidth=1)
    plt.show()
