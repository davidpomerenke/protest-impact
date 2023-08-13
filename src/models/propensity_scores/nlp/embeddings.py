from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess, tokenize
from joblib.parallel import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from src.cache import cache
from src.models.propensity_scores.nlp.basic import get_data
from src.paths import processed_data

path = processed_data / "propensity_scores/nlp/embeddings"


@cache
def train_embeddings(cutoff):
    d = get_data(cutoff=cutoff)
    X = d._X_text["text_lag0"].sample(4000)
    documents = [
        TaggedDocument(simple_preprocess(doc), [tag]) for tag, doc in tqdm(enumerate(X))
    ]
    # tags here have nothing to do with labels!
    model = Doc2Vec(
        documents,
        vector_size=300,
        window=10,
        min_count=5,
        workers=4,
        epochs=20,
        dm=0,
    )
    print("Model trained")
    return model


@cache
def get_embeddings(cutoff):
    d = get_data(cutoff=cutoff)
    model = train_embeddings(cutoff=cutoff)
    embeddings = [
        model.infer_vector(simple_preprocess(doc))
        for doc in tqdm(d._X_text["text_lag0"].iloc[:4000])
    ]
    return embeddings


def f1_embeddings(cutoff):
    X = get_embeddings(cutoff=cutoff)
    d = get_data(cutoff=cutoff)
    clf = LogisticRegression(random_state=0, max_iter=1000, class_weight="balanced")
    cvs = cross_val_score(clf, X, d.y.iloc[:4000], cv=d.tscv, scoring="f1")
    print(f"Cross-validated F1 score: {cvs.mean():.3f} +/- {cvs.std():.3f}")


def precision_recall_curve(cutoff):
    d = get_data(cutoff=cutoff)
    X = get_embeddings(cutoff=cutoff)
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, precision_recall_curve
    from sklearn.model_selection import KFold, cross_val_predict

    # Assuming d is your dataset object with features X and labels y and d.tscv is your time series cross-validator

    clf = LogisticRegression(random_state=0, max_iter=1000, class_weight="balanced")
    cv = KFold(n_splits=20, shuffle=False)
    y_scores = cross_val_predict(
        clf, X, d.y.iloc[:4000], cv=cv, method="predict_proba"
    )[:, 1]

    # Compute precision-recall pairs for different threshold values
    precision, recall, thresholds = precision_recall_curve(d.y.iloc[:4000], y_scores)

    # Compute the average precision score
    average_precision = average_precision_score(d.y.iloc[:4000], y_scores)

    # Plot the PR curve
    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, label=f"Avg. Precision = {average_precision:0.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()
