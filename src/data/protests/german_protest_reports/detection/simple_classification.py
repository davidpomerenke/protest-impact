# create a pipeline
import optuna
import pandas as pd
from protest_impact.data.protests.detection.glpn import load_glpn_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# use hyperparameter optimization to find the best parameters
# compare bow and tfidf for feature extraction
# compare svm, decision tree, random forest, xgboost, logistic regression, ridge regression, MLP for classifier
# use dev set and cross validation

glpn = load_glpn_dataset()


def objective(trial, return_model=False, eval="split"):
    # suggest classifier
    classifier_name = trial.suggest_categorical(
        "classifier",
        [
            "SVC",
            "DecisionTreeClassifier",
            "RandomForestClassifier",
            "XGBClassifier",
            "LogisticRegression",
            "RidgeClassifier",
            "MLPClassifier",
        ],
    )
    if classifier_name == "SVC":
        # suggest parameters
        C = trial.suggest_float("C", 1e-7, 1e3, log=True)
        # create classifier
        classifier_obj = SVC(C=C)
    elif classifier_name == "DecisionTreeClassifier":
        # suggest parameters
        max_depth = trial.suggest_int("max_depth", 1, 32, log=True)
        # create classifier
        classifier_obj = DecisionTreeClassifier(max_depth=max_depth)
    elif classifier_name == "RandomForestClassifier":
        # suggest parameters
        n_estimators = trial.suggest_int("n_estimators", 1, 100)
        max_depth = trial.suggest_int("max_depth", 1, 32, log=True)
        # create classifier
        classifier_obj = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth
        )
    elif classifier_name == "XGBClassifier":
        # suggest parameters
        n_estimators = trial.suggest_int("n_estimators", 1, 100)
        max_depth = trial.suggest_int("max_depth", 1, 32, log=True)
        # create classifier
        classifier_obj = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth)
    if classifier_name == "LogisticRegression":
        # suggest parameters
        C = trial.suggest_float("C", 1e-7, 1e3, log=True)
        # create classifier
        classifier_obj = LogisticRegression(C=C)
    elif classifier_name == "RidgeClassifier":
        # suggest parameters
        alpha = trial.suggest_float("alpha", 1e-7, 1e3, log=True)
        # create classifier
        classifier_obj = RidgeClassifier(alpha=alpha)
    elif classifier_name == "MLPClassifier":
        # suggest parameters
        hidden_layer_sizes = trial.suggest_int("hidden_layer_sizes", 1, 20)
        # create classifier
        classifier_obj = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes)

    # suggest feature extraction
    feature_extraction_name = trial.suggest_categorical(
        "feature_extraction", ["bow", "tfidf"]
    )
    if feature_extraction_name == "bow":
        # suggest parameters
        max_features = trial.suggest_int("max_features", 8, 256, log=True)
        max_ngram = trial.suggest_int("max_ngram", 1, 5)
        # create feature extraction
        feature_extraction_obj = CountVectorizer(
            max_features=max_features, ngram_range=(1, max_ngram)
        )
    elif feature_extraction_name == "tfidf":
        # suggest parameters
        max_features = trial.suggest_int("max_features", 8, 256, log=True)
        ngram_range = trial.suggest_int("ngram_range", 1, 5)
        # create feature extraction
        feature_extraction_obj = TfidfVectorizer(
            max_features=max_features, ngram_range=(1, ngram_range)
        )
    # create pipeline
    pipe = Pipeline([("fe", feature_extraction_obj), ("clf", classifier_obj)])
    if eval == "split":
        # fit
        pipe.fit(glpn["train"]["excerpt"], glpn["train"]["label"])
        if return_model:
            return pipe
        # evaluate on dev
        y_pred = pipe.predict(glpn["dev"]["excerpt"])
        return f1_score(glpn["dev"]["label"], y_pred, average="macro")
    elif eval == "cv":
        if return_model:
            # train on all data
            pipe.fit(glpn["train"]["excerpt"], glpn["train"]["label"])
            return pipe
        # use cross validation (folding by newspaper)
        newspapers = pd.Series(glpn["train"]["newspaper"])
        splits = []
        for newspaper in newspapers.unique():
            train = newspapers != newspaper
            dev = newspapers == newspaper
            splits.append((train, dev))
        scores = cross_val_score(
            pipe,
            glpn["train"]["excerpt"],
            glpn["train"]["label"],
            cv=splits,
            scoring="f1_macro",
        )
        return scores.mean()
