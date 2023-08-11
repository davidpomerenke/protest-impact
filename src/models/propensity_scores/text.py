import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from matplotlib import pyplot as plt
from munch import Munch
from nltk.corpus import stopwords
from peft import LoraConfig, TaskType, get_peft_model
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import (
    KFold,
    TimeSeriesSplit,
    cross_val_predict,
    cross_val_score,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.cache import cache
from src.features.time_series import get_lagged_df
from src.paths import models as model_path


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


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


@cache
def load_dataset(model_name, max_length):
    d = get_data()
    dataset = (
        Dataset.from_dict(
            dict(
                text=d._X_text["text_lag0"],
                labels=d.y,
            )
        )
        .shuffle(seed=54325)
        .select(range(2000))
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = dataset.map(
        lambda sample: tokenizer(
            sample["text"], padding="max_length", truncation=True, max_length=max_length
        ),
    )
    ds = dataset.train_test_split(test_size=0.2)
    train = ds["train"].with_format("torch")
    test = ds["test"].with_format("torch")
    return train, test


f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return f1_metric.compute(predictions=predictions, references=labels)


models = [
    ("prajjwal1/bert-tiny", 512),
    ("bert-base-german-cased", 512),
    ("LennartKeller/longformer-gottbert-base-8192-aw512", 8192),
]


def train_bert(model_name, max_length):
    train_dataset, test_dataset = load_dataset(model_name, max_length=max_length)
    r = 1
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=r, lora_alpha=r, lora_dropout=0.1
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    path = model_path / "bert"
    path.mkdir(exist_ok=True, parents=True)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=path,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            warmup_ratio=0.1,
            evaluation_strategy="steps",
            save_steps=50,
            eval_steps=50,
            save_total_limit=5,
            use_mps_device=get_device() == "mps",
        ),
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    trainer.train()
    # trainer.save_model(path / "final")
    model.save_pretrained(path / "final")
    final_metrics = trainer.evaluate()
    trainer.save_metrics("final", final_metrics)


if __name__ == "__main__":
    if False:
        import nltk

        nltk.download("stopwords")
    train_bert(*models[0])
