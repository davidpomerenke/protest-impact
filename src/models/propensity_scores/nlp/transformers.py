import evaluate
import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.cache import cache
from src.models.propensity_scores.nlp.basic import get_data
from src.paths import models as model_path
from src.paths import processed_data


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
    dataset = Dataset.from_dict(
        dict(
            text=d._X_text["text_lag0"],
            labels=d.y,
        )
    )
    dataset.save_to_disk(processed_data / "propensity-scores/text")
    dataset = dataset.shuffle(seed=54325).select(range(2000))
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


def train_bert(model_name, max_length, path):
    train_dataset, test_dataset = load_dataset(model_name, max_length=max_length)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=1,
        lora_alpha=1,
        lora_dropout=0.1,
        target_modules=[
            "query",
            "key",
            "value",
            "dense",
        ],
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    path = path / model_name.rsplit("/")[-1]
    path.mkdir(exist_ok=True, parents=True)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=path,
            learning_rate=2e-5,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            weight_decay=0.01,
            warmup_ratio=0.1,
            gradient_accumulation_steps=8,
            evaluation_strategy="steps",
            save_steps=1000,
            eval_steps=500,
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
    torch.cuda.empty_cache()
    train_bert(*models[2], path=model_path / "propensity_scores/text")
