import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

from protest_impact.util import project_root

metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def train_model(model, tokenizer, description, dataset, n_epochs=6):
    model_location = (
        project_root / "models" / "protest_detection" / description / "model"
    )

    if model_location.exists():
        return AutoModelForSequenceClassification.from_pretrained(model_location)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=512
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    training_args = TrainingArguments(
        output_dir=model_location.parent / "checkpoints",
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        learning_rate=5e-6,
        weight_decay=0.2,
        num_train_epochs=n_epochs,
        fp16=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["dev"]
        if "dev" in tokenized_datasets
        else tokenized_datasets["test"]
        if "test" in tokenized_datasets
        else None,
        compute_metrics=compute_metrics
        if "dev" in tokenized_datasets or "test" in tokenized_datasets
        else None,
    )
    trainer.train()
    trainer.save_model(model_location)
    return model


from evaluate import TextClassificationEvaluator


def evaluate_(model, tokenizer, test_set):
    eval_results = TextClassificationEvaluator().compute(
        model_or_pipeline=model,
        data=test_set,
        input_column="text",
        label_column="label",
        label_mapping={"LABEL_0": 0, "LABEL_1": 1},
        tokenizer=tokenizer,
        metric=metric,
    )
    return eval_results


from sklearn.metrics import classification_report
from transformers import pipeline


def evaluate_detail(model, tokenizer, test_set):
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    predictions = classifier(a["text"] for a in test_set)
    y_pred = [int(a["label"][-1]) for a in predictions]
    y_true = [a["label"] for a in test_set]
    print(sum(y_true) / len(y_true))
    print(sum(y_pred) / len(y_pred))
    print(classification_report(y_true, y_pred))
