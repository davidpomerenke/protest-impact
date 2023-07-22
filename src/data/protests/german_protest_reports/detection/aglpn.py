from datasets import ClassLabel, DatasetDict, load_dataset

from src import project_root

# AGLPN = another german local protest news dataset? 😵‍💫


def load_aglpn_dataset():
    aglpn = load_dataset(
        str(project_root / "data" / "protest"),
        data_files={"main": "protest_news_shuffled_v2_annotated_v1.jsonl"},
    )
    aglpn["main"] = aglpn["main"].filter(lambda x: x["answer"] != "ignore")
    # fix a stupid mistake during annotation
    # (first marked with relevant/irrelevant, then with accept/reject)
    aglpn["main"] = aglpn["main"].map(
        function=(
            lambda x, i: {
                "label": (1 if x["accept"] == ["relevant"] else 0)
                if i < 800
                else (1 if x["answer"] == "accept" else 0)
            }
        ),
        with_indices=True,
    )
    train_test = aglpn["main"].train_test_split(test_size=500 / 1150, seed=20230211)
    train_positive_class = load_dataset(
        str(project_root / "data" / "protest"),
        data_files={
            "main": "protest_news_shuffled_v2_sample_with_positive_predictions_annotated.jsonl"
        },
    )
    train_positive_class["main"] = train_positive_class["main"].map(
        function=(lambda x: {"label": 1 if x["answer"] == "accept" else 0})
    )
    aglpn = DatasetDict(
        {
            "train": train_test["train"],
            "train.positive": train_positive_class["main"],
            "test": train_test["test"],
        }
    )
    return aglpn
