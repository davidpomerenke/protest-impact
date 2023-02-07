from datasets import ClassLabel, DatasetDict, load_dataset
from protest_impact.util import project_root

# AGLPN = another german local protest news dataset? 😵‍💫


def load_aglpn_dataset():
    aglpn = load_dataset(
        str(project_root / "data" / "protest"),
        data_files={"main": "protest_news_annotated_v1.jsonl"},
    )
    aglpn["main"] = aglpn["main"].filter(lambda x: x["answer"] != "ignore")
    traindev_test = aglpn["main"].train_test_split(test_size=3 / 8, seed=20230206)
    train_dev = traindev_test["train"].train_test_split(test_size=1 / 5, seed=20230206)
    aglpn = DatasetDict(
        {
            "train": train_dev["train"],
            "dev": train_dev["test"],
            "test": traindev_test["test"],
        }
    )
    aglpn = aglpn.map(lambda x: {"label": 1 if x["accept"] == ["relevant"] else 0})
    # protest_news = protest_news.cast_column("accept", ClassLabel(names=["irrelevant", "relevant"]))
    # protest_news = glpn.rename_column("accept", "label")
    return aglpn
