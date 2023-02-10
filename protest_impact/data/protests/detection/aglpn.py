from datasets import ClassLabel, DatasetDict, load_dataset
from protest_impact.util import project_root

# AGLPN = another german local protest news dataset? 😵‍💫


def load_aglpn_dataset():
    aglpn = load_dataset(
        str(project_root / "data" / "protest"),
        data_files={"main": "protest_news_annotated_v1.jsonl"},
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
    traindev_test = aglpn["main"].train_test_split(test_size=4 / 10, seed=20230206)
    train_dev = traindev_test["train"].train_test_split(test_size=1 / 6, seed=20230206)
    aglpn = DatasetDict(
        {
            "train": train_dev["train"],
            "dev": train_dev["test"],
            "test": traindev_test["test"],
        }
    )
    return aglpn
