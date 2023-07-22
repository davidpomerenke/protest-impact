import pandas as pd
from datasets import ClassLabel, load_dataset
from protest_impact.util import project_root


def load_glpn_dataset():
    """
    Loads the German Local Protest News dataset (cf. Wiedemann et al. 2022).
    Available on request from:
    https://zenodo.org/record/7308406
    """
    glpn_path = project_root / "datasets" / "glpn_v1.1"
    data_files = {
        "train": str(glpn_path / "glpn_train.csv"),
        "dev": str(glpn_path / "glpn_dev.csv"),
        "test": str(glpn_path / "glpn_test.csv"),
        "test.time": str(glpn_path / "glpn_test-time.csv"),
        "test.loc": str(glpn_path / "glpn_test-loc.csv"),
    }
    glpn = load_dataset("csv", data_files=data_files)
    glpn = glpn.map(
        function=(lambda x: {"labels": 1 if x["labels"] == "relevant" else 0})
    )
    glpn = glpn.rename_column("labels", "label")
    glpn = glpn.rename_column("excerpt", "text")
    return glpn
