# copied and adapted from https://gist.github.com/rolisz/1b93e60f5b9a85fb5a5b79913fd0ad4c
# (c) Roland Szabo 2022

import copy
from typing import Iterable, List, Optional, Union

import spacy
from prodigy import get_stream, log, recipe
from prodigy.models.matcher import PatternMatcher
from prodigy.types import RecipeSettingsType
from prodigy.util import get_labels


@recipe(
    "textcat.manual_patterns",
    # fmt: off
    dataset=("Dataset to save annotations to", "positional", None, str),
    source=("Data to annotate (file path or '-' to read from standard input)", "positional", None, str),
    spacy_model=("Loadable spaCy pipeline or blank:lang (e.g. blank:en)", "positional", None, str),
    labels=("Comma-separated label(s) to annotate or text file with one label per line", "option", "l", get_labels),
    patterns=("Path to match patterns file", "option", "pt", str),
    # fmt: on
)
def manual(
    dataset: str,
    source: Union[str, Iterable[dict]],
    spacy_model: str,
    labels: Optional[List[str]] = None,
    patterns: Optional[str] = None,
) -> RecipeSettingsType:
    """
    Manually annotate categories that apply to a text. If more than one label
    is specified, categories are added as multiple choice options. If the
    --exclusive flag is set, categories become mutually exclusive, meaning that
    only one can be selected during annotation.
    """
    log("RECIPE: Starting recipe textcat.manual", locals())
    log(f"RECIPE: Annotating with {len(labels)} labels", labels)
    stream = get_stream(source, rehash=True, dedup=True, input_key="text")
    nlp = spacy.load(spacy_model)

    matcher = PatternMatcher(
        nlp,
        prior_correct=5.0,
        prior_incorrect=5.0,
        label_span=False,
        label_task=True,
        filter_labels=labels,
        combine_matches=True,
        task_hash_keys=("label",),
    )
    matcher = matcher.from_disk(patterns)
    stream = add_suggestions(stream, matcher, labels)

    return {
        "view_id": "choice",
        "dataset": dataset,
        "stream": stream,
        "config": {
            "labels": labels,
            "choice_style": "single",
            "choice_auto_accept": True,
            "exclude_by": "task",
            "auto_count_stream": True,
        },
    }


def add_suggestions(stream, matcher, labels):
    texts = (eg for score, eg in matcher(stream))
    options = [{"id": label, "text": label} for label in labels]

    for eg in texts:
        task = copy.deepcopy(eg)

        task["options"] = options
        if "label" in task:
            task["accept"] = [task["label"]]
            del task["label"]
        yield task
