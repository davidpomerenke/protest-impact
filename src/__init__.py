from functools import partial

import pandas as pd

# HELPER FUNCTIONS


def kill_umlauts_without_mercy(s: str) -> str:
    # pointless bloodshedding
    # just for the cause of normalization
    return (
        s.replace("ä", "a")
        .replace("ö", "o")
        .replace("ü", "u")
        .replace("ß", "ss")
        .replace("Ä", "A")
        .replace("Ö", "O")
        .replace("Ü", "U")
    )


def function_name(f):
    """
    Returns the name of either a function or a partial function.
    """
    if isinstance(f, partial):
        return f.func.__name__
    else:
        return f.__name__


start = pd.Timestamp("2020-01-01")
end = pd.Timestamp("2022-12-31")
