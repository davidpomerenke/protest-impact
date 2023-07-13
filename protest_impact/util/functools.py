from functools import partial


def function_name(f):
    """
    Returns the name of either a function or a partial function.
    """
    if isinstance(f, partial):
        return f.func.__name__
    else:
        return f.__name__
