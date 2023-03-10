import requests
from joblib import Memory

from protest_impact.util.path import project_root

memory = Memory(project_root / ".cache", verbose=0)
cache = memory.cache


@cache
def get(url, params=None, **kwargs):
    """Sends a GET request.

    :param url: URL for the new :class:`Request` object.
    :param params: (optional) Dictionary, list of tuples or bytes to send
        in the query string for the :class:`Request`.
    :param \*\*kwargs: Optional arguments that ``request`` takes.
    :return: :class:`Response <Response>` object
    :rtype: requests.Response
    """

    return requests.get(url, params=params, **kwargs)
