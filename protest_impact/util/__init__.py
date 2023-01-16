import re
from pprint import pprint


from protest_impact.util.cache import cache, get, memory
from protest_impact.util.html import html2text
from protest_impact.util.path import project_root


def website_name(url):
    return re.search(
        r"^(?:https?:\/\/)?(?:[^@\/\n]+@)?(?:www\.)?([^:\/?\n]+)", url
    ).group(1)


counter = 0


def log(obj, name=None):
    global counter
    counter += 1
    name = name or "obj" + str(counter)
    if name:
        print(f"{name} = ", end="")
    pprint(obj)
    return obj
