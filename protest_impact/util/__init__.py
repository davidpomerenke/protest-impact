import re
from pprint import pprint

from protest_impact.util.cache import cache, get, memory
from protest_impact.util.html import html2text, website_name
from protest_impact.util.path import fulltext_path, project_root

counter = 0


def log(obj, name=None):
    global counter
    counter += 1
    name = name or "obj" + str(counter)
    if name:
        print(f"{name} = ", end="")
    pprint(obj)
    return obj
