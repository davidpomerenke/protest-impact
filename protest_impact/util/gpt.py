import os

import openai
from dotenv import load_dotenv

from protest_impact.util import cache

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


@cache
def _query(
    prompt,
    engine="text-davinci-003",
    temperature=0,
    max_tokens=2000,
    salt=None,
    **kwargs
):
    response = openai.Completion.create(
        prompt=prompt,
        engine=engine,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )
    return response


def query(prompt, temperature=0, max_tokens=2000, salt=None, **kwargs):
    result = _query(
        prompt, temperature=temperature, max_tokens=max_tokens, salt=salt, **kwargs
    )
    cost = result["usage"]["total_tokens"] * 0.02 / 1000
    return cost, result["choices"][0]["text"].strip()
