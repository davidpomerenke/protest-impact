import os

import openai
from dotenv import load_dotenv
from ratelimiter import RateLimiter

from src.cache import cache

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
rate_limiter = RateLimiter(max_calls=2, period=1)


@cache
def create_completion(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def _ask_gpt(prompt, user_msg, model="gpt-3.5-turbo"):
    completion = rate_limiter(create_completion)(
        model=model,
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
    )
    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens
    cost = in_tokens / 1_000 * 0.0015 + out_tokens / 1_000 * 0.002
    return cost, completion.choices[0].message.content.strip()


def ask_gpt(prompt, user_msg, model="gpt-3.5-turbo"):
    if len(user_msg.split()) > 3000:
        user_msg = " ".join(user_msg.split()[:1000])
    try:
        return _ask_gpt(prompt, user_msg, model)
    except openai.error.InvalidRequestError:
        return None
