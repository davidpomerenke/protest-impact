import json

import holidays
import pandas as pd
from dateparser import parse
from geopy.geocoders import Nominatim
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from src.cache import cache
from src.data import german_region_names
from src.data.protests.german_protest_reports.gpt import ask_gpt
from src.paths import external_data, interim_data

geolocator = Nominatim(user_agent="protest-impact")


def read_data():
    with open(external_data / "mediacloud/protest_news_predicted.jsonl") as f:
        items = [json.loads(line) for line in tqdm(f)]
    return items


# path = interim_data / "german_protest_reports"
# labelpath = path / "labels"
# labelpath.mkdir(exist_ok=True, parents=True)

schema = """
{
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "IS_CLIMATE_PROTEST_EVENT": {
                "type": "boolean"
            },
            "PAST_OR_FUTURE": {
                "type": "string",
                "enum": ["PAST", "FUTURE"]
            },
            "COUNTRY": {
                "type": "string",
                "enum": ["DE", "AT", "CH", "OTHER"]
            },
            "CITY": {
                "type": "string"
            },
            "PROTEST_DATE_YEAR": {
                "type": "integer"
            },
            "PROTEST_DATE_MONTH": {
                "type": "string",
                "enum": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
            },
            "PROTEST_DATE_DAY": {
                "type": "integer"
            },
            "PROTEST_GROUP": {
                "type": "string",
                "enum": ["FFF", "ALG", "XR", "EG", "GP", "OTHER_CLIMATE_GROUP"]
            },
            "N_PARTICIPANTS": {
                "type": "integer"
            },
            "DESCRIPTION": {
                "type": "string"
            }
        },
        "required": ["IS_CLIMATE_PROTEST_EVENT"]
    }
}
"""


system_prompt = f"""
Classify the following text as a climate protest event or not, and - if applicable - fill out the details.
Use German-language city names.
For the DESCRIPTION field, write a one-sentence summary of the event in English language, including the goal of the protest, the protest form, and - if applicable - confrontations with the police; as well as any other relevant information.
The schema allows you to create multiple event objects in case that the text describes multiple events.
Use the following schema:

```json{schema}```
"""


@cache
def get_region(city):
    coordinates = geolocator.geocode(city + ", Germany")
    if coordinates is None:
        print(f"Could not find coordinates for {city}")
        return None
    regions = [r for r in german_region_names if r in coordinates.address]
    return regions[0] if len(regions) > 0 else None


holiday_names = [i[0] for i in holidays.Germany(years=range(2018, 2023)).items()]


@cache
def get_coding(row):
    # get weekdays and holidays from the weeks before and after the protest
    date_helper_text = ""
    row.date = parse(row.date)
    dates = pd.date_range(
        row.date - pd.Timedelta(days=7), row.date + pd.Timedelta(days=7)
    )
    for date in dates:
        # weekday
        date_helper_text += f"{date.strftime('%A')}, "
        # date
        date_helper_text += f"{date.day}. {date.month_name()} {date.year}"
        # holiday
        if date in holidays.Germany(years=range(2018, 2023)):
            date_helper_text += f", {holidays.Germany(years=range(2018, 2023))[date]}"
        date_helper_text += "\n"
    content = date_helper_text + "\n\n"
    content += f"{row.title}\n\n{row.date.date()}\n\n{row.text}"
    content += "\n\n```json\n"
    return ask_gpt(system_prompt, content)


@cache
def parse_response(response):
    try:
        parsed_responses = []
        responses_ = json.loads(response)
        if isinstance(responses_, str):
            return []
        for parsed_response in responses_:
            if not isinstance(parsed_response, dict):
                continue
            if (
                parsed_response["IS_CLIMATE_PROTEST_EVENT"] is True
                and parsed_response["PAST_OR_FUTURE"] == "PAST"
                and parsed_response["COUNTRY"] == "DE"
            ):
                city = (parsed_response["CITY"] or "").split(",")
                parsed_response["CITY"] = city[0]
                parsed_response["PROTEST_DATE"] = parse(
                    f"{parsed_response['PROTEST_DATE_DAY']} {parsed_response['PROTEST_DATE_MONTH']} {parsed_response['PROTEST_DATE_YEAR']}"
                )
                parsed_response["REGION"] = get_region(parsed_response["CITY"])
                parsed_response = {k.lower(): v for k, v in parsed_response.items()}
                parsed_responses.append(parsed_response)
        return parsed_responses
    except json.decoder.JSONDecodeError as e:
        print(e)
        print(response)
        print()
        return []


def coding(limit: int = None):
    items = read_data()
    climate_items = [i for i in items if "Klima" in i["title"] or "Klima" in i["text"]]
    df = pd.DataFrame(climate_items)
    responses = Parallel(n_jobs=1)(
        delayed(get_coding)(row) for i, row in tqdm(list(df.iterrows())[:limit])
    )
    responses = [r for r in responses if r is not None]
    print("Done")
    costs, responses = zip(*responses)
    print(f"Total cost: {sum(costs)}")
    parsed_responses = Parallel(n_jobs=1)(
        delayed(parse_response)(response) for response in tqdm(responses)
    )
    items = [item for sublist in parsed_responses for item in sublist]
    df = pd.DataFrame(items)
    df = df.rename(
        columns={
            "protest_group": "actor",
            "n_participants": "size",
            "city": "location",
            "description": "notes",
        }
    )
    df = df[["protest_date", "region", "location", "actor", "size", "notes"]]
    df = df[df.location.notna() & df.region.notna()]
    print(df.columns)
    print(len(df))
    df = df.drop_duplicates().sort_values(
        ["protest_date", "region", "location", "actor"]
    )
    df.to_csv(interim_data / "german-protest-reports/protests.csv", index=False)


if __name__ == "__main__":
    coding()
