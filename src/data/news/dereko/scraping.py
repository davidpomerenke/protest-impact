import json
from os import environ
from pathlib import Path

import dateparser
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from src import project_root
from src.data.protests.keywords import climate_queries

load_dotenv()

cookie_path = Path("cookies.json")


async def scrape_count_table(
    corpus, code, title, query, headless=True
) -> pd.DataFrame | None:
    path = project_root / "data/ids-dereko/counts" / corpus / code
    path.mkdir(parents=True, exist_ok=True)
    csv_path = path / f"{query}.csv"
    html_path = path / f"{query}.html"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    if not html_path.exists():
        try:
            page, browser, context = await start(headless=headless)
            page, browser = await prepare(page, browser, context)
            page, browser = await select_newspaper(corpus, title, page, browser)
            q = climate_queries(mode="dereko", short=True)[query]
            page, browser = await search(q, page, browser)
            html = await get_table(page, browser)
            with open(path / f"{query}.html", "w") as f:
                f.write(html)
        except Exception as e:
            await browser.close()
            with open(path / f"{query}.log", "w") as f:
                f.write(str(e))
            return
        finally:
            await browser.close()
    html = html_path.read_text()
    df = parse_html_table(html)
    df.to_csv(csv_path, index=False)
    return df


async def start(headless=True) -> tuple[Page, Browser, BrowserContext]:
    p = await async_playwright().start()
    browser = await p.chromium.launch(
        timeout=10_000, downloads_path=Path("_downloads"), headless=headless
    )
    context = await browser.new_context()
    if cookie_path.exists():
        await context.add_cookies(json.loads(cookie_path.read_text()))
    page = await context.new_page()
    return page, browser, context


async def click(page: Page, selector: str, n: int = 0, timeout=5_000) -> Page:
    await page.wait_for_selector(selector, timeout=timeout)
    els = await page.query_selector_all(selector)
    await els[n].click()
    return page


async def prepare(
    page: Page, browser: Browser, context: BrowserContext
) -> tuple[Page, Browser]:
    await page.goto(
        "https://cosmas2.ids-mannheim.de/cosmas2-web/faces/investigation/archive.xhtml"
    )
    if cookie_path.exists():
        try:
            await click(page, "text=Archive")
            return page, browser
        except Exception as e:
            print("Cannot click on 'Archive'.")
            print(e)
            try:
                await click(page, "text=Korpusverwaltung")
                await click(page, "text=Archive")
                return page, browser
            except Exception as e:
                print("Cannot click on 'Korpusverwaltung'.")
                print(e)
    await click(page, "text=Anmeldung")
    await click(page, "text=Login")
    await page.fill('input[name="loginForm:userName"]', environ["DEREKO_USERNAME"])
    await page.fill('input[name="loginForm:password"]', environ["DEREKO_PASSWORD"])
    await click(page, 'input[name="loginForm:j_idt472"]')
    await click(page, "text=OK")
    cookies = await context.cookies()
    cookie_path.write_text(json.dumps(cookies))
    await click(page, "text=Archive")
    return page, browser


async def select_newspaper(
    corpus: str, title: str, page: Page, browser: Browser
) -> tuple[Page, Browser]:
    await page.wait_for_timeout(1_000)
    corpus = corpus.replace("W1", "W")
    await click(page, f"text={corpus} - Archiv der geschriebenen Sprache", -1)
    await page.wait_for_timeout(1_000)
    await click(page, f':has-text("{title}")', -1)
    return page, browser


async def search(query: str, page: Page, browser: Browser) -> tuple[Page, Browser]:
    await page.fill('textarea[name="form:queryString"]', query)
    # await page.set_checked('input[name="form:defaultQueryOperator:0"]', True)
    await page.click("text=Suchen")
    # await page.click('input[name="form:j_idt129"]', timeout=300_000)
    return page, browser


async def get_table(page: Page, browser: Browser) -> tuple[Page, Browser]:
    await page.inner_html("id=form:resultTable", timeout=600_000)
    await page.wait_for_timeout(3_000)
    html = await page.inner_html("id=form:resultTable")
    return html


def parse_html_table(html_content: str) -> pd.DataFrame:
    headers = ["hit_count", "text_count", "date"]
    soup = BeautifulSoup(html_content, "html.parser")
    all_rows = soup.find_all("tr")
    data = []
    for row in all_rows:
        columns = row.find_all("td")
        if len(columns) == 4:
            hit_count = columns[1].get_text(strip=True)
            text_count = columns[2].get_text(strip=True)
            date_str = columns[3].get_text(strip=True)
            date = date = dateparser.parse(date_str, languages=["de"])
            data.append([hit_count, text_count, date])
    return pd.DataFrame(data, columns=headers)[["date", "hit_count", "text_count"]]


async def close(page: Page, browser: Browser) -> None:
    await page.click("text=Abmeldung")
    await page.click("text=Logout")
    await browser.close()
