import asyncio
import json
import re
import shutil
from calendar import monthrange
from datetime import date
from os import environ
from pathlib import Path
from zipfile import ZipFile

import dateparser
import pandas as pd
from dotenv import load_dotenv
from munch import Munch
from playwright.async_api import Browser, BrowserContext, Page, async_playwright
from striprtf.striprtf import rtf_to_text

from src.paths import external_data

load_dotenv()

cookie_path = Path("cookies.json")

permalink_climate = "https://advance-lexis-com.mu.idm.oclc.org/api/permalink/6bd70d80-8b5c-46dc-8a95-803a79780ca2/?context=1516831"


async def main():
    await scrape("klima*", headless=False)


def process_dowloads():
    for path in sorted((external_data / "nexis/zip").glob("**/*.zip")):
        process(path)


async def scrape(
    query=None, headless=True, start=2020, end=2022
) -> pd.DataFrame | None:
    try:
        page, browser, context = await setup(headless=headless)
        page, browser, context = await login(page, browser, context)
        page, browser, context = await search(query, page, browser, context)
        for year in range(start, end):
            for month in range(1, 13):
                res = await search_by_month(year, month, page, browser, context)
                if res is None:
                    continue
                page, browser, context = res
                page, browser, context = await download(
                    1_000, year, month, page, browser, context
                )
        # await page.wait_for_timeout(600_000)
        await context.add_cookies(json.loads(cookie_path.read_text()))
    except Exception as e:
        print(e)
        await page.wait_for_timeout(600_000)
    finally:
        await browser.close()


async def setup(headless=True) -> tuple[Page, Browser, BrowserContext]:
    path = external_data / "nexis" / "tmp"
    path.mkdir(parents=True, exist_ok=True)
    p = await async_playwright().start()
    browser = await p.chromium.launch(
        timeout=10_000, downloads_path=path, headless=headless
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


async def login(
    page: Page, browser: Browser, context: BrowserContext
) -> tuple[Page, Browser, BrowserContext]:
    if not cookie_path.exists():
        await page.goto("http://umlib.nl/lexis_go")
        element = await page.query_selector('text="Maastricht University"')
        button = await element.query_selector("xpath=../..//button")
        await button.dispatch_event("click")
        await page.fill('input[id="userNameInput"]', environ["UNI_USER"])
        await page.fill('input[id="passwordInput"]', environ["UNI_PASSWORD"])
        await click(page, 'span[id="submitButton"]')
        await page.wait_for_timeout(5_000)
        cookies = await context.cookies()
        cookie_path.write_text(json.dumps(cookies))
    else:
        await page.goto("https://advance-lexis-com.mu.idm.oclc.org/")
    return page, browser, context


async def search(
    query: str,
    page: Page,
    browser: Browser,
    context: BrowserContext,
) -> tuple[Page, Browser, BrowserContext]:
    await page.fill("lng-expanding-textarea", query)
    await click(page, "lng-search-button")
    try:
        await click(page, 'span[class="filter-text"]', timeout=15_000)
        await page.wait_for_timeout(5_000)
    except Exception:
        pass
    await click(page, 'button[data-filtertype="source"]')
    await page.wait_for_timeout(3_000)
    el = await page.query_selector('input[data-value="dpa-AFX ProFeed"]')
    await el.dispatch_event("click")
    await page.wait_for_timeout(5_000)
    await click(page, 'span[id="sortbymenulabel"]')
    await page.wait_for_timeout(1_000)
    await click(page, 'button[data-value="dateascending"]')
    await page.wait_for_timeout(5_000)
    return page, browser, context


async def search_by_month(
    year: int, month: int, page: Page, browser: Browser, context: BrowserContext
) -> tuple[Page, Browser, BrowserContext] | None:
    existing_files = (external_data / "nexis/zip").glob(f"{year}-{month:02d}/*.zip")
    if any([a for a in existing_files if not a.name.endswith("00.zip")]):
        # then we already have all files for this month
        return None
    try:
        await click(page, 'span[class="filter-text"]', n=1)
    except Exception:
        pass
    await page.wait_for_timeout(5_000)
    try:
        await click(
            page, 'button[data-filtertype="datestr-news"][data-action="expand"]'
        )
    except Exception as e:
        print(e)
        pass
    await page.wait_for_timeout(3_000)
    await page.fill('input[class="min-val"]', f"01/{month}/{year}")
    await page.wait_for_timeout(2_000)
    day = monthrange(year, month)[1]
    await page.fill('input[class="max-val"]', f"{day}/{month}/{year}")
    await page.wait_for_timeout(2_000)
    await click(page, 'div[class="date-form"]')
    await page.wait_for_timeout(1_000)
    await click(page, 'button[class="save btn secondary"]')
    await page.wait_for_timeout(10_000)
    return page, browser, context


async def visit_permalink(
    permalink: str, page: Page, browser: Browser, context: BrowserContext
) -> tuple[Page, Browser, BrowserContext]:
    await page.goto(permalink)
    await click(page, 'input[data-action="viewlink"]')
    await page.wait_for_timeout(10_000)
    return page, browser, context


async def download(
    n: int, year, month, page: Page, browser: Browser, context: BrowserContext
) -> tuple[Page, Browser, BrowserContext]:
    el = await page.query_selector('header[class="resultsHeader"]')
    n_results = int(re.search(r"\((\d+)\)", await el.inner_text()).group(1))
    for i in range(0, min(n_results, n), 100):
        range_ = f"{i+1}-{min(i+100, n_results)}"
        dest_path = external_data / f"nexis/zip/{year}-{month:02d}/{range_}.zip"
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if dest_path.exists():
            continue
        await click(page, 'span[class="icon la-Download"]')
        await page.wait_for_timeout(2_000)
        await page.fill('input[id="SelectedRange"]', range_)
        async with page.expect_download(timeout=120_000) as download_info:
            await click(page, 'button[data-action="download"]')
        download = await download_info.value
        tmp_path = await download.path()
        shutil.move(tmp_path, dest_path)
        print(f"Downloaded {dest_path}")
        process(dest_path)
        await page.wait_for_timeout(2_000)
    return page, browser, context


def unpack(path: Path) -> list[str]:
    with ZipFile(path) as zipObj:
        plaintexts = []
        for file in zipObj.filelist:
            rtf = zipObj.read(file).decode("utf-8")
            plaintext = rtf_to_text(rtf, errors="ignore").strip()
            plaintext = plaintext.replace("\xa0", " ")
            plaintexts.append(plaintext)
    return plaintexts


def parse(plaintext: str) -> Munch:
    title, rest = plaintext.split("\n", 1)
    feed, rest = rest.split("\n", 1)
    date, rest = rest.split("\n", 1)
    date = dateparser.parse(date.strip(), languages=["de"])
    meta, rest = rest.split("Body", 1)
    if rest.strip().startswith("Zusammenfassung\n"):
        _, summary, rest = rest.strip().split("\n", 2)
        summary = summary.strip()
    else:
        summary = None
    try:
        location, rest = re.split(r" ?\(dpa| ?\(dap", rest, 1)
        location = location.strip()
        if rest.startswith("/"):
            region, rest = re.split(r"\) ?- ?|\) ?– ?", rest[1:], 1)
            region = region.strip()
        else:
            region = ""
            rest = rest[4:]
    except ValueError:
        location = None
        region = None
    if "Graphic" in rest:
        text, _ = rest.split("Graphic", 1)
    else:
        text, _ = rest.split("Load-Date", 1)
    return Munch(
        title=title.strip(),
        date=date.isoformat(),
        summary=summary,
        location=location,
        region=region,
        text=text.strip(),
    )


def process(path: Path):
    texts = unpack(path)
    for text in texts:
        item = parse(text)
        datestr = date.strftime(dateparser.parse(item.date), "%Y-%m-%d")
        jpath = external_data / "nexis/json" / datestr / f"{item.title[:50]}.json"
        jpath.parent.mkdir(parents=True, exist_ok=True)
        jpath.write_text(json.dumps(item, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
