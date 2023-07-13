import json
from os import environ
from pathlib import Path

from dotenv import load_dotenv
from playwright.async_api import async_playwright

load_dotenv()


async def prepare():
    p = await async_playwright().start()
    browser = await p.firefox.launch(
        timeout=10_000, downloads_path=Path("_downloads"), headless=False
    )
    context = await browser.new_context()
    cookie_path = Path("cookies.json")
    if cookie_path.exists():
        await context.add_cookies(json.loads(cookie_path.read_text()))
    page = await context.new_page()
    await page.goto("https://cosmas2.ids-mannheim.de/cosmas2-web/")
    if cookie_path.exists():
        await page.click("text=Recherche")
    else:
        await page.click("text=Anmeldung")
        await page.click("text=Login")
        await page.fill('input[name="loginForm:userName"]', environ["DEREKO_USERNAME"])
        await page.fill('input[name="loginForm:password"]', environ["DEREKO_PASSWORD"])
        await page.click('input[name="loginForm:j_idt505"]')
        await page.click("text=OK")
        # cookies = await context.cookies()
        # cookie_path.write_text(json.dumps(cookies))
    await page.click("text=W - Archiv der geschriebenen Sprache")
    await page.click("text=W-öffentlich - alle")
    await page.fill('textarea[name="form:queryString"]', "klimaschutz")
    # await page.set_checked('input[name="form:defaultQueryOperator:0"]', True)
    # await page.click("text=Suchen")
    # await page.click('input[name="form:j_idt129"]')
    await page.wait_for_timeout(30_000)
    return page, browser


async def clickthrough(page, browser):
    await page.bring_to_front()
    await page.click("text=Volltext")
    n = int((await page.inner_text("id=form:hitCount")).replace(".", ""))
    download_limit = 10_000
    page_limit = 200
    start = 1_250_000
    for i in range(start // download_limit, n // download_limit + 1):
        await page.click("text=Volltext")
        try:
            await page.click('button[name="form:btn"]')
            await page.click("text=ges. KWIC deaktivieren")
        except Exception:
            await page.wait_for_timeout(5000)
            await page.click('button[name="form:btn"]')
            await page.wait_for_timeout(5000)
            await page.click("text=ges. KWIC deaktivieren")
        for j in range(download_limit // page_limit):
            await page.fill(
                'input[name="form:goToFulltextIndexField"]',
                str(i * download_limit + j * 200 + 1),
            )
            await page.click('button[name="form:j_idt140"]')  # springen
            await page.click('button[name="form:btn"]')
            await page.click("text=Seite aktivieren")
        await page.click("text=Export")
        await page.select_option('select[name="form:j_idt102"]', "RTF (neu)")
        await page.set_checked('input[name="form:j_idt120"]', False)  # KWIC
        await page.set_checked(
            'input[name="form:j_idt154"]', True
        )  # nur ausgewählte Treffer
        await page.set_checked('input[name="form:j_idt169"]', True)  # exportieren
        await page.set_checked('input[name="form:j_idt171"]', False)  # nachher
        await page.set_checked('input[name="form:j_idt173"]', False)  # fett
        await page.click('input[name="form:j_idt192"]')  # exportieren
        await page.wait_for_timeout(90_000)
        async with page.expect_download() as download_info:
            await page.click(
                'button[name="exportDialogForm:downloadButton"]'
            )  # download
        download = await download_info.value
        path = await download.path()
        await download.save_as(f"downloads/{str(i*download_limit)}.rtf")
        await page.click('button[name="exportDialogForm:j_idt636"]')  # schließen
    return page, browser


async def close(page, browser):
    await page.click("text=Abmeldung")
    await page.click("text=Logout")
    await browser.close()
