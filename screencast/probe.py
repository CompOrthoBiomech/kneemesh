import asyncio
from playwright.async_api import async_playwright
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path="/usr/bin/google-chrome")
        page = await b.new_page(viewport={"width": 1600, "height": 900})
        page.on("console", lambda m: print("console:", m.type, m.text) if m.type in ("error","warning") else None)
        await page.goto("http://localhost:9123/")
        await page.wait_for_timeout(4000)
        await page.screenshot(path="probe1.png")
        await page.get_by_text("Add Label").click()
        await page.wait_for_timeout(1500)
        await page.screenshot(path="probe2.png")
        await b.close()
asyncio.run(main())
