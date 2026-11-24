import asyncio
import sys
from typing import ClassVar

sys.path.insert(0, "/tmp/screencast")
from playwright.async_api import async_playwright
from record import OVERLAY, Demo


class R:
    fast: ClassVar[list[object]] = []


async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path="/usr/bin/google-chrome")
        ctx = await b.new_context(viewport={"width": 1600, "height": 900})
        await ctx.add_init_script(OVERLAY)
        page = await ctx.new_page()
        await page.goto("http://localhost:9123/")
        await page.wait_for_timeout(3000)
        d = Demo(page, R())
        for t in (
            "Lateral Meniscus",
            "Medial Meniscus",
            "Femur",
            "Tibia",
            "Patella",
            "Patellar Cartilage",
            "Femoral Cartilage",
            "Lateral Tibial Cartilage",
            "Medial Tibial Cartilage",
        ):
            print(
                t,
                await d.structure_row(t).inner_text(),
                await d.structure_group(t).locator(".v-list-group__items .v-list-item", has_text="Mesh").count(),
            )
        for icon in ("mdi-file-import-outline", "mdi-folder-open-outline", "mdi-cube-scan", "mdi-box-cutter", "mdi-crop-free"):
            print(icon, await d.toolbar_button(icon).count())
        print("selects", await page.locator(".v-toolbar .v-select").count())
        await d.open_child("Lateral Meniscus", "Mesh")
        print("elt select", await page.locator(".v-select", has_text="Element type").count())
        print("mesh btn", await page.locator("button", has_text="Mesh").filter(has=page.locator("i.mdi-play")).count())
        await d.click(d.toolbar_button("mdi-box-cutter"))
        print("thumb", await page.locator(".v-toolbar .v-slider-thumb").count(), "pin", await d.toolbar_button("mdi-pin-off-outline").count())
        await d.click(d.toolbar_button("mdi-box-cutter"))
        print("labels group", await page.locator(".v-list-item", has_text="Labels").first.inner_text())
        await page.screenshot(path="probe3.png")
        await b.close()


asyncio.run(main())
