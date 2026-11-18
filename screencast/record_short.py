"""Record a captioned 30-second 4K cut of kneemesh-gui for social media, reusing record.py's cursor, recorder, and helpers.

Run the GUI first from the repository root: uv run kneemesh-gui --server --port 9123 --view-ratio 3
Then: uv run --with playwright python screencast/record_short.py [--url http://localhost:9123/] [--out short.mp4]

The configurations are imported and the camera turned upright before recording starts. Each job's wait is cut to a brief
fast-forward flash, so building and meshing seem to take a moment.
"""

import argparse
import asyncio
import math
import shutil
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from playwright.async_api import async_playwright
from record import DEVICE_SCALE, HEIGHT, OVERLAY, WIDTH, Demo, Recorder

# Seconds of video each skipped wait takes.
FLASH = 0.6
REPOSITORY = "github.com/CompOrthoBiomech/kneemesh"

STYLE = """
#demo-caption { bottom: 48px; max-width: 1080px; font: 600 34px/1.3 Roboto, sans-serif; padding: 16px 30px; border-left-width: 7px; }
#demo-skip { position: fixed; left: calc(440px + (100vw - 440px) / 2); top: 50%; transform: translate(-50%, -50%);
  font: 700 64px/1 Roboto, sans-serif; color: #fff; background: rgba(15, 18, 28, 0.8); padding: 18px 34px; border-radius: 16px;
  pointer-events: none; z-index: 2147483641; display: none; }
#demo-skip.show { display: block; }
#demo-end { position: fixed; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 22px;
  background: rgba(10, 12, 20, 0.88); color: #fff; font-family: Roboto, sans-serif; pointer-events: none; z-index: 2147483644;
  opacity: 0; transition: opacity 0.5s; }
#demo-end.show { opacity: 1; }
#demo-end .title { font-size: 72px; font-weight: 700; }
#demo-end .title span { color: #ffc400; }
#demo-end .link { font-size: 34px; color: #ddd; }
"""

ELEMENTS = f"""
() => {{
  const skip = document.createElement('div');
  skip.id = 'demo-skip';
  skip.textContent = '⏩';
  const end = document.createElement('div');
  end.id = 'demo-end';
  end.innerHTML = '<div class="title">kneemesh <span>·</span> open source</div><div class="link">{REPOSITORY}</div>';
  document.body.append(skip, end);
}}
"""


class ShortRecorder(Recorder):
    """A recorder that cuts each skipped period to FLASH seconds of video."""

    def __init__(self, directory: Path) -> None:
        super().__init__(directory)
        # Periods of wall time, each cut to the given seconds of video.
        self.skipped: list[tuple[float, float, float]] = []
        self.lags: list[float] = []

    def _frame(self, params: dict) -> None:
        super()._frame(params)
        self.lags.append(time.time() - params["metadata"]["timestamp"])

    def _video_time(self, t: float) -> float:
        # Frames carry Chrome's timestamps and the periods this process's clock; the least lag of a frame's arrival behind its
        # timestamp is the clocks' offset.
        offset = min(self.lags)
        out = t
        for start, end, target in ((start - offset, end - offset, target) for start, end, target in self.skipped):
            if t > start:
                elapsed = min(t, end) - start
                out -= elapsed - target * elapsed / (end - start)
        return out

    def encode(self, output: Path) -> None:
        """Encode as Recorder does, keeping one frame per 60th of a second of video, since cuts pack thousands into a moment."""
        kept: list[tuple[float, Path]] = []
        slot = -1
        for t, path in sorted(self.frames):
            # The last frame of each 60th of a second of video.
            if kept and int(self._video_time(t) * 60) == slot:
                kept[-1] = (t, path)
            else:
                kept.append((t, path))
                slot = int(self._video_time(t) * 60)
        self.frames = kept
        super().encode(output)


class ShortDemo(Demo):
    async def move(self, x: float, y: float, seconds: float | None = None) -> None:
        """Move as Demo does, but faster by default."""
        await super().move(x, y, seconds if seconds is not None else min(0.6, 0.2 + math.hypot(x - self.x, y - self.y) / 2400))

    async def say(self, text: str | None, hold: float = 0.0) -> None:
        await self.caption(text, "", hold)

    async def skip(self, action: Callable[[], Awaitable[None]]) -> None:
        """Run action behind a fast-forward badge, cut to FLASH seconds in the video."""
        assert isinstance(self.recorder, ShortRecorder)
        start = time.time()
        await self.page.evaluate("() => document.getElementById('demo-skip').classList.add('show')")
        await action()
        await self.page.evaluate("() => document.getElementById('demo-skip').classList.remove('show')")
        await self.pause(0.1)
        self.recorder.skipped.append((start, time.time(), FLASH))

    async def drag(self, path: list[tuple[float, float]], seconds: float = 2.0) -> None:
        """Drag as Demo does, its hold at the end, which the remote view needs, cut short in the video."""
        assert isinstance(self.recorder, ShortRecorder)
        await self.move(*path[0], 0.35)
        await self.page.mouse.down()
        legs = len(path) - 1
        for x, y in path[1:]:
            await self.move(x, y, seconds / legs)
        start = time.time()
        for dx in (0.5, 0.0):
            await self.pause(0.2)
            await self.page.mouse.move(self.x + dx, self.y)
        await self.pause(0.2)
        await self.page.mouse.up()
        await self.pause(0.3)
        self.recorder.skipped.append((start, time.time(), 0.2))

    async def press(self, icon: str) -> None:
        """Click a toolbar button and move off it before its tooltip shows."""
        await self.click(self.toolbar_button(icon), 0.05)
        await self.move(self.x, self.y + 160, 0.25)

    async def import_config(self, *path: str) -> None:
        await self.click(self.toolbar_button("mdi-file-import-outline"), 0.8)
        for name in path:
            await self.dialog_item(name)
        await self.dialog_open()
        await self.wait_idle(fast=False)

    def mesh_button(self):
        return self.page.locator("button", has_text="Mesh").filter(has=self.page.locator("i.mdi-play")).first


async def prepare(demo: ShortDemo) -> None:
    """Load the knee's labels and settings and stand the knee upright, before recording."""
    page = demo.page
    await demo.click(demo.toolbar_button("mdi-console"), 0.5)
    await demo.import_config("examples", "knee_brep_hex.json")
    await demo.import_config("knee_mesh_hex8.json")
    await demo.click(page.locator(".v-list-item", has_text="Labels").first, 0.6)
    splitter = page.locator(".kneemesh-splitter")
    box = await splitter.bounding_box()
    assert box is not None
    await demo.drag([(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2), (box["x"] + box["width"] / 2, 650)], 0.5)
    await demo.click(demo.structure_row("Lateral Meniscus").locator(".v-list-item__content"), 0.6)
    await demo.stand_upright()
    await demo.move(WIDTH * 0.62, HEIGHT * 0.3)
    await demo.pause(1.5)


async def run(demo: ShortDemo) -> None:
    page = demo.page

    # Labels
    await demo.say("Knee MRI labels → simulation meshes", 0.3)
    await demo.turn(azimuth=-40, seconds=1.2)

    # Geometry
    await demo.press("mdi-cube-outline")
    await demo.skip(demo.wait_idle)
    await demo.say("Smooth geometry for all nine structures", 0.3)
    await demo.turn(azimuth=40, seconds=1.2)

    # Meshes and quality
    await demo.press("mdi-cube-scan")
    await demo.skip(demo.wait_idle)
    await demo.say("Hex & tet meshes, quality at a glance", 0.6)
    await demo.choose(demo.toolbar_select(1), "Scaled Jacobian")
    await demo.move(WIDTH * 0.62, HEIGHT * 0.35, 0.3)
    await demo.pause(1.2)

    # Clipping
    await demo.say("Clip to inspect the inner elements")
    await demo.press("mdi-box-cutter")
    await demo.pause(2.2)

    # Hex20 menisci
    await demo.say("Quadratic hex20 menisci")
    group = demo.structure_group("Lateral Meniscus")
    await demo.click(group.locator(".v-list-group__items .v-list-item", has_text="Mesh").first, 0.4)
    await demo.choose(page.locator(".v-select", has_text="Element type").first, "hex20")

    async def remesh() -> None:
        await demo.click(demo.mesh_button(), 0.3)
        await demo.wait_idle()
        await demo.open_child("Medial Meniscus", "Mesh")
        await demo.choose(page.locator(".v-select", has_text="Element type").first, "hex20")
        await demo.click(demo.mesh_button(), 0.3)
        await demo.wait_idle()
        await demo.click(demo.toolbar_button("mdi-box-cutter"), 0.3)
        for title in ("Femur", "Tibia", "Patella", "Patellar Cartilage", "Femoral Cartilage", "Lateral Tibial Cartilage", "Medial Tibial Cartilage"):
            await demo.toggle_eye(title)
        await demo.choose(demo.toolbar_select(1), "Structure colors")
        await demo.click(demo.toolbar_button("mdi-crop-free"), 0.5)
        await demo.move(WIDTH * 0.62, HEIGHT * 0.3, 0.3)

    await demo.skip(remesh)
    await demo.swing(50, 1.3)

    # End card, without the cursor
    await demo.say(None)
    await page.evaluate("""() => {
      for (const id of ['demo-cursor', 'demo-halo']) document.getElementById(id).style.display = 'none';
      document.getElementById('demo-end').classList.add('show');
    }""")
    await demo.pause(1.5)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:9123/")
    parser.add_argument("--out", default="short.mp4")
    args = parser.parse_args()
    frames = Path(__file__).parent / "frames"
    shutil.rmtree(frames, ignore_errors=True)
    frames.mkdir()
    recorder = ShortRecorder(frames)
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path="/usr/bin/google-chrome")
        context = await browser.new_context(viewport={"width": WIDTH, "height": HEIGHT}, device_scale_factor=DEVICE_SCALE)
        await context.add_init_script(OVERLAY)
        page = await context.new_page()
        page.on("console", lambda m: print("console:", m.type, m.text) if m.type == "error" else None)
        await page.goto(args.url)
        await page.wait_for_selector(".v-toolbar")
        await page.wait_for_timeout(2500)
        await page.add_style_tag(content=STYLE)
        await page.evaluate(ELEMENTS)
        demo = ShortDemo(page, recorder)
        await page.mouse.move(demo.x, demo.y)
        try:
            await prepare(demo)
            await recorder.start(page)
            await run(demo)
        finally:
            await page.screenshot(path=str(Path(__file__).parent / "last.png"))
            if hasattr(recorder, "cdp"):
                await recorder.stop()
            await browser.close()
    print(f"{len(recorder.frames)} frames; skipped {[round(e - s, 1) for s, e, _ in recorder.skipped]} s")
    print(f"clock offset {min(recorder.lags):.3f} s, median lag {sorted(recorder.lags)[len(recorder.lags) // 2]:.3f} s")
    recorder.encode(Path(args.out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
