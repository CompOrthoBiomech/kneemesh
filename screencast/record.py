"""Record a captioned screencast of kneemesh-gui with a highlighted cursor.

Run the GUI first from the repository root: uv run kneemesh-gui --server --port 9123 --view-ratio 3
Then: uv run --with playwright python record.py [--url http://localhost:9123/] [--out screencast.mp4]
"""

import argparse
import asyncio
import base64
import math
import re
import shutil
import subprocess
import time
from pathlib import Path

from playwright.async_api import Locator, Page, async_playwright

# The page is laid out at 1600 x 900 and rendered at 2.4 device pixels per CSS pixel: 3840 x 2160 (4K).
WIDTH, HEIGHT = 1600, 900
DEVICE_SCALE = 2.4
# Run the GUI with --view-ratio 3 so the 3D view renders past the device pixels; 2.4 itself shows no image.
TRACKBALL_DEGREES = 200.0
ANCHOR_AZIMUTH = 70.0
ANCHOR_ELEVATION = 15.0
WAIT_SPEEDUP = 6.0

OVERLAY = r"""
(() => {
  const install = () => {
    if (document.getElementById('demo-cursor')) return;
    const style = document.createElement('style');
    style.textContent = `
      #demo-halo { position: fixed; left: 0; top: 0; width: 46px; height: 46px; margin: -23px 0 0 -23px; border-radius: 50%;
        background: rgba(255, 214, 0, 0.40); border: 2px solid rgba(255, 170, 0, 0.95); pointer-events: none;
        z-index: 2147483646; transition: transform 0.12s, background 0.12s; }
      #demo-halo.down { transform: scale(0.65); background: rgba(255, 80, 40, 0.55); border-color: rgba(220, 40, 0, 0.95); }
      #demo-cursor { position: fixed; left: 0; top: 0; pointer-events: none; z-index: 2147483647; }
      .demo-ripple { position: fixed; width: 20px; height: 20px; margin: -10px 0 0 -10px; border-radius: 50%;
        border: 3px solid rgba(255, 120, 0, 0.9); pointer-events: none; z-index: 2147483645; animation: demo-ripple 0.6s ease-out forwards; }
      @keyframes demo-ripple { to { transform: scale(4); opacity: 0; } }
      #demo-caption { position: fixed; left: calc(440px + (100vw - 440px) / 2); bottom: 222px; transform: translateX(-50%);
        max-width: 1040px; width: max-content; background: rgba(15, 18, 28, 0.86); color: #fff; font: 500 20px/1.4 Roboto, sans-serif;
        padding: 12px 22px; border-radius: 10px; border-left: 5px solid #ffc400; pointer-events: none; z-index: 2147483640;
        box-shadow: 0 4px 18px rgba(0,0,0,0.4); transition: opacity 0.25s; opacity: 0; }
      #demo-caption.show { opacity: 1; }
      #demo-caption .step { display: block; font-size: 13px; letter-spacing: 0.08em; text-transform: uppercase; color: #ffc400; }
    `;
    document.head.appendChild(style);
    const halo = document.createElement('div');
    halo.id = 'demo-halo';
    const cursor = document.createElement('div');
    cursor.id = 'demo-cursor';
    cursor.innerHTML = '<svg width="26" height="30" viewBox="0 0 20 24"><path d="M2 1 L2 19 L6.6 14.6 L9.6 21.6 L12.6 20.3 L9.7 13.5 L16 13.5 Z" fill="#fff" stroke="#000" stroke-width="1.4" stroke-linejoin="round"/></svg>';
    const caption = document.createElement('div');
    caption.id = 'demo-caption';
    document.body.append(halo, cursor, caption);
    const place = (x, y) => {
      halo.style.left = x + 'px'; halo.style.top = y + 'px';
      cursor.style.transform = `translate(${x - 2}px, ${y - 1}px)`;
    };
    place(-100, -100);
    window.addEventListener('mousemove', (e) => place(e.clientX, e.clientY), true);
    window.addEventListener('mousedown', (e) => {
      halo.classList.add('down');
      const ripple = document.createElement('div');
      ripple.className = 'demo-ripple';
      ripple.style.left = e.clientX + 'px'; ripple.style.top = e.clientY + 'px';
      document.body.appendChild(ripple);
      setTimeout(() => ripple.remove(), 700);
    }, true);
    window.addEventListener('mouseup', () => halo.classList.remove('down'), true);
    window.demoCaption = (step, text) => {
      if (!text) { caption.classList.remove('show'); return; }
      caption.innerHTML = (step ? `<span class="step">${step}</span>` : '') + text;
      caption.classList.add('show');
    };
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', install); else install();
})();
"""


class Recorder:
    """Collects screencast frames from Chrome with their wall-clock times, and periods to speed up."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.frames: list[tuple[float, Path]] = []
        self.fast: list[tuple[float, float]] = []

    async def start(self, page: Page) -> None:
        self.cdp = await page.context.new_cdp_session(page)
        self.cdp.on("Page.screencastFrame", self._frame)
        await self.cdp.send(
            "Page.startScreencast",
            {"format": "jpeg", "quality": 92, "maxWidth": round(WIDTH * DEVICE_SCALE), "maxHeight": round(HEIGHT * DEVICE_SCALE)},
        )

    def _frame(self, params: dict) -> None:
        path = self.directory / f"frame_{len(self.frames):06d}.jpg"
        path.write_bytes(base64.b64decode(params["data"]))
        self.frames.append((params["metadata"]["timestamp"], path))
        asyncio.create_task(self.cdp.send("Page.screencastFrameAck", {"sessionId": params["sessionId"]}))

    async def stop(self) -> None:
        await self.cdp.send("Page.stopScreencast")

    def _video_time(self, t: float) -> float:
        """Wall time t mapped onto the video's timeline, with fast periods compressed."""
        out = t
        for start, end in self.fast:
            if t > start:
                out -= (min(t, end) - start) * (1 - 1 / WAIT_SPEEDUP)
        return out

    def encode(self, output: Path) -> None:
        frames = sorted(self.frames)
        times = [self._video_time(t) for t, _ in frames]
        listing = self.directory / "frames.txt"
        with listing.open("w") as f:
            for i, (_, path) in enumerate(frames):
                duration = (times[i + 1] - times[i]) if i + 1 < len(frames) else 2.0
                f.write(f"file '{path}'\nduration {max(duration, 0.001):.4f}\n")
            f.write(f"file '{frames[-1][1]}'\n")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(listing),
                "-vf",
                f"fps=30,scale={round(WIDTH * DEVICE_SCALE)}:{round(HEIGHT * DEVICE_SCALE)}:flags=lanczos,format=yuv420p",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-preset",
                "slow",
                "-movflags",
                "+faststart",
                str(output),
            ],
            check=True,
        )


class Demo:
    def __init__(self, page: Page, recorder: Recorder) -> None:
        self.page = page
        self.recorder = recorder
        self.x, self.y = WIDTH / 2, HEIGHT / 2

    async def pause(self, seconds: float) -> None:
        await self.page.wait_for_timeout(seconds * 1000)

    async def caption(self, text: str | None, step: str = "", hold: float = 0.0) -> None:
        await self.page.evaluate("([s, t]) => window.demoCaption(s, t)", [step, text or ""])
        if hold:
            await self.pause(hold)

    async def move(self, x: float, y: float, seconds: float | None = None) -> None:
        distance = math.hypot(x - self.x, y - self.y)
        seconds = seconds if seconds is not None else min(1.0, 0.25 + distance / 1600)
        steps = max(2, int(seconds * 60))
        x0, y0 = self.x, self.y
        for i in range(1, steps + 1):
            s = i / steps
            e = s * s * (3 - 2 * s)
            await self.page.mouse.move(x0 + (x - x0) * e, y0 + (y - y0) * e)
            await self.page.wait_for_timeout(1000 / 60)
        self.x, self.y = x, y

    async def point(self, locator: Locator, seconds: float | None = None) -> None:
        await locator.scroll_into_view_if_needed()
        box = await locator.bounding_box()
        assert box is not None, f"{locator} has no box"
        await self.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, seconds)

    async def click(self, locator: Locator, after: float = 0.6) -> None:
        await self.point(locator)
        await self.pause(0.15)
        await self.page.mouse.down()
        await self.pause(0.12)
        await self.page.mouse.up()
        await self.pause(after)

    async def drag(self, path: list[tuple[float, float]], seconds: float = 2.0) -> None:
        await self.move(*path[0])
        await self.page.mouse.down()
        legs = len(path) - 1
        for x, y in path[1:]:
            await self.move(x, y, seconds / legs)
        # The remote view throttles mouse moves, so hold at the end and repeat it, or the drag's last stretch is lost.
        for dx in (0.5, 0.0):
            await self.pause(0.2)
            await self.page.mouse.move(self.x + dx, self.y)
        await self.pause(0.2)
        await self.page.mouse.up()
        await self.pause(0.4)

    async def wheel(self, dy: float, ticks: int = 6) -> None:
        for _ in range(ticks):
            await self.page.mouse.wheel(0, dy / ticks)
            await self.pause(0.08)
        await self.pause(0.5)

    # The 3D view

    async def view_box(self) -> dict:
        box = await self.page.locator(".v-main").bounding_box()
        assert box is not None
        log = self.page.locator(".kneemesh-log").first
        bottom = box["y"] + box["height"]
        if await log.is_visible():
            bottom -= 200
        return {"x": box["x"], "y": box["y"], "w": box["width"], "h": bottom - box["y"]}

    async def turn(self, azimuth: float = 0.0, elevation: float = 0.0, seconds: float = 2.2) -> None:
        """Drag across the view to turn the camera by degrees: azimuth about the view's up direction, positive moving the camera
        to the left of the scene, and elevation, positive moving it up.

        VTK's trackball camera turns 200 degrees across the full width or height of the view.
        """
        v = await self.view_box()
        dx = -azimuth / TRACKBALL_DEGREES * v["w"]
        dy = elevation / TRACKBALL_DEGREES * v["h"]
        cx, cy = v["x"] + v["w"] / 2, v["y"] + v["h"] * 0.45
        await self.drag([(cx - dx / 2, cy - dy / 2), (cx + dx / 2, cy + dy / 2)], seconds)

    async def stand_upright(self) -> None:
        """Turn the camera from VTK's default, looking down the femoral shaft (+z, superior, in LPS) with posterior up, to the
        anchor view: superior up, ANCHOR_AZIMUTH degrees from anterior toward lateral, and looking down by ANCHOR_ELEVATION.

        Later turns about the view's up direction then turn the knee about its shaft.
        """
        await self.turn(elevation=-90, seconds=2.5)
        await self.turn(azimuth=ANCHOR_AZIMUTH, seconds=2.5)
        await self.turn(elevation=ANCHOR_ELEVATION, seconds=1.0)

    async def swing(self, degrees: float = 50, seconds: float = 2.0) -> None:
        """Turn about the shaft toward anterior and back to the anchor view."""
        await self.turn(azimuth=-degrees, seconds=seconds)
        await self.turn(azimuth=degrees, seconds=seconds)

    async def orbit(self) -> None:
        """Turn about the shaft through anterior to the far side and back to the anchor view."""
        # Drags of 70 degrees come back to where they started; longer ones drifted by tens of degrees.
        for azimuth in (-ANCHOR_AZIMUTH, -ANCHOR_AZIMUTH, ANCHOR_AZIMUTH, ANCHOR_AZIMUTH):
            await self.turn(azimuth=azimuth, seconds=1.8)

    # Widgets

    def toolbar_button(self, icon: str) -> Locator:
        return self.page.locator(f".v-toolbar button:has(i.{icon})").first

    def structure_row(self, title: str) -> Locator:
        exact = self.page.locator(".v-list-item-title", has_text=re.compile(f"^{re.escape(title)}$"))
        return self.page.locator(".v-list-item:has(button i.mdi-target, button i.mdi-target-account)").filter(has=exact).first

    def structure_group(self, title: str) -> Locator:
        return (
            self.page.locator(".v-list-group")
            .filter(
                has=self.page.locator(":scope > .v-list-item:has(button i.mdi-target, button i.mdi-target-account)").filter(
                    has=self.page.locator(".v-list-item-title", has_text=re.compile(f"^{re.escape(title)}$"))
                )
            )
            .first
        )

    async def toggle_eye(self, title: str) -> None:
        await self.click(self.structure_row(title).locator("button").nth(1), 0.5)

    async def toggle_solo(self, title: str) -> None:
        await self.click(self.structure_row(title).locator("button").nth(0), 0.8)

    async def open_child(self, structure: str, child: str) -> None:
        group = self.structure_group(structure)
        item = group.locator(".v-list-group__items .v-list-item", has_text=child).first
        if not await item.is_visible():
            await self.click(self.structure_row(structure).locator(".v-list-item__content"), 0.6)
        await self.click(item, 0.8)

    async def choose(self, select: Locator, option: str) -> None:
        await self.click(select, 0.6)
        await self.click(self.page.locator(".v-overlay--active .v-list-item", has_text=option).first, 0.8)

    def toolbar_select(self, index: int) -> Locator:
        return self.page.locator(".v-toolbar .v-select").nth(index)

    async def dialog_item(self, name: str) -> None:
        item = self.page.locator(".v-dialog .v-list-item").filter(has=self.page.get_by_text(name, exact=True)).first
        await self.click(item, 0.7)

    async def dialog_open(self) -> None:
        await self.click(self.page.locator(".v-dialog .v-card-actions button", has_text="Open"), 0.6)

    async def wait_idle(self, label: str | None = None, fast: bool = True, minimum: float = 1.0) -> None:
        """Wait for the running job or label preview to finish, sped up in the video."""
        start = time.time()
        if label:
            await self.caption(label + f'<br><span style="font-size:15px;color:#bbb">(waiting, sped up {WAIT_SPEEDUP:g}×)</span>', "Working")
        await self.pause(minimum)
        await self.page.wait_for_function(
            """() => ![...document.querySelectorAll('.v-toolbar .v-progress-linear')].some((e) => !e.closest('.v-field'))
                 &&!document.querySelector('.v-main .v-card .text-caption.mr-2')?.textContent.trim()""",
            timeout=600_000,
            polling=250,
        )
        await self.pause(1.5)
        if fast:
            self.recorder.fast.append((start, time.time()))
        if label:
            await self.caption(None)


async def run(demo: Demo) -> None:
    page = demo.page

    # Introduction
    await demo.caption("kneemesh turns segmented knee images into B-rep geometry and finite element meshes.", "kneemesh GUI", 4)
    await demo.caption(
        "The tree on the left holds Settings, imported Labels, and each structure's Geometry and Mesh; the 3D view and the log are on the right.",
        "Layout",
        4.5,
    )

    # Part 1: one label
    await demo.caption("First, add a single label: the real lateral meniscus from <code>tests/dat</code>.", "1 · Add a label")
    await demo.click(page.locator(".v-list-item", has_text="Add Label").first, 1.0)
    await demo.caption("The file dialog browses the machine running the pipeline (files can also be uploaded).", "1 · Add a label")
    await demo.dialog_item("tests")
    await demo.dialog_item("dat")
    await demo.dialog_item("lateral_meniscus.nii")
    await demo.dialog_open()
    await demo.wait_idle(fast=False)
    await demo.caption(
        "The label is rendered as the B-rep pipeline first sees it: a signed distance field of the voxels, Gaussian-smoothed, "
        "contoured with flying edges, then windowed-sinc smoothed — no voxel staircase.",
        "1 · Label rendering",
        6.5,
    )
    await demo.caption("Drag in the view to rotate the camera; the scroll wheel zooms.", "1 · Label rendering")
    await demo.stand_upright()
    await demo.wheel(-300)
    await demo.wheel(300)
    await demo.caption(
        "Expanding the label shows its structure name, the value it takes in the segmentation, and whether it is included.",
        "1 · Label settings",
    )
    label_group = page.locator(".v-list-group").filter(has=page.locator(":scope > .v-list-item", has_text="lateral_meniscus.nii")).first
    await demo.click(label_group.locator(":scope > .v-list-item").first, 3.0)
    await demo.caption("Remove it with the delete button; its surface leaves the view.", "1 · Remove the label")
    await demo.click(label_group.locator("button:has(i.mdi-delete-outline)"), 2.0)

    # Part 2a: assemble
    await demo.caption(
        "Now walk through the pipeline with the knee examples. Import <code>examples/knee_assemble.json</code>, the "
        "<code>kneemesh-assemble</code> configuration naming all nine structures' label files.",
        "2 · Import: assemble",
    )
    await demo.click(demo.toolbar_button("mdi-file-import-outline"), 1.0)
    await demo.click(page.locator(".v-dialog button:has(i.mdi-arrow-up)"), 0.6)
    await demo.click(page.locator(".v-dialog button:has(i.mdi-arrow-up)"), 0.6)
    await demo.dialog_item("examples")
    await demo.dialog_item("knee_assemble.json")
    await demo.dialog_open()
    await demo.wait_idle("Extracting the nine label surfaces")
    await demo.caption("Collapse Labels and drag the splitter down to see every structure.", "2 · Labels")
    await demo.click(page.locator(".v-list-item", has_text="Labels").first, 0.6)
    splitter = page.locator(".kneemesh-splitter")
    box = await splitter.bounding_box()
    assert box is not None
    sx, sy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    await demo.drag([(sx, sy), (sx, 650)], 1.0)
    await demo.caption("Nine labels: femur, tibia, and patella, both menisci, and four cartilages. Rotate to look around the knee.", "2 · Labels")
    await demo.orbit()
    await demo.caption("Each structure's eye button hides or shows it. Hide the femur and patella to see the joint space…", "2 · Hide and show")
    await demo.toggle_eye("Femur")
    await demo.toggle_eye("Patella")
    await demo.swing()
    await demo.caption("…and show them again.", "2 · Hide and show")
    await demo.toggle_eye("Femur")
    await demo.toggle_eye("Patella")
    await demo.pause(1.0)

    # Part 2b: geometry
    await demo.caption(
        "Import <code>examples/knee_brep_hex.json</code>, the <code>kneemesh-brep</code> configuration: labels from the assembled "
        "<code>knee.nrrd</code> and geometry settings for hexahedral meshing.",
        "3 · Import: geometry",
    )
    await demo.click(demo.toolbar_button("mdi-file-import-outline"), 1.0)
    await demo.dialog_item("knee_brep_hex.json")
    await demo.dialog_open()
    await demo.wait_idle("Extracting surfaces of the assembled segmentation's nine values")
    await demo.caption(
        "Each structure's Geometry page edits its settings: the menisci build butterfly blocks for hexahedra.", "3 · Geometry settings"
    )
    await demo.open_child("Lateral Meniscus", "Geometry")
    await demo.pause(3.0)
    await demo.caption(
        "The B-reps this configuration wrote already exist, so show them instead of rebuilding (the cube button would build them).",
        "3 · Geometry",
    )
    await demo.click(demo.toolbar_button("mdi-folder-open-outline"), 0.5)
    await demo.wait_idle("Triangulating the B-reps with gmsh for display")
    await demo.caption("B-rep surfaces with their curves; the bones are decimated, triangulated surfaces.", "3 · Geometry")
    await demo.orbit()
    await demo.caption("Hide the femur and tibia to see the menisci and cartilage sheets.", "3 · Hide and show")
    await demo.toggle_eye("Femur")
    await demo.toggle_eye("Tibia")
    await demo.toggle_eye("Patella")
    await demo.swing()
    await demo.caption("The display menu compares stages: switch back to the labels, then to the geometry.", "3 · Stages")
    await demo.choose(demo.toolbar_select(0), "Labels")
    await demo.pause(1.5)
    await demo.choose(demo.toolbar_select(0), "Geometry")
    await demo.pause(1.0)
    await demo.caption("Show the bones again.", "3 · Hide and show")
    await demo.toggle_eye("Femur")
    await demo.toggle_eye("Tibia")
    await demo.toggle_eye("Patella")
    await demo.choose(demo.toolbar_select(0), "Latest stage")

    # Part 2c: mesh
    await demo.caption(
        "Import <code>examples/knee_mesh_hex8.json</code>, the <code>kneemesh-mesh</code> configuration: linear hexahedra for the "
        "soft tissue and graded tetrahedra for the bones.",
        "4 · Import: mesh",
    )
    await demo.click(demo.toolbar_button("mdi-file-import-outline"), 1.0)
    await demo.dialog_item("knee_mesh_hex8.json")
    await demo.dialog_open()
    await demo.pause(1.5)
    await demo.caption("Mesh every structure with the toolbar's mesh button.", "4 · Mesh")
    await demo.click(demo.toolbar_button("mdi-cube-scan"), 0.5)
    await demo.wait_idle("Meshing all nine structures and optimizing the hexahedra")
    await demo.caption("The meshes, elements outlined. Rotate around the knee.", "4 · Mesh")
    await demo.orbit()
    await demo.caption("Hide the bones to look at the soft-tissue hexahedra, then show them again.", "4 · Hide and show")
    for bone in ("Femur", "Tibia", "Patella"):
        await demo.toggle_eye(bone)
    await demo.swing()
    for bone in ("Femur", "Tibia", "Patella"):
        await demo.toggle_eye(bone)
    await demo.pause(1.0)

    # Part 3: quality and clipping
    await demo.caption("Color the meshes by scaled Jacobian: blue is well shaped, red poorly.", "5 · Quality")
    await demo.choose(demo.toolbar_select(1), "Scaled Jacobian")
    await demo.swing(40)
    await demo.caption("The box cutter clips by a plane, keeping whole cells, so the interior elements show.", "5 · Clipping")
    await demo.click(demo.toolbar_button("mdi-box-cutter"), 1.5)
    await demo.caption("The slider moves the plane toward or away from the viewer.", "5 · Clipping")
    thumb = page.locator(".v-toolbar .v-slider-thumb").first
    box = await thumb.bounding_box()
    assert box is not None
    tx, ty = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    await demo.drag([(tx, ty), (tx - 45, ty), (tx + 40, ty), (tx, ty)], 4.0)
    await demo.caption("Unpinned, the plane faces the viewer as the view turns, always cutting across the line of sight.", "5 · Clipping")
    await demo.swing(60, 2.5)
    await demo.caption("Pinned, the plane stays put in the model and turns with it.", "5 · Clipping")
    await demo.click(demo.toolbar_button("mdi-pin-off-outline"), 0.6)
    await demo.swing(60, 2.5)
    await demo.click(demo.toolbar_button("mdi-pin"), 0.6)
    await demo.caption("Now the lateral meniscus alone: its target button shows only that structure.", "5 · Lateral meniscus")
    await demo.toggle_solo("Lateral Meniscus")
    await demo.pause(1.0)
    await demo.drag([(tx, ty), (tx - 40, ty), (tx + 35, ty), (tx, ty)], 4.0)
    await demo.swing(60, 2.5)
    await demo.caption("Clipping off: the whole meniscus by scaled Jacobian.", "5 · Lateral meniscus")
    await demo.click(demo.toolbar_button("mdi-box-cutter"), 1.0)
    await demo.swing(40)
    await demo.toggle_solo("Lateral Meniscus")

    # Part 4: hex20 menisci
    await demo.caption("Switch both menisci to quadratic hexahedra: each structure's Mesh page has its own element type.", "6 · Hex20 menisci")
    await demo.open_child("Lateral Meniscus", "Mesh")
    await demo.choose(page.locator(".v-select", has_text="Element type").first, "hex20")
    await demo.caption("Mesh the lateral meniscus again with its Mesh button.", "6 · Hex20 menisci")
    await demo.click(page.locator("button", has_text="Mesh").filter(has=page.locator("i.mdi-play")).first, 0.5)
    await demo.wait_idle("Meshing the lateral meniscus with 20-node hexahedra")
    await demo.caption("The same for the medial meniscus.", "6 · Hex20 menisci")
    await demo.open_child("Medial Meniscus", "Mesh")
    await demo.choose(page.locator(".v-select", has_text="Element type").first, "hex20")
    await demo.click(page.locator("button", has_text="Mesh").filter(has=page.locator("i.mdi-play")).first, 0.5)
    await demo.wait_idle("Meshing the medial meniscus with 20-node hexahedra")
    await demo.caption("Hide everything but the menisci, color by structure, and fit the camera.", "6 · Hex20 menisci")
    for title in ("Femur", "Tibia", "Patella", "Patellar Cartilage", "Femoral Cartilage", "Lateral Tibial Cartilage", "Medial Tibial Cartilage"):
        await demo.toggle_eye(title)
    await demo.choose(demo.toolbar_select(1), "Structure colors")
    await demo.click(demo.toolbar_button("mdi-crop-free"), 1.0)
    await demo.caption("The quadratic menisci alone.", "6 · Hex20 menisci")
    await demo.orbit()
    await demo.caption("That's the full path: labels → geometry → meshes.", "kneemesh GUI", 4)
    await demo.caption(None)
    await demo.pause(1.0)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:9123/")
    parser.add_argument("--out", default="screencast.mp4")
    args = parser.parse_args()
    frames = Path(__file__).parent / "frames"
    shutil.rmtree(frames, ignore_errors=True)
    frames.mkdir()
    recorder = Recorder(frames)
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path="/usr/bin/google-chrome")
        context = await browser.new_context(viewport={"width": WIDTH, "height": HEIGHT}, device_scale_factor=DEVICE_SCALE)
        await context.add_init_script(OVERLAY)
        page = await context.new_page()
        page.on("console", lambda m: print("console:", m.type, m.text) if m.type == "error" else None)
        await page.goto(args.url)
        await page.wait_for_selector(".v-toolbar")
        await page.wait_for_timeout(2500)
        await recorder.start(page)
        demo = Demo(page, recorder)
        await page.mouse.move(demo.x, demo.y)
        try:
            await run(demo)
        finally:
            await page.screenshot(path=str(Path(__file__).parent / "last.png"))
            await recorder.stop()
            await browser.close()
    print(f"{len(recorder.frames)} frames; fast periods {[(round(e - s, 1)) for s, e in recorder.fast]}")
    recorder.encode(Path(args.out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
