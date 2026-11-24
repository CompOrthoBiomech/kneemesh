"""Exercise kneemesh and its GUI, then print every shared object and module the process loaded, as JSON.

docker/prune.py runs this to learn what the image must keep, and the
Dockerfile runs it again in the final image as a smoke test. It builds and
meshes a synthetic meniscus and a synthetic tibial cartilage on its bone
through the console scripts, then builds the GUI and renders the meshes in
its scene, so that VTK's rendering, web, and IO libraries load.
"""

import importlib
import json
import math
import os
import pkgutil
import sys
import tempfile
import uuid
from pathlib import Path

import SimpleITK as sitk


def _grid(size: tuple[int, int, int], spacing: tuple[float, float, float], origin: tuple[float, float, float]) -> tuple[sitk.Image, ...]:
    source = sitk.PhysicalPointImageSource()
    source.SetOutputPixelType(sitk.sitkVectorFloat64)
    source.SetSize(size)
    source.SetSpacing(spacing)
    source.SetOrigin(origin)
    points = source.Execute()
    return tuple(sitk.VectorIndexSelectionCast(points, axis) for axis in range(3))


def meniscus() -> sitk.Image:
    """A triangular cross section swept 270 degrees about z, as the tests' meniscus_label."""
    x, y, z = _grid((101, 101, 17), (0.5, 0.5, 1.0), (-25.0, -25.0, -8.0))
    inner, outer, height = 12.0, 20.0, 6.0
    radius = sitk.Sqrt(x * x + y * y)
    above_base = z + height / 2
    section = (radius <= outer) & (above_base >= 0.0) & (above_base <= height * (radius - inner) / (outer - inner))
    return sitk.Cast(section & (sitk.Abs(sitk.Atan2(y, x)) <= math.radians(135.0)), sitk.sitkUInt8)


def tibia_and_cartilage() -> sitk.Image:
    """A spherical bone (1) of radius 40 mm with a cartilage cap (2) up to 3 mm thick over an ellipse of 15 by 10 mm."""
    x, y, z = _grid((101, 81, 61), (0.5, 0.5, 0.5), (-25.0, -20.0, -25.0))
    depth = sitk.Sqrt(x * x + y * y + (z + 40.0) * (z + 40.0)) - 40.0
    ellipse = (x / 15.0) * (x / 15.0) + (y / 10.0) * (y / 10.0)
    cartilage = (depth > 0.0) & (ellipse < 1.0) & (depth < 3.0 * (1.0 - ellipse))
    bone = depth <= 0.0
    return sitk.Cast(bone + 2 * cartilage, sitk.sitkUInt8)


def run_pipelines(directory: Path) -> list[Path]:
    from kneemesh.pipeline import brep_main, mesh_main

    sitk.WriteImage(meniscus(), str(directory / "meniscus.nrrd"))
    sitk.WriteImage(tibia_and_cartilage(), str(directory / "tibia.nii"))
    out = str(directory)
    hex_breps = ["--meniscus.hex-blocks", "true", "--cartilage.hex-blocks", "true", "--output", "_hex.brep"]
    assert (
        brep_main(["--label-map", str(directory / "meniscus.nrrd"), "--labels", '{"lateral_meniscus": 1}', "--output-directory", out, *hex_breps])
        == 0
    )
    tibia = ["--label-map", str(directory / "tibia.nii"), "--labels", '{"tibia": 1, "medial_tibial_cartilage": 2}', "--output-directory", out]
    assert brep_main([*tibia, *hex_breps]) == 0
    # Overclosure removal moves the tibia out of its cartilage, tessellating the B-rep and measuring distances.
    assert brep_main([*tibia, "--output", "_tet.brep", "--overclosure.enabled", "true"]) == 0
    breps = {"lateral_meniscus": f"{out}/lateral_meniscus_hex.brep", "medial_tibial_cartilage": f"{out}/medial_tibial_cartilage_hex.brep"}
    hexahedra = ["--breps", json.dumps(breps), "--output-directory", out, "--meniscus.transverse-size", "2", "--meniscus.circumferential-size", "4"]
    assert mesh_main([*hexahedra, "--element-type", "hex20", "--assembled-mesh", f"{out}/knee.vtkhdf"]) == 0
    tetrahedra = {"medial_tibial_cartilage": f"{out}/medial_tibial_cartilage_tet.brep", "tibia": f"{out}/tibia.stl"}
    assert mesh_main(["--breps", json.dumps(tetrahedra), "--output-directory", out, "--element-type", "tet10", "--output", "_tet.vtu"]) == 0
    return sorted(directory.glob("*.vtu"))


def run_gui(meshes: list[Path]) -> None:
    import asyncio

    from trame.app import get_server

    from kneemesh.config import Structure
    from kneemesh.disk import read_brep
    from kneemesh.gui.app import KneemeshApp
    from kneemesh.gui.view import read_mesh_preview
    from kneemesh.gui.worker import brep_surface

    # Local rendering's serializers, which VtkRemoteView does not import.
    importlib.import_module("trame_vtk.modules.vtk.serializers.serialize")
    app = KneemeshApp(get_server(f"trace-{uuid.uuid4().hex}", client_type="vue3"))
    scene = app.scene
    structures: list[Structure] = ["lateral_meniscus", "medial_tibial_cartilage", "tibia"]
    for structure, path in zip(structures, meshes, strict=False):
        scene.set_mesh(structure, read_mesh_preview(path))
    scene.set_geometry("medial_meniscus", brep_surface(read_brep(next(meshes[0].parent.glob("*_hex.brep")))))
    scene.set_clipping(True)
    scene.refresh([], "latest", [], metric="scaled_jacobian")
    scene.render_window.Render()
    # What VtkRemoteView sends: a still render encoded as an image.
    from vtkmodules.vtkWebCore import vtkWebApplication

    web = vtkWebApplication()
    web.StillRenderToString(scene.render_window, 0, 90)
    web.StillRenderToBuffer(scene.render_window, 0, 90)

    # The GUI builds in a spawned worker process: compose a segmentation from an imported label and build it there.
    from kneemesh.gui.jobs import JobRunner

    directory = meshes[0].parent
    entry = {
        "id": 1,
        "path": str(directory / "meniscus.nrrd"),
        "name": "meniscus",
        "file_value": 1,
        "structure": "lateral_meniscus",
        "value": 1,
        "include": True,
    }
    job = {
        "kind": "geometry",
        "config": {"output_directory": str(directory / "gui")},
        "entries": [entry],
        "structures": ["lateral_meniscus"],
        "interpolator": "label_gaussian",
        "padding": 4,
        "preview_directory": str(directory / "previews"),
    }
    result = asyncio.run(JobRunner().run(job, lambda *_: None))
    assert Path(result["lateral_meniscus"]["preview"]).is_file()


def import_everything() -> None:
    import kneemesh

    for module in pkgutil.walk_packages(kneemesh.__path__, "kneemesh."):
        importlib.import_module(module.name)


def loaded_objects() -> list[str]:
    paths = {line.split(maxsplit=5)[5].strip() for line in Path("/proc/self/maps").read_text().splitlines() if len(line.split(maxsplit=5)) == 6}
    return sorted(path for path in paths if ".so" in Path(path).name)


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        os.chdir(directory)  # the console scripts write their logs and settings to the working directory
        meshes = run_pipelines(Path(directory))
        run_gui(meshes)
    import_everything()
    json.dump({"objects": loaded_objects(), "modules": sorted(sys.modules)}, sys.stdout)


if __name__ == "__main__":
    main()
