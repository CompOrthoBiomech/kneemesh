"""Pipeline jobs the GUI runs in a separate process, and the surfaces that preview their results.

gmsh holds global state, needs the main thread, and can crash the process,
so the GUI runs every job that touches it in a process of its own through
run_job. A job reports through a queue: ``("log", level, message)`` for
each log record, then ``("done", result)`` or ``("error", message)``.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import gmsh
import numpy as np
from loguru import logger
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData

from kneemesh.brep import BRep, gmsh_session
from kneemesh.config import BrepMesh, LabelMapBrep, Structure
from kneemesh.disk import read_brep, write_mesh, write_poly_data, write_segmentation
from kneemesh.gui.labels import LabelEntry, compose_segmentation
from kneemesh.pipeline import build_structure, run_mesh
from kneemesh.tet_meshing import Mesh, gmsh_options, import_solid

# Triangles across the shortest side of a B-rep's bounding box in its preview, and per circle of curvature.
_PREVIEW_DIVISIONS = 40
_PREVIEW_ELEMENTS_PER_CIRCLE = 24


class JobQueue(Protocol):
    """The part of a multiprocessing queue a job puts its reports on."""

    def put(self, obj: object, /) -> None: ...


def run_job(job: dict[str, Any], queue: JobQueue) -> None:
    """Run a GUI job and report its log and result on queue.

    A job is a dict whose ``kind`` is ``geometry``, ``mesh``, or ``preview``:

    - geometry: ``entries`` (LabelEntry dicts) composed into a segmentation
      with ``interpolator`` and ``padding``, written to
      ``assembled_label_map`` if that is given, then each of ``structures``
      built with the LabelMapBrep ``config`` and previewed.
    - mesh: the BrepMesh ``config``, whose breps are meshed, each also
      written as a .vtu preview.
    - preview: each B-rep in ``breps`` ({structure: path}) previewed.

    Previews go to ``preview_directory``. The result maps each structure to
    the paths of its output and its preview.
    """
    logger.remove()
    logger.add(lambda message: queue.put(("log", message.record["level"].name, message.record["message"])), level="INFO", format="{message}")
    logger.enable("kneemesh")
    try:
        preview_directory = Path(job["preview_directory"])
        preview_directory.mkdir(parents=True, exist_ok=True)
        match job["kind"]:
            case "geometry":
                result = _geometry_job(job, preview_directory)
            case "mesh":
                result = _mesh_job(job, preview_directory)
            case "preview":
                result = {
                    structure: {"preview": str(_preview_brep(Path(path), structure, preview_directory))} for structure, path in job["breps"].items()
                }
            case kind:
                raise ValueError(f"Unknown job kind {kind!r}")
    except Exception as error:  # noqa: BLE001 - any failure is reported to the GUI rather than lost with the process.
        logger.opt(exception=error).debug("Job failed")
        queue.put(("error", f"{type(error).__name__}: {error}"))
        return
    queue.put(("done", result))


def _geometry_job(job: dict[str, Any], preview_directory: Path) -> dict[str, dict[str, str]]:
    config = LabelMapBrep.model_validate(job["config"])
    entries: list[LabelEntry] = job["entries"]
    structures: Sequence[Structure] = job["structures"]
    logger.info("Composing the segmentation of {} labels", sum(entry["include"] for entry in entries))
    segmentation = compose_segmentation(entries, job["interpolator"], job["padding"])
    if assembled := job.get("assembled_label_map"):
        write_segmentation(segmentation, assembled)
        logger.info("Wrote the segmentation to {}", assembled)
    result: dict[str, dict[str, str]] = {}
    for structure in structures:
        logger.info("Building the geometry of {}", structure)
        geometry = build_structure(segmentation, structure, config)
        path = config.output_path(structure)
        result[structure] = {"output": str(path), "preview": str(path)}
        if isinstance(geometry, BRep):
            result[structure]["preview"] = str(_write_brep_preview(geometry, structure, preview_directory))
    return result


def _mesh_job(job: dict[str, Any], preview_directory: Path) -> dict[str, dict[str, str]]:
    config = BrepMesh.model_validate(job["config"])
    meshes = run_mesh(config)
    result: dict[str, dict[str, str]] = {}
    for structure, mesh in meshes.items():
        preview = preview_directory / f"{structure}_mesh.vtu"
        # The elements alone, without the boundary faces that would draw over them.
        write_mesh(Mesh(mesh.nodes, mesh.elements, mesh.element_types), preview)
        result[structure] = {"output": str(config.output_path(structure)), "preview": str(preview)}
    return result


def _preview_brep(path: Path, structure: str, preview_directory: Path) -> Path:
    logger.info("Previewing {}", path)
    return _write_brep_preview(read_brep(path), structure, preview_directory)


def _write_brep_preview(brep: BRep, structure: str, preview_directory: Path) -> Path:
    path = preview_directory / f"{structure}_geometry.vtp"
    write_poly_data(brep_surface(brep), path)
    return path


def brep_surface(brep: BRep) -> vtkPolyData:
    """A B-rep's faces as triangles and its edges as lines, for display, in one vtkPolyData.

    The triangles are a coarse gmsh surface mesh, sized from the B-rep's
    bounding box and its curvature. A cell array "edge" is 1 on the lines.

    Raises:
        ValueError: If the B-rep has no faces.
    """
    with gmsh_session():
        import_solid_or_shell(brep)
        x_min, y_min, z_min, x_max, y_max, z_max = gmsh.model.getBoundingBox(-1, -1)
        size = max(min(x_max - x_min, y_max - y_min, z_max - z_min), 1e-6) / _PREVIEW_DIVISIONS * 4
        options = {
            "Mesh.MeshSizeMax": size,
            "Mesh.MeshSizeMin": size / 10,
            "Mesh.MeshSizeFromCurvature": _PREVIEW_ELEMENTS_PER_CIRCLE,
            "Mesh.Algorithm": 6,
        }
        with gmsh_options(options):
            gmsh.model.mesh.generate(2)
        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        index = np.zeros(int(np.max(node_tags)) + 1, dtype=np.int64)
        index[np.asarray(node_tags, dtype=np.int64)] = np.arange(len(node_tags))
        triangles = _elements(2, index)
        lines = _elements(1, index)
    if not len(triangles):
        raise ValueError("The B-rep has no faces to preview")

    surface = vtkPolyData()
    points = vtkPoints()
    points.SetData(numpy_to_vtk(np.asarray(coordinates, dtype=np.float64).reshape(-1, 3), deep=True))
    surface.SetPoints(points)
    surface.SetPolys(_cell_array(triangles))
    surface.SetLines(_cell_array(lines))
    # VTK orders cells verts, lines, polys, strips.
    edge = np.concatenate([np.ones(len(lines), dtype=np.uint8), np.zeros(len(triangles), dtype=np.uint8)])
    edge_array = numpy_to_vtk(edge, deep=True)
    edge_array.SetName("edge")
    surface.GetCellData().AddArray(edge_array)
    return surface


def import_solid_or_shell(brep: BRep) -> None:
    """Load a B-rep into the current gmsh model, whether it holds a solid or only surfaces."""
    try:
        import_solid(brep)
    except ValueError:
        if not gmsh.model.getEntities(2):
            raise


def _elements(element_type: int, index: np.ndarray) -> np.ndarray:
    """The current gmsh model's linear elements of a type, as rows of node indices; gmsh types 1 and 2 are lines and triangles."""
    nodes_per_element = element_type + 1
    _, element_nodes = gmsh.model.mesh.getElementsByType(element_type)
    return index[np.asarray(element_nodes, dtype=np.int64)].reshape(-1, nodes_per_element)


def _cell_array(connectivity: np.ndarray) -> vtkCellArray:
    cells = vtkCellArray()
    offsets = np.arange(0, connectivity.size + 1, max(connectivity.shape[1], 1), dtype=np.int64) if len(connectivity) else np.zeros(1, np.int64)
    cells.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True), numpy_to_vtkIdTypeArray(connectivity.ravel().astype(np.int64), deep=True))
    return cells
