"""Pipeline jobs the GUI runs in a separate process, and the surfaces that preview their results.

gmsh holds global state, needs the main thread, and can crash the process,
so the GUI runs every job that touches it in a process of its own through
run_job. A job reports through a queue: ``("log", level, message)`` for
each log record, then ``("done", result)`` or ``("error", message)``.
"""

import pickle
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
from kneemesh.config import BONE_STRUCTURES, BrepMesh, LabelMapBrep, MeniscusBrep, Structure
from kneemesh.disk import read_brep, read_surface, write_brep, write_mesh, write_poly_data, write_segmentation, write_surface
from kneemesh.gui.labels import LabelEntry, compose_segmentation
from kneemesh.overclosure import Adjustable, resolve_overclosure
from kneemesh.pipeline import build_adjustable, meniscus_loft, run_mesh
from kneemesh.tet_meshing import gmsh_options, import_solid

# Triangles across the shortest side of a B-rep's bounding box in its preview, and per circle of curvature.
_PREVIEW_DIVISIONS = 40
_PREVIEW_ELEMENTS_PER_CIRCLE = 24


class JobQueue(Protocol):
    """The part of a multiprocessing queue a job puts its reports on."""

    def put(self, obj: object, /) -> None: ...


def run_job(job: dict[str, Any], queue: JobQueue) -> None:
    """Run a GUI job and report its log and result on queue.

    A job is a dict whose ``kind`` is ``geometry``, ``mesh``, ``preview``, or ``overclosure``:

    - geometry: ``entries`` (LabelEntry dicts) composed into a segmentation
      with ``interpolator`` and ``padding``, written to
      ``assembled_label_map`` if that is given, then each of ``structures``
      built with the LabelMapBrep ``config`` and previewed. What each was
      built from is saved beside its preview, for overclosure jobs.
    - overclosure: the geometry files ``geometry`` ({structure: path})
      moved apart where they overlap (overclosure.resolve_overclosure),
      with the ``overclosure`` section of the LabelMapBrep ``config``. A
      cartilage or meniscus moves only if this session built the file, so
      what it was built from is saved; the rest stay in place. Moved files
      are rewritten in place and previewed; the result names only those.
    - mesh: the BrepMesh ``config``, whose breps are meshed, each also
      written as a .vtu preview with its faces and quality metrics.
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
            case "overclosure":
                result = _overclosure_job(job, preview_directory)
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
        built = build_adjustable(segmentation, structure, config)
        path = config.output_path(structure)
        meniscus = config.section(structure, MeniscusBrep) if built.slices is not None else None
        _save_source(built, meniscus, path, structure, preview_directory)
        result[structure] = _written(built.geometry, path, structure, preview_directory)
    return result


def _overclosure_job(job: dict[str, Any], preview_directory: Path) -> dict[str, dict[str, str]]:
    config = LabelMapBrep.model_validate(job["config"])
    paths: dict[str, Path] = {structure: Path(path) for structure, path in job["geometry"].items()}
    structures = {structure: _load_adjustable(structure, path, preview_directory) for structure, path in paths.items()}
    logger.info("Moving overlapping structures apart with clearance {}", config.overclosure.clearance)
    resolved = resolve_overclosure(structures, config.overclosure.priority, config.overclosure.clearance)
    result: dict[str, dict[str, str]] = {}
    for structure, adjusted in resolved.items():
        if adjusted.geometry is structures[structure].geometry:
            continue
        path = paths[structure]
        if isinstance(adjusted.geometry, BRep):
            write_brep(adjusted.geometry, path)
        else:
            write_surface(adjusted.geometry, path)
        logger.info("Wrote {}", path)
        source = _read_source(structure, path, preview_directory)
        meniscus = MeniscusBrep.model_validate(source["meniscus"]) if source is not None and source["meniscus"] is not None else None
        _save_source(adjusted, meniscus, path, structure, preview_directory)
        result[structure] = _written(adjusted.geometry, path, structure, preview_directory)
    if not result:
        logger.info("No structures overlap")
    return result


def _written(geometry: BRep | vtkPolyData, path: Path, structure: str, preview_directory: Path) -> dict[str, str]:
    """The paths of a structure's written geometry and its preview: a B-rep's surface, or the bone's surface itself."""
    preview = _write_brep_preview(geometry, structure, preview_directory) if isinstance(geometry, BRep) else path
    return {"output": str(path), "preview": str(preview)}


def _source_path(structure: str, preview_directory: Path) -> Path:
    return preview_directory / f"{structure}_source.pkl"


def _save_source(built: Adjustable, meniscus: MeniscusBrep | None, path: Path, structure: str, preview_directory: Path) -> None:
    """Save what a cartilage or meniscus was built from beside its preview, with the file it was written to; bones need nothing."""
    if built.sheet is None and built.slices is None:
        return
    source = {
        "output": str(path.resolve()),
        "sheet": built.sheet,
        "slices": built.slices,
        "meniscus": meniscus.model_dump(mode="json") if meniscus is not None else None,
    }
    _source_path(structure, preview_directory).write_bytes(pickle.dumps(source))


def _read_source(structure: str, path: Path, preview_directory: Path) -> dict[str, Any] | None:
    """What this session built a structure's file from, or None if it did not build that file."""
    source_path = _source_path(structure, preview_directory)
    if not source_path.is_file():
        return None
    # The GUI's own temporary directory, written by its own worker processes.
    source: dict[str, Any] = pickle.loads(source_path.read_bytes())
    return source if source["output"] == str(path.resolve()) else None


def _load_adjustable(structure: str, path: Path, preview_directory: Path) -> Adjustable:
    """A structure's geometry file with what this session built it from, if it did."""
    if structure in BONE_STRUCTURES:
        return Adjustable(read_surface(path))
    brep = read_brep(path)
    source = _read_source(structure, path, preview_directory)
    if source is None:
        logger.warning("{} was not built in this session, so it stays in place: build it to let it move", structure)
        return Adjustable(brep)
    if source["sheet"] is not None:
        return Adjustable(brep, sheet=source["sheet"])
    return Adjustable(brep, slices=source["slices"], loft=meniscus_loft(MeniscusBrep.model_validate(source["meniscus"])))


def _mesh_job(job: dict[str, Any], preview_directory: Path) -> dict[str, dict[str, str]]:
    config = BrepMesh.model_validate(job["config"])
    meshes = run_mesh(config)
    result: dict[str, dict[str, str]] = {}
    for structure, mesh in meshes.items():
        preview = preview_directory / f"{structure}_mesh.vtu"
        # The whole mesh, faces and quality metrics included; the view picks the cells to draw.
        write_mesh(mesh, preview)
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
