"""The GUI's 3D scene: each structure's label surface, geometry, and mesh, rendered offscreen for trame's remote view."""

from collections.abc import Collection, Iterable, Sequence
from pathlib import Path
from typing import Literal

# Rendering needs its factories registered by import.
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401  # pyright: ignore[reportUnusedImport]
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkUnstructuredGrid
from vtkmodules.vtkFiltersGeometry import vtkDataSetSurfaceFilter
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor

from kneemesh.config import BONE_STRUCTURES, STRUCTURES, Structure
from kneemesh.gui.labels import LabelEntry

type Stage = Literal["label", "geometry", "mesh"]
type DisplayMode = Literal["latest", "label", "geometry", "mesh"]

STAGES: tuple[Stage, ...] = ("label", "geometry", "mesh")

COLORS: dict[Structure, tuple[float, float, float]] = {
    "femur": (0.89, 0.85, 0.76),
    "tibia": (0.85, 0.82, 0.72),
    "patella": (0.92, 0.88, 0.80),
    "lateral_meniscus": (0.47, 0.67, 0.89),
    "medial_meniscus": (0.33, 0.55, 0.82),
    "lateral_tibial_cartilage": (0.93, 0.55, 0.50),
    "medial_tibial_cartilage": (0.87, 0.45, 0.45),
    "patellar_cartilage": (0.95, 0.66, 0.42),
    "femoral_cartilage": (0.80, 0.50, 0.70),
}
"""Each structure's color in the scene."""

_UNASSIGNED_COLOR = (0.6, 0.6, 0.6)
_EDGE_COLOR = (0.15, 0.15, 0.15)
_BONE_OPACITY = 0.6


class Scene:
    """Actors for each structure's stages and each imported label, and the offscreen window that renders them."""

    def __init__(self) -> None:
        self.renderer: vtkRenderer = vtkRenderer()
        self.renderer.SetBackground(0.32, 0.34, 0.43)
        self.renderer.SetBackground2(0.08, 0.08, 0.12)
        self.renderer.GradientBackgroundOn()
        self.render_window: vtkRenderWindow = vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.OffScreenRenderingOn()
        self.interactor: vtkRenderWindowInteractor = vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        self.interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
        self._labels: dict[int, list[vtkActor]] = {}
        self._stages: dict[tuple[Stage, Structure], list[vtkActor]] = {}

    def set_label(self, entry_id: int, surface: vtkPolyData | None) -> None:
        """Show an imported label's surface, or remove it with None."""
        self._replace(self._labels, entry_id, [_surface_actor(surface)] if surface is not None else [])

    def set_geometry(self, structure: Structure, surface: vtkPolyData) -> None:
        """Show a structure's geometry: a B-rep preview whose "edge" cells are its curves, or a bone's surface."""
        actors = _geometry_actors(surface)
        actors[0].GetProperty().SetColor(*COLORS[structure])
        if structure in BONE_STRUCTURES:
            actors[0].GetProperty().SetOpacity(_BONE_OPACITY)
        self._replace(self._stages, ("geometry", structure), actors)

    def set_mesh(self, structure: Structure, grid: vtkUnstructuredGrid) -> None:
        """Show a structure's mesh, its outer faces with their edges."""
        surface = vtkDataSetSurfaceFilter()
        surface.SetInputData(grid)
        surface.Update()
        actor = _surface_actor(surface.GetOutput())
        actor.GetProperty().EdgeVisibilityOn()
        actor.GetProperty().SetEdgeColor(*_EDGE_COLOR)
        actor.GetProperty().SetColor(*COLORS[structure])
        self._replace(self._stages, ("mesh", structure), [actor])

    def clear(self, stage: Stage, structure: Structure) -> None:
        """Remove a structure's geometry or mesh."""
        self._replace(self._stages, (stage, structure), [])

    def has(self, stage: Stage, structure: Structure) -> bool:
        """Whether a structure's geometry or mesh is in the scene."""
        return bool(self._stages.get((stage, structure)))

    def refresh(self, entries: Sequence[LabelEntry], display: DisplayMode, hidden: Collection[str]) -> None:
        """Show what display asks for and color it by structure.

        "latest" shows each structure's mesh, else its geometry, else its
        included label; another mode shows only that stage. Labels not
        naming a structure show only as labels, in gray. Structures in
        hidden show nothing.
        """
        for entry in entries:
            structure = entry["structure"]
            visible = entry["include"] and structure not in hidden
            if structure is not None and display == "latest":
                visible = visible and _latest(self, structure) == "label"
            else:
                visible = visible and display in ("latest", "label")
            color = COLORS[structure] if structure is not None else _UNASSIGNED_COLOR
            for actor in self._labels.get(entry["id"], []):
                actor.SetVisibility(visible)
                actor.GetProperty().SetColor(*color)
                actor.GetProperty().SetOpacity(_BONE_OPACITY if structure in BONE_STRUCTURES else 1.0)
        for (stage, structure), actors in self._stages.items():
            shown = structure not in hidden and (display == stage or (display == "latest" and _latest(self, structure) == stage))
            for actor in actors:
                actor.SetVisibility(shown)

    def reset_camera(self) -> None:
        """Fit the camera to the visible actors."""
        self.renderer.ResetCamera()

    def _replace[Key](self, actors: dict[Key, list[vtkActor]], key: Key, new: Iterable[vtkActor]) -> None:
        for actor in actors.pop(key, []):
            self.renderer.RemoveActor(actor)
        new = list(new)
        for actor in new:
            self.renderer.AddActor(actor)
        if new:
            actors[key] = new


def _latest(scene: Scene, structure: Structure) -> Stage:
    return "mesh" if scene.has("mesh", structure) else "geometry" if scene.has("geometry", structure) else "label"


def _surface_actor(surface: vtkPolyData) -> vtkActor:
    mapper = vtkPolyDataMapper()
    mapper.SetInputData(surface)
    mapper.ScalarVisibilityOff()
    actor = vtkActor()
    actor.SetMapper(mapper)
    return actor


def _geometry_actors(surface: vtkPolyData) -> list[vtkActor]:
    """A face actor, and for a B-rep preview an actor of its curves in the edge color."""
    if not surface.GetNumberOfLines():
        return [_surface_actor(surface)]
    faces = vtkPolyData()
    faces.SetPoints(surface.GetPoints())
    faces.SetPolys(surface.GetPolys())
    curves = vtkPolyData()
    curves.SetPoints(surface.GetPoints())
    curves.SetLines(surface.GetLines())
    curve_actor = _surface_actor(curves)
    curve_actor.GetProperty().SetColor(*_EDGE_COLOR)
    curve_actor.GetProperty().SetLineWidth(2)
    # Keep the curves from being hidden by the faces they bound.
    curve_actor.GetMapper().SetResolveCoincidentTopologyToPolygonOffset()
    face_actor = _surface_actor(faces)
    face_actor.GetMapper().SetResolveCoincidentTopologyToPolygonOffset()
    return [face_actor, curve_actor]


def read_mesh_preview(path: str | Path) -> vtkUnstructuredGrid:
    """Read a mesh preview the worker wrote as .vtu.

    Raises:
        ValueError: If the file is not a VTK XML unstructured grid.
    """
    reader = vtkXMLUnstructuredGridReader()
    if not reader.CanReadFile(str(path)):
        raise ValueError(f"Not a VTK XML unstructured grid: {path}")
    reader.SetFileName(str(path))
    reader.Update()
    return reader.GetOutput()


def structure_title(structure: str) -> str:
    """A structure's name for display: lateral_meniscus as Lateral Meniscus."""
    return structure.replace("_", " ").title()


STRUCTURE_ITEMS = [{"title": structure_title(structure), "value": structure} for structure in STRUCTURES]
"""Choices of a structure select."""
