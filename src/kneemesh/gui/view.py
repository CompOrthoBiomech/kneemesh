"""The GUI's 3D scene: each structure's label surface, geometry, and mesh, rendered offscreen for trame's remote view."""

from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

# Rendering needs its factories registered by import.
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401  # pyright: ignore[reportUnusedImport]
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkLookupTable
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkUnstructuredGrid
from vtkmodules.vtkFiltersExtraction import vtkExtractCellsByType
from vtkmodules.vtkFiltersGeometry import vtkDataSetSurfaceFilter
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader
from vtkmodules.vtkRenderingAnnotation import vtkScalarBarActor
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkColorTransferFunction,
    vtkPolyDataMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
)

from kneemesh.config import BONE_STRUCTURES, STRUCTURES, Structure
from kneemesh.gui.labels import LabelEntry
from kneemesh.mesh_optimization import FROBENIUS_ASPECT, SCALED_JACOBIAN, WARP

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
_NAN_COLOR = (0.5, 0.5, 0.5)

# VTK's cell types of volume elements and of surface elements, first and second order.
_VOLUME_CELLS = (10, 12, 24, 25, 29)
_SURFACE_CELLS = (5, 9, 22, 23, 28)


@dataclass(frozen=True)
class Metric:
    """A mesh quality metric the view can color meshes by.

    Attributes:
        name: Its cell array's name in a mesh preview (mesh_optimization.element_quality).
        title: Its name for display.
        higher_is_better: Whether higher values are better, which colors them blue and lower ones red.
        faces: Whether it is measured on boundary faces rather than elements.
    """

    name: str
    title: str
    higher_is_better: bool
    faces: bool = False


METRICS: dict[str, Metric] = {
    metric.name: metric
    for metric in (
        Metric(SCALED_JACOBIAN, "Scaled Jacobian", higher_is_better=True),
        Metric(FROBENIUS_ASPECT, "Frobenius aspect ratio", higher_is_better=False),
        Metric(WARP, "Surface warp (degrees)", higher_is_better=False, faces=True),
    )
}
"""The quality metrics meshes can be colored by, by name."""


@dataclass
class _MeshSurfaces:
    """A mesh's drawn surfaces: its elements' outer faces, carrying element metrics, and its boundary faces, carrying face metrics."""

    elements: vtkPolyData
    faces: vtkPolyData


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
        self._meshes: dict[Structure, _MeshSurfaces] = {}
        self.lookup_table: vtkLookupTable = vtkLookupTable()
        self.scalar_bar: vtkScalarBarActor = vtkScalarBarActor()
        self.scalar_bar.SetLookupTable(self.lookup_table)
        self.scalar_bar.SetNumberOfLabels(5)
        self.scalar_bar.SetLabelFormat("%-#6.3g")
        self.scalar_bar.SetMaximumWidthInPixels(90)
        self.scalar_bar.SetPosition(0.88, 0.1)
        self.scalar_bar.SetHeight(0.8)
        self.scalar_bar.UnconstrainedFontSizeOn()
        self.scalar_bar.SetVerticalTitleSeparation(8)
        for text in (self.scalar_bar.GetTitleTextProperty(), self.scalar_bar.GetLabelTextProperty()):
            text.SetFontSize(14)
            text.SetColor(1.0, 1.0, 1.0)
            text.ItalicOff()
            text.ShadowOff()
        self.scalar_bar.VisibilityOff()
        self.renderer.AddViewProp(self.scalar_bar)

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
        """Show a structure's mesh, its outer faces with their edges.

        The grid holds a volume mesh's elements and its boundary faces, or a
        surface mesh's elements, with their quality metrics as cell arrays.
        """
        volume = _cells_of(grid, _VOLUME_CELLS)
        faces = _cells_of(grid, _SURFACE_CELLS)
        surfaces = _MeshSurfaces(volume if volume.GetNumberOfCells() else faces, faces)
        self._meshes[structure] = surfaces
        actor = _surface_actor(surfaces.elements)
        actor.GetProperty().EdgeVisibilityOn()
        actor.GetProperty().SetEdgeColor(*_EDGE_COLOR)
        actor.GetProperty().SetColor(*COLORS[structure])
        self._replace(self._stages, ("mesh", structure), [actor])

    def clear(self, stage: Stage, structure: Structure) -> None:
        """Remove a structure's geometry or mesh."""
        self._replace(self._stages, (stage, structure), [])
        if stage == "mesh":
            self._meshes.pop(structure, None)

    def metrics(self) -> list[str]:
        """The quality metrics some mesh in the scene carries, in the order of METRICS."""
        return [
            name
            for name, metric in METRICS.items()
            if any(_array(_surface(surfaces, metric), name) is not None for surfaces in self._meshes.values())
        ]

    def has(self, stage: Stage, structure: Structure) -> bool:
        """Whether a structure's geometry or mesh is in the scene."""
        return bool(self._stages.get((stage, structure)))

    def refresh(
        self,
        entries: Sequence[LabelEntry],
        display: DisplayMode,
        hidden: Collection[str],
        solo: Structure | None = None,
        metric: str | None = None,
    ) -> None:
        """Show what display asks for and color it by structure, or meshes by a quality metric.

        "latest" shows each structure's mesh, else its geometry, else its
        included label; another mode shows only that stage. Labels not
        naming a structure show only as labels, in gray. Structures in
        hidden show nothing, and with solo, only that structure shows. With
        a metric of METRICS, the meshes shown are colored by it over the
        range of their values, with a scalar bar; meshes without it are gray.
        """
        if solo is not None:
            hidden = set(STRUCTURES) - {solo}
        for entry in entries:
            structure = entry["structure"]
            visible = entry["include"] and structure not in hidden and (solo is None or structure == solo)
            if structure is not None and display == "latest":
                visible = visible and _latest(self, structure) == "label"
            else:
                visible = visible and display in ("latest", "label")
            color = COLORS[structure] if structure is not None else _UNASSIGNED_COLOR
            for actor in self._labels.get(entry["id"], []):
                actor.SetVisibility(visible)
                actor.GetProperty().SetColor(*color)
                actor.GetProperty().SetOpacity(_BONE_OPACITY if structure in BONE_STRUCTURES else 1.0)
        shown_meshes: list[tuple[vtkActor, _MeshSurfaces]] = []
        for (stage, structure), actors in self._stages.items():
            shown = structure not in hidden and (display == stage or (display == "latest" and _latest(self, structure) == stage))
            for actor in actors:
                actor.SetVisibility(shown)
            if stage == "mesh" and shown:
                shown_meshes.extend((actor, self._meshes[structure]) for actor in actors)
        self._color_meshes(shown_meshes, METRICS.get(metric) if metric is not None else None)

    def _color_meshes(self, meshes: Sequence[tuple[vtkActor, _MeshSurfaces]], metric: Metric | None) -> None:
        values = [] if metric is None else [_array(_surface(surfaces, metric), metric.name) for _, surfaces in meshes]
        finite = np.concatenate([np.zeros(0), *(array[np.isfinite(array)] for array in values if array is not None)])
        for actor, surfaces in meshes:
            mapper = actor.GetMapper()
            assert isinstance(mapper, vtkPolyDataMapper)
            if metric is None or not len(finite):
                mapper.SetInputData(surfaces.elements)
                mapper.ScalarVisibilityOff()
                continue
            mapper.SetInputData(_surface(surfaces, metric))
            mapper.SetScalarModeToUseCellFieldData()
            mapper.SelectColorArray(metric.name)
            mapper.SetLookupTable(self.lookup_table)
            mapper.UseLookupTableScalarRangeOn()
            mapper.ScalarVisibilityOn()
        if metric is None or not len(finite):
            self.scalar_bar.VisibilityOff()
            return
        low, high = float(finite.min()), float(finite.max())
        _fill_lookup_table(self.lookup_table, metric.higher_is_better)
        self.lookup_table.SetTableRange(low, high if high > low else low + 1e-9)
        self.scalar_bar.SetTitle(metric.title.replace(" (", "\n("))
        self.scalar_bar.VisibilityOn()

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


def _cells_of(grid: vtkUnstructuredGrid, cell_types: Iterable[int]) -> vtkPolyData:
    """The outer faces of a grid's cells of some types, as polygons carrying the cells' data."""
    extract = vtkExtractCellsByType()
    extract.SetInputData(grid)
    for cell_type in cell_types:
        extract.AddCellType(cell_type)
    surface = vtkDataSetSurfaceFilter()
    surface.SetInputConnection(extract.GetOutputPort())
    surface.Update()
    output = vtkPolyData()
    output.ShallowCopy(surface.GetOutput())
    return output


def _surface(surfaces: _MeshSurfaces, metric: Metric) -> vtkPolyData:
    """The surface a metric is drawn on: the boundary faces for a face metric, if the mesh has any, or else the elements."""
    return surfaces.faces if metric.faces and surfaces.faces.GetNumberOfCells() else surfaces.elements


def _array(surface: vtkPolyData, name: str) -> np.ndarray | None:
    """A surface's cell array, or None if it lacks it or holds no value of it."""
    array = surface.GetCellData().GetArray(name)
    if array is None:
        return None
    values = vtk_to_numpy(array).astype(np.float64)
    return values if not np.isnan(values).all() else None


def _fill_lookup_table(table: vtkLookupTable, higher_is_better: bool) -> None:
    """Fill a lookup table with a cool-to-warm map, blue for good values and red for bad ones, and gray for NaN."""
    colors = vtkColorTransferFunction()
    colors.SetColorSpaceToDiverging()
    blue, red = (0.230, 0.299, 0.754), (0.706, 0.016, 0.150)
    low, high = (red, blue) if higher_is_better else (blue, red)
    colors.AddRGBPoint(0.0, *low)
    colors.AddRGBPoint(1.0, *high)
    size = 256
    table.SetNumberOfTableValues(size)
    for index in range(size):
        table.SetTableValue(index, *colors.GetColor(index / (size - 1)), 1.0)
    table.SetNanColor(*_NAN_COLOR, 1.0)
    # Values beyond the range, such as the infinite aspect ratio of an inverted element, take the end colors.
    table.UseAboveRangeColorOff()
    table.UseBelowRangeColorOff()
    table.Build()


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
