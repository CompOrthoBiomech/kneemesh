"""The GUI's 3D scene: each structure's label surface, geometry, and mesh, rendered offscreen for trame's remote view.

Everything shown runs through a pipeline (_Display): optionally the cells of
some types, a crinkle clip by the scene's plane when clipping is on, which
keeps whole cells, the outer faces with nonlinear cells subdivided twice, and
point normals split at feature edges, as ParaView's Generate Surface Normals
makes them, for smooth shading.
"""

from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, override

import numpy as np

# Rendering needs its factories registered by import.
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401  # pyright: ignore[reportUnusedImport]
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.vtkCommonCore import vtkCommand, vtkLookupTable, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkDataSet, vtkPlane, vtkPolyData, vtkUnstructuredGrid
from vtkmodules.vtkCommonExecutionModel import vtkAlgorithm, vtkTrivialProducer
from vtkmodules.vtkFiltersCore import vtkPolyDataNormals
from vtkmodules.vtkFiltersExtraction import vtkExtractCellsByType, vtkExtractGeometry
from vtkmodules.vtkFiltersGeometry import vtkDataSetSurfaceFilter, vtkUnstructuredGridGeometryFilter
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

SUBDIVISION_LEVEL = 2
"""Times each nonlinear face is subdivided for display, so a second-order edge is drawn as 2^2 = 4 segments."""

FEATURE_ANGLE = 30.0
"""Angle in degrees between faces above which point normals are split, so the edge stays sharp: ParaView's default."""

_UNASSIGNED_COLOR = (0.6, 0.6, 0.6)
_EDGE_COLOR = (0.15, 0.15, 0.15)
_BONE_OPACITY = 0.6
_NAN_COLOR = (0.5, 0.5, 0.5)

# VTK's cell types of volume elements and of surface elements, first and second order, and those of second order.
_VOLUME_CELLS = (10, 12, 24, 25, 29)
_SURFACE_CELLS = (5, 9, 22, 23, 28)
_NONLINEAR_CELLS = frozenset({22, 23, 24, 25, 28, 29})

# The edges of each VTK surface cell type: corner pairs of linear cells, and corners then middle node of quadratic ones.
_LINEAR_EDGES = {5: ((0, 1), (1, 2), (2, 0)), 9: ((0, 1), (1, 2), (2, 3), (3, 0))}
_QUADRATIC_EDGES = {
    22: ((0, 1, 3), (1, 2, 4), (2, 0, 5)),
    23: ((0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 0, 7)),
    28: ((0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 0, 7)),
}

type ClipMode = Literal["view", "fixed"]


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


class _Display:
    """The pipeline drawing a data set: its cells of some types, crinkle-clipped when clipping, its outer faces, and their normals.

    The crinkle clip (vtkExtractGeometry with boundary cells) keeps every
    cell on the far side of the plane or crossing it, whole, so a mesh's
    elements show as they are, not cut. Nonlinear cells' faces are
    subdivided SUBDIVISION_LEVEL times.
    """

    def __init__(self, data: vtkDataSet, plane: vtkPlane, cell_types: Iterable[int] | None = None, normals: bool = True) -> None:
        producer = vtkTrivialProducer()
        producer.SetOutput(data)
        self.source: vtkAlgorithm = producer
        if cell_types is not None:
            extract = vtkExtractCellsByType()
            extract.SetInputConnection(producer.GetOutputPort())
            for cell_type in cell_types:
                extract.AddCellType(cell_type)
            self.source = extract
        self.source.Update()
        self.clip: vtkExtractGeometry = vtkExtractGeometry()
        self.clip.SetInputConnection(self.source.GetOutputPort())
        self.clip.SetImplicitFunction(plane)
        self.clip.ExtractInsideOn()
        self.clip.ExtractBoundaryCellsOn()
        self.surface: vtkDataSetSurfaceFilter = vtkDataSetSurfaceFilter()
        self.surface.SetNonlinearSubdivisionLevel(SUBDIVISION_LEVEL)
        self.output: vtkAlgorithm = self.surface
        if normals:
            point_normals = vtkPolyDataNormals()
            point_normals.SetInputConnection(self.surface.GetOutputPort())
            point_normals.SetFeatureAngle(FEATURE_ANGLE)
            point_normals.SplittingOn()
            point_normals.ConsistencyOn()
            point_normals.NonManifoldTraversalOn()
            point_normals.ComputePointNormalsOn()
            self.output = point_normals
        self.set_clipped(False)

    @property
    def data(self) -> vtkDataSet:
        """The unclipped cells drawn."""
        return self.source.GetOutputDataObject(0)

    @property
    def cells(self) -> vtkAlgorithm:
        """The algorithm whose output is the cells drawn, clipped or not."""
        return self._cells

    def set_clipped(self, clipped: bool) -> None:
        """Draw the cells crinkle-clipped by the plane, or all of them."""
        self._cells: vtkAlgorithm = self.clip if clipped else self.source
        self.surface.SetInputConnection(self._cells.GetOutputPort())

    def is_empty(self) -> bool:
        """Whether there are no cells to draw."""
        return not self.data.GetNumberOfCells()


class _EdgeLines(VTKPythonAlgorithmBase):
    """The edges of a grid's surface cells as lines, each quadratic edge sampled at 2^SUBDIVISION_LEVEL + 1 points along its curve.

    Drawing a subdivided nonlinear face's edges would draw the subdivision's
    triangles too, so a nonlinear mesh's edges are drawn by this instead.
    """

    def __init__(self) -> None:
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=1, inputType="vtkUnstructuredGrid", nOutputPorts=1, outputType="vtkPolyData")

    @override
    def RequestData(self, request: Any, inInfo: Any, outInfo: Any) -> int:
        grid = vtkUnstructuredGrid.GetData(inInfo[0])
        output = vtkPolyData.GetData(outInfo)
        output.ShallowCopy(edge_lines(grid))
        return 1


def edge_lines(grid: vtkUnstructuredGrid) -> vtkPolyData:
    """The edges of a grid's triangles and quadrilaterals, linear or quadratic, each once, as polylines following their curves."""
    lines = vtkPolyData()
    if not grid.GetNumberOfCells():
        return lines
    points = vtk_to_numpy(grid.GetPoints().GetData()).astype(np.float64)
    types = vtk_to_numpy(grid.GetCellTypes())
    offsets = vtk_to_numpy(grid.GetCells().GetOffsetsArray())[:-1]
    connectivity = vtk_to_numpy(grid.GetCells().GetConnectivityArray())
    polylines: list[np.ndarray] = []
    for table, samples in ((_LINEAR_EDGES, 2), (_QUADRATIC_EDGES, 2**SUBDIVISION_LEVEL + 1)):
        rows = [connectivity[offsets[types == cell_type][:, None] + np.array(edge)] for cell_type, edges in table.items() for edge in edges]
        rows = [row for row in rows if len(row)]
        if not rows:
            continue
        edges = np.vstack(rows)
        # Each edge once, by its sorted corners.
        _, first = np.unique(np.sort(edges[:, :2], axis=1), axis=0, return_index=True)
        edges = edges[first]
        t = np.linspace(0.0, 1.0, samples)[None, :, None]
        start, end = points[edges[:, 0]][:, None], points[edges[:, 1]][:, None]
        if samples == 2:
            polylines.append(start + t * (end - start))
        else:
            middle = points[edges[:, 2]][:, None]
            polylines.append(start * (1 - t) * (1 - 2 * t) + end * t * (2 * t - 1) + middle * 4 * t * (1 - t))
    if not polylines:
        return lines
    vertices = np.concatenate([line.reshape(-1, 3) for line in polylines])
    sizes = np.concatenate([np.full(len(line), line.shape[1]) for line in polylines])
    vtk_points = vtkPoints()
    vtk_points.SetData(numpy_to_vtk(vertices, deep=True))
    cells = vtkCellArray()
    cells.SetData(
        numpy_to_vtkIdTypeArray(np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64), deep=True),
        numpy_to_vtkIdTypeArray(np.arange(len(vertices), dtype=np.int64), deep=True),
    )
    lines.SetPoints(vtk_points)
    lines.SetLines(cells)
    return lines


@dataclass
class _Item:
    """What the scene shows for a label or one stage of a structure: its actors and the pipelines drawing them."""

    actors: list[vtkActor]
    displays: list[_Display]


@dataclass
class _MeshItem(_Item):
    """A mesh's actor, drawing either its elements, carrying element metrics, or its boundary faces, carrying face metrics.

    A nonlinear mesh draws its edges with a second actor, through the
    edge pipelines of its elements and of its faces.
    """

    elements: _Display = field(kw_only=True)
    faces: _Display = field(kw_only=True)
    edges: dict[str, vtkAlgorithm] = field(kw_only=True, default_factory=dict[str, vtkAlgorithm])

    def show(self, faces: bool) -> None:
        """Draw the boundary faces, if the mesh has any, or else the elements."""
        name = "faces" if faces and not self.faces.is_empty() else "elements"
        display = self.faces if name == "faces" else self.elements
        _mapper(self.actors[0]).SetInputConnection(display.output.GetOutputPort())
        if self.edges:
            _mapper(self.actors[1]).SetInputConnection(self.edges[name].GetOutputPort())

    def surface(self, faces: bool) -> _Display:
        """The display a metric of elements, or of faces, is drawn on."""
        return self.faces if faces and not self.faces.is_empty() else self.elements


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
        self._labels: dict[int, _Item] = {}
        self._stages: dict[tuple[Stage, Structure], _Item] = {}
        self.plane: vtkPlane = vtkPlane()
        self.clipping: bool = False
        self.clip_mode: ClipMode = "view"
        self.clip_offset: float = 0.0
        self._fixed_plane: tuple[np.ndarray, np.ndarray] | None = None
        # The plane follows the camera as it renders, so it stays facing the viewer however the view turns.
        self.renderer.AddObserver(vtkCommand.StartEvent, lambda *_: self._follow_camera())
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
        if surface is None:
            self._replace(self._labels, entry_id, None)
            return
        display = self._display(surface)
        self._replace(self._labels, entry_id, _Item([_actor(display)], [display]))

    def set_geometry(self, structure: Structure, surface: vtkPolyData) -> None:
        """Show a structure's geometry: a B-rep preview whose lines are its curves, or a bone's surface."""
        displays = [self._display(_faces(surface))]
        if surface.GetNumberOfLines():
            displays.append(self._display(_curves(surface), normals=False))
        actors = [_actor(display) for display in displays]
        actors[0].GetProperty().SetColor(*COLORS[structure])
        if structure in BONE_STRUCTURES:
            actors[0].GetProperty().SetOpacity(_BONE_OPACITY)
        if len(actors) > 1:
            actors[1].GetProperty().SetColor(*_EDGE_COLOR)
            actors[1].GetProperty().SetLineWidth(2)
        self._replace(self._stages, ("geometry", structure), _Item(actors, displays))

    def set_mesh(self, structure: Structure, grid: vtkUnstructuredGrid) -> None:
        """Show a structure's mesh, its outer faces with their edges.

        The grid holds a volume mesh's elements and its boundary faces, or a
        surface mesh's elements, with their quality metrics as cell arrays.
        """
        volume = self._display(grid, _VOLUME_CELLS)
        faces = self._display(grid, _SURFACE_CELLS)
        elements = volume if not volume.is_empty() else faces
        surface = _actor(elements)
        surface.GetProperty().SetColor(*COLORS[structure])
        actors = [surface]
        edges: dict[str, vtkAlgorithm] = {}
        if _NONLINEAR_CELLS & set(np.unique(vtk_to_numpy(grid.GetCellTypes())).tolist()):
            # Subdivided faces would show their subdivision as edges, so the cells' own edges are drawn along their curves.
            for name, display in (("elements", elements), ("faces", faces)):
                boundary = vtkUnstructuredGridGeometryFilter()
                boundary.SetInputConnection(display.cells.GetOutputPort())
                lines = _EdgeLines()
                lines.SetInputConnection(boundary.GetOutputPort())
                edges[name] = lines
            # The edges must follow the clip, so they take the displays' current cells; set_clipping reconnects them.
            edge_actor = vtkActor()
            edge_mapper = vtkPolyDataMapper()
            edge_mapper.SetResolveCoincidentTopologyToPolygonOffset()
            edge_actor.SetMapper(edge_mapper)
            edge_actor.GetProperty().SetColor(*_EDGE_COLOR)
            actors.append(edge_actor)
        else:
            surface.GetProperty().EdgeVisibilityOn()
            surface.GetProperty().SetEdgeColor(*_EDGE_COLOR)
        item = _MeshItem(actors, [volume, faces], elements=elements, faces=faces, edges=edges)
        item.show(faces=False)
        self._replace(self._stages, ("mesh", structure), item)

    def clear(self, stage: Stage, structure: Structure) -> None:
        """Remove a structure's geometry or mesh."""
        self._replace(self._stages, (stage, structure), None)

    def metrics(self) -> list[str]:
        """The quality metrics some mesh in the scene carries, in the order of METRICS."""
        meshes = [item for item in self._stages.values() if isinstance(item, _MeshItem)]
        return [name for name, metric in METRICS.items() if any(_array(item.surface(metric.faces).data, name) is not None for item in meshes)]

    def has(self, stage: Stage, structure: Structure) -> bool:
        """Whether a structure's geometry or mesh is in the scene."""
        return (stage, structure) in self._stages

    def set_clipping(self, enabled: bool, mode: ClipMode = "view", offset: float = 0.0) -> None:
        """Crinkle-clip everything by a plane, keeping the cells beyond it and those it crosses, whole.

        In "view" mode the plane faces the viewer through the camera's focal
        point and turns with the camera. "fixed" freezes the plane as it is
        when the mode is chosen, so it turns with the geometry. offset moves
        the plane away from the viewer, as a fraction of the scene's radius.
        """
        if mode == "fixed" and (self._fixed_plane is None or not self.clipping):
            self._fixed_plane = self._camera_plane()
        elif mode == "view":
            self._fixed_plane = None
        self.clipping, self.clip_mode, self.clip_offset = enabled, mode, offset
        for item in (*self._labels.values(), *self._stages.values()):
            for display in item.displays:
                display.set_clipped(enabled)
            if isinstance(item, _MeshItem):
                for name, lines in item.edges.items():
                    display = item.faces if name == "faces" else item.elements
                    lines.GetInputAlgorithm().SetInputConnection(display.cells.GetOutputPort())
        self._place_plane()

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
            item = self._labels.get(entry["id"])
            for actor in item.actors if item is not None else []:
                actor.SetVisibility(visible)
                actor.GetProperty().SetColor(*color)
                actor.GetProperty().SetOpacity(_BONE_OPACITY if structure in BONE_STRUCTURES else 1.0)
        shown_meshes: list[_MeshItem] = []
        for (stage, structure), item in self._stages.items():
            shown = structure not in hidden and (display == stage or (display == "latest" and _latest(self, structure) == stage))
            for actor in item.actors:
                actor.SetVisibility(shown)
            if isinstance(item, _MeshItem) and shown:
                shown_meshes.append(item)
        self._color_meshes(shown_meshes, METRICS.get(metric) if metric is not None else None)

    def reset_camera(self) -> None:
        """Fit the camera to the visible actors."""
        self.renderer.ResetCamera()

    def _display(self, data: vtkDataSet, cell_types: Iterable[int] | None = None, normals: bool = True) -> _Display:
        display = _Display(data, self.plane, cell_types, normals)
        display.set_clipped(self.clipping)
        return display

    def _color_meshes(self, meshes: Sequence[_MeshItem], metric: Metric | None) -> None:
        values = [] if metric is None else [_array(item.surface(metric.faces).data, metric.name) for item in meshes]
        finite = np.concatenate([np.zeros(0), *(array[np.isfinite(array)] for array in values if array is not None)])
        colored = metric is not None and len(finite) > 0
        for item in meshes:
            item.show(faces=colored and metric is not None and metric.faces)
            mapper = _mapper(item.actors[0])
            if not colored or metric is None:
                mapper.ScalarVisibilityOff()
                continue
            mapper.SetScalarModeToUseCellFieldData()
            mapper.SelectColorArray(metric.name)
            mapper.SetLookupTable(self.lookup_table)
            mapper.UseLookupTableScalarRangeOn()
            mapper.ScalarVisibilityOn()
        if not colored or metric is None:
            self.scalar_bar.VisibilityOff()
            return
        low, high = float(finite.min()), float(finite.max())
        _fill_lookup_table(self.lookup_table, metric.higher_is_better)
        self.lookup_table.SetTableRange(low, high if high > low else low + 1e-9)
        self.scalar_bar.SetTitle(metric.title.replace(" (", "\n("))
        self.scalar_bar.VisibilityOn()

    def _camera_plane(self) -> tuple[np.ndarray, np.ndarray]:
        """The plane through the camera's focal point facing it: its normal, toward the viewer, and its center."""
        camera = self.renderer.GetActiveCamera()
        return -np.array(camera.GetDirectionOfProjection()), np.array(camera.GetFocalPoint())

    def _follow_camera(self) -> None:
        if self.clipping and self.clip_mode == "view":
            self._place_plane()

    def _place_plane(self) -> None:
        normal, center = self._fixed_plane if self._fixed_plane is not None else self._camera_plane()
        # The clip keeps the side the normal points away from, so a positive offset, away from the viewer, cuts deeper.
        origin = center - self.clip_offset * self._radius() * normal
        self.plane.SetNormal(*normal.tolist())
        self.plane.SetOrigin(*origin.tolist())

    def _radius(self) -> float:
        """Half the diagonal of the bounding box of everything in the scene, unclipped."""
        bounds = [item_display.data.GetBounds() for item in (*self._labels.values(), *self._stages.values()) for item_display in item.displays]
        bounds = [box for box in bounds if box[0] <= box[1]]
        if not bounds:
            return 1.0
        boxes = np.array(bounds)
        low, high = boxes[:, 0::2].min(axis=0), boxes[:, 1::2].max(axis=0)
        return float(np.linalg.norm(high - low)) / 2

    def _replace[Key](self, items: dict[Key, _Item], key: Key, new: _Item | None) -> None:
        old = items.pop(key, None)
        for actor in old.actors if old is not None else []:
            self.renderer.RemoveActor(actor)
        if new is not None:
            for actor in new.actors:
                self.renderer.AddActor(actor)
            items[key] = new


def _latest(scene: Scene, structure: Structure) -> Stage:
    return "mesh" if scene.has("mesh", structure) else "geometry" if scene.has("geometry", structure) else "label"


def _mapper(actor: vtkActor) -> vtkPolyDataMapper:
    mapper = actor.GetMapper()
    assert isinstance(mapper, vtkPolyDataMapper)
    return mapper


def _actor(display: _Display) -> vtkActor:
    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(display.output.GetOutputPort())
    mapper.ScalarVisibilityOff()
    # Keep edges and curves from being hidden by the faces they bound.
    mapper.SetResolveCoincidentTopologyToPolygonOffset()
    actor = vtkActor()
    actor.SetMapper(mapper)
    return actor


def _faces(surface: vtkPolyData) -> vtkPolyData:
    faces = vtkPolyData()
    faces.SetPoints(surface.GetPoints())
    faces.SetPolys(surface.GetPolys())
    return faces


def _curves(surface: vtkPolyData) -> vtkPolyData:
    curves = vtkPolyData()
    curves.SetPoints(surface.GetPoints())
    curves.SetLines(surface.GetLines())
    return curves


def _array(data: vtkDataSet, name: str) -> np.ndarray | None:
    """A data set's cell array, or None if it lacks it or holds no value of it."""
    array = data.GetCellData().GetArray(name)
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
