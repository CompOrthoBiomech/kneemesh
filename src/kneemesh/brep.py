"""B-rep values, gmsh sessions, and B-spline construction shared by every structure."""

import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import gmsh
import numpy as np
from loguru import logger
from scipy.interpolate import BSpline, make_interp_spline
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData, vtkPolyLine

from kneemesh.vol import ContourSlice

MAX_SPLINE_ORDER = 25  # Open Cascade's limit on B-spline degree.


@dataclass(frozen=True)
class BRep:
    """A B-rep shape held as the text of an Open Cascade .brep file."""

    text: str


@contextmanager
def gmsh_session() -> Iterator[None]:
    """Run a block on a temporary gmsh model, leaving gmsh as it was found."""
    owns_gmsh = not gmsh.isInitialized()
    if owns_gmsh:
        gmsh.initialize(readConfigFiles=False, interruptible=False)
        gmsh.option.setNumber("General.Terminal", 0)
    previous_model = gmsh.model.getCurrent()
    gmsh.model.add(f"kneemesh-{uuid.uuid4().hex}")
    try:
        yield
    finally:
        gmsh.model.remove()
        if owns_gmsh:
            gmsh.finalize()
        else:
            gmsh.model.setCurrent(previous_model)


def export_brep() -> BRep:
    """Return the current gmsh model's highest-dimension shapes as a BRep. Call inside gmsh_session."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "shape.brep"
        gmsh.write(str(path))
        text = path.read_text(encoding="utf-8")
    counts = [len(gmsh.model.getEntities(dim)) for dim in range(4)]
    logger.debug("B-rep has {} vertices, {} edges, {} faces, and {} volumes ({} kB)", *counts, len(text) // 1024)
    return BRep(text)


def chord_parameters(points: np.ndarray) -> np.ndarray:
    """Return parameters from 0 to 1 proportional to chord length around a closed contour, ending at the repeated first point."""
    closed = np.vstack((points, points[:1]))
    lengths = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))))
    return lengths / lengths[-1]


def periodic_spline(points: np.ndarray, spline_order: int) -> BSpline:
    """Return a periodic B-spline through points, over parameters 0 to 1 proportional to chord length."""
    return make_interp_spline(chord_parameters(points), np.vstack((points, points[:1])), k=spline_order, bc_type="periodic")


def open_chord_parameters(points: np.ndarray) -> np.ndarray:
    """Return parameters from 0 to 1 proportional to chord length along an open sequence of points."""
    lengths = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    return lengths / lengths[-1]


def clamped_interpolation(parameters: np.ndarray, points: np.ndarray, degree: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the knots and poles of a clamped B-spline through points at parameters.

    Averaged knots make this work at any degree. Both ends are pinned to the
    exact end points, so curves fitted to the same end points meet exactly.
    """
    if degree == 1:
        knots = np.concatenate(([parameters[0]], parameters, [parameters[-1]]))
    else:
        interior = [parameters[index + 1 : index + degree + 1].mean() for index in range(len(parameters) - degree - 1)]
        knots = np.concatenate(([parameters[0]] * (degree + 1), interior, [parameters[-1]] * (degree + 1)))
    poles = np.array(make_interp_spline(parameters, points, k=degree, t=knots).c)
    poles[0], poles[-1] = points[0], points[-1]
    return knots, poles


def add_bspline_curve(knots: np.ndarray, poles: np.ndarray, degree: int) -> int:
    """Add an open B-spline curve from scipy-style full knots and poles, returning its tag."""
    unique_knots, multiplicities = np.unique(knots, return_counts=True)
    tags = [gmsh.model.occ.addPoint(*pole.tolist()) for pole in poles]
    return gmsh.model.occ.addBSpline(tags, degree=degree, knots=unique_knots.tolist(), multiplicities=multiplicities.tolist())


def add_swept_surface(rows: list[tuple[np.ndarray, np.ndarray]], degree: int, along: np.ndarray, along_degree: int) -> int:
    """Add a B-spline surface through section curves sharing one knot vector, returning its tag.

    Each column of poles is interpolated along the curve at the same
    parameters, so a surface's edge depends only on that edge's points and
    matches its neighbor's exactly.
    """
    knots = rows[0][0]
    grid = np.array([poles for _, poles in rows])
    columns = [clamped_interpolation(along, grid[:, column], along_degree) for column in range(grid.shape[1])]
    along_knots = columns[0][0]
    poles = np.stack([column_poles for _, column_poles in columns], axis=1)
    unique_knots, multiplicities = np.unique(knots, return_counts=True)
    unique_along, along_multiplicities = np.unique(along_knots, return_counts=True)
    return gmsh.model.occ.addBSplineSurface(
        [gmsh.model.occ.addPoint(*pole.tolist()) for row in poles for pole in row],
        poles.shape[1],
        degreeU=degree,
        degreeV=along_degree,
        knotsU=unique_knots.tolist(),
        knotsV=unique_along.tolist(),
        multiplicitiesU=multiplicities.tolist(),
        multiplicitiesV=along_multiplicities.tolist(),
    )


def periodic_poles_and_knots(spline: BSpline, point_count: int) -> tuple[np.ndarray, list[float]]:
    """Return a periodic spline's poles and knots as Open Cascade takes them."""
    # Scipy repeats the first poles at the end of its coefficients and pads its
    # knots on both sides. Open Cascade takes each only once.
    return spline.c[:point_count], spline.t[spline.k : len(spline.t) - spline.k].tolist()


def add_closed_wire(spline: BSpline, point_count: int) -> int:
    """Add a closed wire of one periodic B-spline, returning its tag."""
    poles, knots = periodic_poles_and_knots(spline, point_count)
    # Repeating the first tag is how gmsh is told the curve is periodic.
    curve = gmsh.model.occ.addBSpline(add_closed_points(poles), degree=spline.k, knots=knots, multiplicities=[1] * len(knots))
    return gmsh.model.occ.addWire([curve], checkClosed=True)


def split_periodic_spline(spline: BSpline, parameters: np.ndarray) -> list[BSpline]:
    """Split a periodic spline exactly into clamped splines at parameters.

    parameters must be sorted, distinct, and start at 0, so the last piece ends at 1.
    """
    degree = spline.k
    splits = [float(parameter) for parameter in parameters] + [1.0]

    # Raising a knot's multiplicity to the degree makes the curve pass through
    # a pole there, so the poles between two such knots define that piece alone.
    refined = BSpline(spline.t, spline.c, degree)
    for split in splits:
        missing = degree - int(np.sum(refined.t == split))
        if missing > 0:
            refined = refined.insert_knot(split, missing)
    # Index from a start knot's last copy and an end knot's first, which also
    # holds where scipy clamps the ends of a degree 1 spline instead of padding.
    pieces = []
    for start, end in pairwise(splits):
        after_start = int(np.flatnonzero(refined.t == start)[-1]) + 1
        at_end = int(np.flatnonzero(refined.t == end)[0])
        knots = np.concatenate(([start] * (degree + 1), refined.t[after_start:at_end], [end] * (degree + 1)))
        pieces.append(BSpline(knots, refined.c[after_start - 1 - degree : at_end], degree))
    return pieces


def add_closed_points(poles: np.ndarray) -> list[int]:
    """Add a point per pole, returning their tags with the first repeated at the end.

    gmsh treats a B-spline whose first and last point tags match as periodic.
    """
    tags = [gmsh.model.occ.addPoint(*pole.tolist()) for pole in poles]
    return [*tags, tags[0]]


def contours_to_polydata(slices: list[ContourSlice]) -> vtkPolyData:
    """Assemble contour slices into a single poly data of closed polylines.

    Args:
        slices: Contours from slice_isocontour.

    Returns:
        A vtkPolyData with one vtkPolyLine per slice, in the order given. Each
        polyline ends on its own first point id to close the contour, so no
        point is duplicated.

    Raises:
        ValueError: If slices is empty.
    """
    if not slices:
        raise ValueError("No contour slices to assemble")

    points = vtkPoints()
    points.SetData(numpy_to_vtk(np.vstack([contour.points for contour in slices]), deep=True))

    lines = vtkCellArray()
    first_id = 0
    for contour in slices:
        count = len(contour.points)
        polyline = vtkPolyLine()
        ids = polyline.GetPointIds()
        ids.SetNumberOfIds(count + 1)
        for index in range(count):
            ids.SetId(index, first_id + index)
        ids.SetId(count, first_id)
        lines.InsertNextCell(polyline)
        first_id += count

    poly_data = vtkPolyData()
    poly_data.SetPoints(points)
    poly_data.SetLines(lines)
    return poly_data
