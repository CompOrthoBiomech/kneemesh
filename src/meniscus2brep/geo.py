"""Generation of B-rep geometry."""

import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import gmsh
import numpy as np
from scipy.interpolate import make_interp_spline
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData, vtkPolyLine

from meniscus2brep.vol import ContourSlice

_MAX_SPLINE_ORDER = 25  # Open Cascade's limit on B-spline degree.
_DEFAULT_CAP_OFFSET_RADII = 0.25


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
    gmsh.model.add(f"meniscus2brep-{uuid.uuid4().hex}")
    try:
        yield
    finally:
        gmsh.model.remove()
        if owns_gmsh:
            gmsh.finalize()
        else:
            gmsh.model.setCurrent(previous_model)


def contours_to_brep(
    slices: list[ContourSlice],
    spline_order: int = 3,
    cap: Literal["none", "planar", "smooth"] = "none",
    cap_offset: float | None = None,
) -> BRep:
    """Loft a B-spline surface through contour slices, optionally capping its ends.

    Each contour becomes a periodic B-spline through all of its points, and the
    surface interpolates those curves in the order given. Without caps it spans
    only the first to last slice.

    Args:
        slices: Contours from slice_isocontour, in order along the curve.
        spline_order: Polynomial degree of the contour splines and the maximum
            degree of the surface. 3 is cubic.
        cap: How to close the two ends. "none" leaves an open surface.
            "planar" closes each end with a flat face in its slice's plane.
            "smooth" closes each with a convex dome converging to one point.
            Either cap produces a closed solid.
        cap_offset: For smooth caps, how far each apex lies beyond its slice's
            center along the curve, in physical units. By default, a quarter
            of the radius of a circle with the end slice's area.

    Returns:
        The lofted shape.

    Raises:
        ValueError: If there are fewer than 2 slices, spline_order is outside
            1 to 25, a contour has too few points for spline_order, cap is
            unknown, or cap_offset is not positive.
    """
    if cap not in ("none", "planar", "smooth"):
        raise ValueError(f"cap must be 'none', 'planar', or 'smooth', got {cap!r}")
    if cap_offset is not None and cap_offset <= 0:
        raise ValueError(f"cap_offset must be positive, got {cap_offset}")
    if len(slices) < 2:
        raise ValueError("Lofting needs at least 2 contour slices")
    if not 1 <= spline_order <= _MAX_SPLINE_ORDER:
        raise ValueError(f"spline_order must be between 1 and {_MAX_SPLINE_ORDER}, got {spline_order}")
    if any(len(contour.points) <= spline_order for contour in slices):
        raise ValueError(f"Contours need more than {spline_order} points for spline_order {spline_order}")

    with gmsh_session():
        splines = [_periodic_spline(_project_onto_plane(contour), spline_order) for contour in slices]
        wires = [_add_contour_wire(poles, knots, spline_order) for poles, knots in splines]
        # Open Cascade only honors a loft degree of 1 for a ruled surface.
        loft = gmsh.model.occ.addThruSections(wires, makeSolid=False, makeRuled=spline_order == 1, maxDegree=spline_order)
        ends = [(slices[0], splines[0], wires[0], -1.0), (slices[-1], splines[-1], wires[-1], 1.0)]
        if cap == "planar":
            caps = [gmsh.model.occ.addPlaneSurface([wire]) for _, _, wire, _ in ends]
        elif cap == "smooth":
            caps = [
                _add_smooth_cap(*spline, spline_order, contour, sign * contour.normal, cap_offset)
                for contour, spline, _, sign in ends
            ]
        if cap != "none":
            # addThruSections' makeSolid returns a volume with no end faces.
            shell = gmsh.model.occ.addSurfaceLoop([*(tag for _, tag in loft), *caps], sewing=True)
            gmsh.model.occ.addVolume([shell])
        gmsh.model.occ.synchronize()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shape.brep"
            gmsh.write(str(path))
            return BRep(path.read_text(encoding="utf-8"))


def _project_onto_plane(contour: ContourSlice) -> np.ndarray:
    """Return a contour's points exactly in its plane.

    VTK's single-precision output leaves them about 3e-7 off the plane, which
    is outside Open Cascade's tolerance for building a planar end face.
    """
    heights = (contour.points - contour.center) @ contour.normal
    return contour.points - np.outer(heights, contour.normal)


def _periodic_spline(points: np.ndarray, spline_order: int) -> tuple[np.ndarray, list[float]]:
    """Return the poles and knots of a periodic B-spline interpolating points, without repeats."""
    closed = np.vstack((points, points[:1]))
    lengths = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))))
    spline = make_interp_spline(lengths / lengths[-1], closed, k=spline_order, bc_type="periodic")

    # Scipy repeats the first poles at the end of its coefficients and pads its
    # knots on both sides. Open Cascade takes each only once.
    return spline.c[: len(points)], spline.t[spline_order : len(spline.t) - spline_order].tolist()


def _add_contour_wire(poles: np.ndarray, knots: list[float], spline_order: int) -> int:
    """Add a closed wire of one periodic B-spline, returning its tag."""
    # Repeating the first tag is how gmsh is told the curve is periodic.
    curve = gmsh.model.occ.addBSpline(_closed_tags(poles), degree=spline_order, knots=knots, multiplicities=[1] * len(knots))
    return gmsh.model.occ.addWire([curve], checkClosed=True)


def _closed_tags(poles: np.ndarray) -> list[int]:
    tags = [gmsh.model.occ.addPoint(*pole.tolist()) for pole in poles]
    return [*tags, tags[0]]


def _add_smooth_cap(
    poles: np.ndarray,
    knots: list[float],
    spline_order: int,
    contour: ContourSlice,
    outward: np.ndarray,
    offset: float | None,
) -> int:
    """Add a dome from a contour to one point, returning its surface tag.

    Each meridian is a quadratic Bezier from the contour straight out along the
    curve, then in to the apex, so the dome is convex and meets the apex flat.
    """
    height = _DEFAULT_CAP_OFFSET_RADII * float(np.sqrt(contour.area / np.pi)) if offset is None else offset
    lift = height * outward
    apex = gmsh.model.occ.addPoint(*(contour.center + lift).tolist())
    rows = [*_closed_tags(poles), *_closed_tags(poles + lift), *[apex] * (len(poles) + 1)]
    return gmsh.model.occ.addBSplineSurface(
        rows,
        len(poles) + 1,
        degreeU=spline_order,
        degreeV=2,
        knotsU=knots,
        knotsV=[0.0, 1.0],
        multiplicitiesU=[1] * len(knots),
        multiplicitiesV=[3, 3],
    )


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
