"""Generation of B-rep geometry."""

import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import gmsh
import numpy as np
from scipy.interpolate import make_interp_spline
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData, vtkPolyLine

from meniscus2brep.vol import ContourSlice

_MAX_SPLINE_ORDER = 25  # Open Cascade's limit on B-spline degree.


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


def contours_to_brep(slices: list[ContourSlice], spline_order: int = 3, solid: bool = False) -> BRep:
    """Loft a B-spline surface through contour slices.

    Each contour becomes a periodic B-spline through all of its points, and the
    surface interpolates those curves in the order given. It spans only the
    first to last slice.

    Args:
        slices: Contours from slice_isocontour, in order along the curve.
        spline_order: Polynomial degree of the contour splines and the maximum
            degree of the surface. 3 is cubic.
        solid: Cap the two ends to make a closed solid instead of an open surface.

    Returns:
        The lofted shape.

    Raises:
        ValueError: If there are fewer than 2 slices, spline_order is outside
            1 to 25, or a contour has too few points for spline_order.
    """
    if len(slices) < 2:
        raise ValueError("Lofting needs at least 2 contour slices")
    if not 1 <= spline_order <= _MAX_SPLINE_ORDER:
        raise ValueError(f"spline_order must be between 1 and {_MAX_SPLINE_ORDER}, got {spline_order}")
    if any(len(contour.points) <= spline_order for contour in slices):
        raise ValueError(f"Contours need more than {spline_order} points for spline_order {spline_order}")

    with gmsh_session():
        wires = [_add_contour_wire(_project_onto_plane(contour), spline_order) for contour in slices]
        # Open Cascade only honors a loft degree of 1 for a ruled surface.
        loft = gmsh.model.occ.addThruSections(wires, makeSolid=False, makeRuled=spline_order == 1, maxDegree=spline_order)
        if solid:
            _close_with_planar_caps([tag for _, tag in loft], wires[0], wires[-1])
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


def _close_with_planar_caps(surfaces: list[int], first_wire: int, last_wire: int) -> None:
    # addThruSections' makeSolid returns a volume with no end faces.
    caps = [gmsh.model.occ.addPlaneSurface([wire]) for wire in (first_wire, last_wire)]
    gmsh.model.occ.addVolume([gmsh.model.occ.addSurfaceLoop([*surfaces, *caps], sewing=True)])


def _add_contour_wire(points: np.ndarray, spline_order: int) -> int:
    """Add a closed wire of one periodic B-spline interpolating points, returning its tag."""
    closed = np.vstack((points, points[:1]))
    lengths = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))))
    spline = make_interp_spline(lengths / lengths[-1], closed, k=spline_order, bc_type="periodic")

    # Scipy repeats the first poles at the end of its coefficients and pads its
    # knots on both sides. Open Cascade takes each only once.
    pole_tags = [gmsh.model.occ.addPoint(*pole.tolist()) for pole in spline.c[: len(points)]]
    knots = spline.t[spline_order : len(spline.t) - spline_order].tolist()
    # Repeating the first tag is how gmsh is told the curve is periodic.
    curve = gmsh.model.occ.addBSpline([*pole_tags, pole_tags[0]], degree=spline_order, knots=knots, multiplicities=[1] * len(knots))
    return gmsh.model.occ.addWire([curve], checkClosed=True)


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
