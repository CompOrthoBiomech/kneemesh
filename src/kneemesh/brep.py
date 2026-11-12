"""Generation of B-rep geometry."""

import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Literal

import gmsh
import numpy as np
from loguru import logger
from scipy.interpolate import BSpline, make_interp_spline
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData, vtkPolyLine

from meniscus2brep.vol import ContourSlice

_MAX_SPLINE_ORDER = 25  # Open Cascade's limit on B-spline degree.
_DEFAULT_CAP_OFFSET_RADII = 0.25
_PIECE_SAMPLES = 20
_MIN_HEX_POINTS = 8
# Inner block corners lie this fraction of the way from the section's centroid to the split points.
_INNER_BLOCK_SCALE = 0.5
# Width, as a fraction of the contour's length, of the Gaussian weighting turns near butterfly splits.
_SPLIT_TURN_WIDTH = 1 / 16


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
    _require_loftable(slices, spline_order)

    with gmsh_session():
        splines = [_periodic_spline(_project_onto_plane(contour), spline_order) for contour in slices]
        wires = [_add_contour_wire(spline, len(contour.points)) for spline, contour in zip(splines, slices)]
        # Open Cascade only honors a loft degree of 1 for a ruled surface.
        # Open Cascade's default C2 continuity makes the loft overshoot in its end
        # spans where real sections taper quickly toward the horns, folding the
        # surface so gmsh cannot mesh it. C1 lets it follow the sections.
        loft = gmsh.model.occ.addThruSections(
            wires, makeSolid=False, makeRuled=spline_order == 1, maxDegree=spline_order, continuity="C1"
        )
        ends = [(slices[0], splines[0], wires[0], -1.0), (slices[-1], splines[-1], wires[-1], 1.0)]
        caps = None
        if cap == "planar":
            caps = [gmsh.model.occ.addPlaneSurface([wire]) for _, _, wire, _ in ends]
        elif cap == "smooth":
            caps = [_add_smooth_cap(spline, contour, sign * contour.normal, cap_offset) for contour, spline, _, sign in ends]
        if cap != "none" and caps is not None:
            # addThruSections' makeSolid returns a volume with no end faces.
            shell = gmsh.model.occ.addSurfaceLoop([*(tag for _, tag in loft), *caps], sewing=True)
            gmsh.model.occ.addVolume([shell])
        gmsh.model.occ.synchronize()
        return _export_brep()


def contours_to_hex_blocks(slices: list[ContourSlice], spline_order: int = 3) -> BRep:
    """Build a planar-capped solid from contour slices as five blocks for hexahedral meshing.

    The section is split with a butterfly (O-grid) partition: an inner
    quadrilateral surrounded by four blocks, one for each of four arcs of the
    boundary. Every boundary point meets a spoke running inward, so the cells
    along the boundary stay close to square even at its sharp corners. A
    partition that puts block corners at those corners instead gives cells
    there the corner's angle, or nearly 180 degrees where smoothing rounds it.

    The four split points divide each contour into arcs of equal length,
    rotated to keep them as far as possible from the sharpest turns averaged
    over all slices. Straight spokes run from each split point to a corner of
    the inner quadrilateral, partway toward the section's centroid.

    The blocks' faces are B-spline surfaces interpolating the slices with a
    knot vector shared by every face along the curve, so neighboring faces
    meet exactly. They differ slightly from contours_to_brep's loft.

    Args:
        slices: Contours from slice_isocontour, in order along the curve, all
            with the same number of points.
        spline_order: Polynomial degree of the section curves. The degree along
            the curve is at most 3.

    Returns:
        A solid of five blocks, for hexahedral_mesh.

    Raises:
        ValueError: If there are fewer than 2 slices, spline_order is outside
            1 to 25, a contour has too few points for spline_order or fewer
            than 8, or contours have unequal point counts.
    """
    _require_loftable(slices, spline_order)
    if len({len(contour.points) for contour in slices}) > 1:
        raise ValueError("Hexahedral blocks need the same number of points on every contour")
    if len(slices[0].points) < _MIN_HEX_POINTS:
        raise ValueError(f"Hexahedral blocks need at least {_MIN_HEX_POINTS} points on each contour")

    splits = _butterfly_split_indices(slices)
    samples = np.linspace(0.0, 1.0, _PIECE_SAMPLES)
    # pieces[slice][arc] holds points along the four boundary arcs; arc i runs from split i to split i + 1.
    pieces, inner_corners = [], []
    for contour in slices:
        points = np.roll(_project_onto_plane(contour), -int(splits[0]), axis=0)
        parameters = np.interp(splits - splits[0], np.arange(len(points) + 1), _chord_parameters(points))
        curves = _split_periodic_spline(_periodic_spline(points, spline_order), parameters)
        slice_pieces = [curve(curve.t[0] + samples * (curve.t[-1] - curve.t[0])) for curve in curves]
        centroid = _section_centroid(contour)
        pieces.append(slice_pieces)
        inner_corners.append([centroid + _INNER_BLOCK_SCALE * (piece[0] - centroid) for piece in slice_pieces])

    centers = np.array([contour.center for contour in slices])
    along = _open_chord_parameters(centers)
    along_degree = min(spline_order, 3, len(slices) - 1)
    line_knots = np.array([0.0, 0.0, 1.0, 1.0])
    arc_rows = [[_clamped_interpolation(samples, slice_pieces[arc], spline_order) for slice_pieces in pieces] for arc in range(4)]
    # Spoke i runs from split i to inner corner i, and inner side i from inner corner i to i + 1.
    spoke_rows = [[(line_knots, np.array([slice_pieces[i][0], corners[i]])) for slice_pieces, corners in zip(pieces, inner_corners)] for i in range(4)]
    inner_rows = [[(line_knots, np.array([corners[i], corners[(i + 1) % 4]])) for corners in inner_corners] for i in range(4)]

    with gmsh_session():
        arcs = [_add_swept_surface(rows, spline_order, along, along_degree) for rows in arc_rows]
        spokes = [_add_swept_surface(rows, 1, along, along_degree) for rows in spoke_rows]
        inner = [_add_swept_surface(rows, 1, along, along_degree) for rows in inner_rows]
        caps = []
        for end in (0, -1):
            arc_curves = [_add_bspline_curve(*arc_rows[i][end], spline_order) for i in range(4)]
            spoke_curves = [_add_bspline_curve(*spoke_rows[i][end], 1) for i in range(4)]
            inner_curves = [_add_bspline_curve(*inner_rows[i][end], 1) for i in range(4)]
            caps.append(
                [
                    gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop(sides)])
                    for sides in _butterfly_block_sides(arc_curves, spoke_curves, inner_curves)
                ]
            )
        blocks = []
        for block, sides in enumerate(_butterfly_block_sides(arcs, spokes, inner)):
            faces = [*sides, caps[0][block], caps[1][block]]
            blocks.append((3, gmsh.model.occ.addVolume([gmsh.model.occ.addSurfaceLoop(faces, sewing=True)])))
        # Sewing copies the shared spoke and inner faces into each block. They
        # are identical surfaces, so fragmenting merges them back into one each.
        gmsh.model.occ.fragment(blocks[:1], blocks[1:])
        gmsh.model.occ.synchronize()
        return _export_brep()


def _require_loftable(slices: list[ContourSlice], spline_order: int) -> None:
    if len(slices) < 2:
        raise ValueError("Lofting needs at least 2 contour slices")
    if not 1 <= spline_order <= _MAX_SPLINE_ORDER:
        raise ValueError(f"spline_order must be between 1 and {_MAX_SPLINE_ORDER}, got {spline_order}")
    if any(len(contour.points) <= spline_order for contour in slices):
        raise ValueError(f"Contours need more than {spline_order} points for spline_order {spline_order}")


def _export_brep() -> BRep:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "shape.brep"
        gmsh.write(str(path))
        text = path.read_text(encoding="utf-8")
    counts = [len(gmsh.model.getEntities(dim)) for dim in range(4)]
    logger.debug("B-rep has {} vertices, {} edges, {} faces, and {} volumes ({} kB)", *counts, len(text) // 1024)
    return BRep(text)


def _butterfly_block_sides(arcs: list[int], spokes: list[int], inner: list[int]) -> list[list[int]]:
    """Return the four sides of each block: the inner block, then the outer block on each boundary arc."""
    outer = [[arcs[i], spokes[(i + 1) % 4], inner[i], spokes[i]] for i in range(4)]
    return [list(inner), *outer]


def _project_onto_plane(contour: ContourSlice) -> np.ndarray:
    """Return a contour's points exactly in its plane.

    VTK's single-precision output leaves them about 3e-7 off the plane, which
    is outside Open Cascade's tolerance for building a planar end face.
    """
    heights = (contour.points - contour.center) @ contour.normal
    return contour.points - np.outer(heights, contour.normal)


def _chord_parameters(points: np.ndarray) -> np.ndarray:
    """Return parameters from 0 to 1 proportional to chord length around a closed contour, ending at the repeated first point."""
    closed = np.vstack((points, points[:1]))
    lengths = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))))
    return lengths / lengths[-1]


def _periodic_spline(points: np.ndarray, spline_order: int) -> BSpline:
    """Return a periodic B-spline through points, over parameters 0 to 1 proportional to chord length."""
    return make_interp_spline(_chord_parameters(points), np.vstack((points, points[:1])), k=spline_order, bc_type="periodic")


def _open_chord_parameters(points: np.ndarray) -> np.ndarray:
    """Return parameters from 0 to 1 proportional to chord length along an open sequence of points."""
    lengths = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    return lengths / lengths[-1]


def _butterfly_split_indices(slices: list[ContourSlice]) -> np.ndarray:
    """Return four sorted point indices splitting every contour into the butterfly's boundary arcs.

    The splits divide the contour into arcs of equal length, by chord length
    averaged over all slices. Their rotation minimizes the turning angle,
    averaged over all slices, near them: each point's turn is weighted by a
    Gaussian of its distance from the nearest split. The first split is a
    point index and the others may fall between points.
    """
    turns = np.mean([_turning_angles(contour.points) for contour in slices], axis=0)
    positions = np.mean([_chord_parameters(contour.points) for contour in slices], axis=0)
    count = len(turns)
    offsets = np.arange(4) / 4

    def cost(start: int) -> float:
        distances = np.abs(positions[:count, None] - (positions[start] + offsets)) % 1.0
        distances = np.minimum(distances, 1.0 - distances)
        return float(np.sum(turns[:, None] * np.exp(-0.5 * (distances / _SPLIT_TURN_WIDTH) ** 2)))

    start = min(range(count), key=cost)
    # Unwrap one turn past the end so splits beyond the last point map to indices above count.
    unwrapped = np.concatenate((positions[:count], positions[:count] + 1.0, [2.0]))
    splits = np.interp(positions[start] + offsets, unwrapped, np.arange(2 * count + 1))
    logger.debug("Butterfly splits at contour indices {}", np.round(splits % count, 2).tolist())
    return splits


def _section_centroid(contour: ContourSlice) -> np.ndarray:
    """Return the area centroid of a contour's polygon, in physical coordinates."""
    offsets = contour.points - contour.center
    u, v = offsets @ contour.u_axis, offsets @ contour.v_axis
    cross = u * np.roll(v, -1) - np.roll(u, -1) * v
    area = cross.sum() / 2
    centroid_u = np.sum((u + np.roll(u, -1)) * cross) / (6 * area)
    centroid_v = np.sum((v + np.roll(v, -1)) * cross) / (6 * area)
    return contour.center + centroid_u * contour.u_axis + centroid_v * contour.v_axis


def _clamped_interpolation(parameters: np.ndarray, points: np.ndarray, degree: int) -> tuple[np.ndarray, np.ndarray]:
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


def _add_bspline_curve(knots: np.ndarray, poles: np.ndarray, degree: int) -> int:
    unique_knots, multiplicities = np.unique(knots, return_counts=True)
    tags = [gmsh.model.occ.addPoint(*pole.tolist()) for pole in poles]
    return gmsh.model.occ.addBSpline(tags, degree=degree, knots=unique_knots.tolist(), multiplicities=multiplicities.tolist())


def _add_swept_surface(rows: list[tuple[np.ndarray, np.ndarray]], degree: int, along: np.ndarray, along_degree: int) -> int:
    """Add a B-spline surface through section curves sharing one knot vector, returning its tag.

    Each column of poles is interpolated along the curve at the same
    parameters, so a surface's edge depends only on that edge's points and
    matches its neighbor's exactly.
    """
    knots = rows[0][0]
    grid = np.array([poles for _, poles in rows])
    columns = [_clamped_interpolation(along, grid[:, column], along_degree) for column in range(grid.shape[1])]
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


def _turning_angles(points: np.ndarray) -> np.ndarray:
    incoming = points - np.roll(points, 1, axis=0)
    outgoing = np.roll(points, -1, axis=0) - points
    cosines = np.sum(incoming * outgoing, axis=1) / (np.linalg.norm(incoming, axis=1) * np.linalg.norm(outgoing, axis=1))
    return np.arccos(np.clip(cosines, -1.0, 1.0))


def _periodic_poles_and_knots(spline: BSpline, point_count: int) -> tuple[np.ndarray, list[float]]:
    """Return a periodic spline's poles and knots as Open Cascade takes them."""
    # Scipy repeats the first poles at the end of its coefficients and pads its
    # knots on both sides. Open Cascade takes each only once.
    return spline.c[:point_count], spline.t[spline.k : len(spline.t) - spline.k].tolist()


def _add_contour_wire(spline: BSpline, point_count: int) -> int:
    """Add a closed wire of one periodic B-spline, returning its tag."""
    poles, knots = _periodic_poles_and_knots(spline, point_count)
    # Repeating the first tag is how gmsh is told the curve is periodic.
    curve = gmsh.model.occ.addBSpline(_closed_tags(poles), degree=spline.k, knots=knots, multiplicities=[1] * len(knots))
    return gmsh.model.occ.addWire([curve], checkClosed=True)


def _split_periodic_spline(spline: BSpline, parameters: np.ndarray) -> list[BSpline]:
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


def _closed_tags(poles: np.ndarray) -> list[int]:
    tags = [gmsh.model.occ.addPoint(*pole.tolist()) for pole in poles]
    return [*tags, tags[0]]


def _add_smooth_cap(
    spline: BSpline,
    contour: ContourSlice,
    outward: np.ndarray,
    offset: float | None,
) -> int:
    """Add a dome from a contour to one point, returning its surface tag.

    Each meridian is a quadratic Bezier from the contour straight out along the
    curve, then in to the apex, so the dome is convex and meets the apex flat.
    """
    poles, knots = _periodic_poles_and_knots(spline, len(contour.points))
    height = _DEFAULT_CAP_OFFSET_RADII * float(np.sqrt(contour.area / np.pi)) if offset is None else offset
    lift = height * outward
    apex = gmsh.model.occ.addPoint(*(contour.center + lift).tolist())
    rows = [*_closed_tags(poles), *_closed_tags(poles + lift), *[apex] * (len(poles) + 1)]
    return gmsh.model.occ.addBSplineSurface(
        rows,
        len(poles) + 1,
        degreeU=spline.k,
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
