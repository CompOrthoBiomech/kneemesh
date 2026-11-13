"""B-rep generation for the femoral cartilage. The femur is not implemented yet.

The femoral cartilage wraps the condyles, so it is a sheet over a cylinder
about the knee's flexion axis rather than over a plane (CylinderFrame). Its
domain, unrolled, is a U: the trochlea joins two condylar legs around the
intercondylar notch. For hexahedra the U is split like three half
butterflies glued along the trochlea: a core of five quadrilaterals, one per
leg and three across the trochlea divided where the legs and the notch meet
it, inside a ring of twelve blocks along the perimeter.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from loguru import logger
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import ConvexHull, KDTree

from kneemesh.brep import BRep
from kneemesh.cartilage import CartilageSides
from kneemesh.geo.sheet import (
    DEFAULT_MIN_THICKNESS_HEXAHEDRAL,
    DEFAULT_PERIMETER_RADIUS_HEXAHEDRAL,
    CartilageSheet,
    CylinderFrame,
    blocks_to_hex_brep,
    blocks_to_solid,
    check_unfolded,
    coons_block,
    fit_sheet,
    inset_loops,
    resample_open,
    ring_blocks,
    split_perimeter,
)

DEFAULT_RING_WIDTH = 0.5
"""Default width of the ring of blocks along the perimeter, as a fraction of the condylar legs' half-width."""

# Points along each side of a block, as in geo.sheet.
_PATCH_SAMPLES = 20
# Spacing, in physical units, of the points that measure distances from the perimeter and smooth the core's curved sides.
_CROSSING_STEP = 0.05
# Step, in physical units, between rows tried for the legs' ends and the trochlea's front.
_SCAN_STEP = 0.25
# The legs end where the inset is this fraction of its width at the notch's apex level.
_END_WIDTH = 0.5
# The trochlea's core stops where the inset is this fraction of its width at the B row.
_FRONT_WIDTH = 0.5
# Gaussian sigma smoothing the core's curved sides, as a fraction of the ring's width.
_SIDE_SMOOTHING = 0.25
# The notch must reach at least this fraction of the way across the domain from its mouth.
_NOTCH_DEPTH = 0.25
# Pixel size, in physical units, of the distance map that finds the inset.
_INSET_PIXEL = 0.1
# Order of the core's boundary vertices, counterclockwise from the left leg's outer end corner. The
# legs end at E, the trochlea's quadrilaterals start at the B row level with the notch and end at the F row.
_CORE_LOOP = ("E0", "E1", "B1", "B2", "E2", "E3", "B3", "F3", "F2", "F1", "F0", "B0")
# Each core quadrilateral's corners, counterclockwise from the corner at a = 0, b = 0.
_CORE_QUADS = (("E0", "E1", "B1", "B0"), ("B0", "B1", "F1", "F0"), ("B1", "B2", "F2", "F1"), ("B2", "B3", "F3", "F2"), ("E2", "E3", "B3", "B2"))


def fit_femoral_sheet(
    sides: CartilageSides,
    perimeter_radius: float = DEFAULT_PERIMETER_RADIUS_HEXAHEDRAL,
    perimeter_inset: float = 0.0,
    min_thickness: float = DEFAULT_MIN_THICKNESS_HEXAHEDRAL,
    hole_thickness: float = 0.0,
) -> CartilageSheet:
    """Fit the femoral cartilage as a sheet over the cylinder its bone side wraps.

    The arguments are fit_sheet's; see there. Heights are distances from the
    cylinder's axis, and the domain is the cylinder unrolled.

    Raises:
        ValueError: If a parameter is out of range, the sides wrap too far
            around to unroll, or they do not overlap enough.
    """
    return fit_sheet(sides, perimeter_radius, perimeter_inset, min_thickness, hole_thickness, CylinderFrame.fit(sides))


def femoral_sheet_to_brep(sheet: CartilageSheet, ring_width: float = DEFAULT_RING_WIDTH) -> BRep:
    """Build a solid of the femoral cartilage's sheet for tetrahedral meshing, over its block partition, cutting out its holes.

    Raises:
        ValueError: If the domain is not U-shaped or its partition folds.
    """
    return blocks_to_solid(sheet, femoral_blocks(sheet, ring_width))


def femoral_sheet_to_hex_blocks(sheet: CartilageSheet, ring_width: float = DEFAULT_RING_WIDTH) -> BRep:
    """Build the femoral cartilage's sheet as seventeen blocks for hexahedral meshing, ignoring holes.

    Raises:
        ValueError: If the domain is not U-shaped or its partition folds.
    """
    if sheet.holes:
        logger.warning("Hexahedral blocks cannot have holes; ignoring {} holes", len(sheet.holes))
    return blocks_to_hex_brep(sheet, femoral_blocks(sheet, ring_width))


def femoral_blocks(sheet: CartilageSheet, ring_width: float = DEFAULT_RING_WIDTH) -> list[np.ndarray]:
    """Partition the femoral cartilage's U-shaped domain into blocks: five core quadrilaterals and a ring of twelve.

    The notch is the deepest pocket between the perimeter and its convex
    hull. Rows run parallel to the hull edge across it (the lid), at
    heights w from the notch's apex toward the trochlea. The core's sides
    along the legs and the trochlea's flanks follow the outline a ring width
    inside the perimeter (the inset), smoothed, so the ring is about equally
    wide everywhere; the straight leg ends (E row), trochlear front (F row),
    and notch top (B row) keep the core's corners near right angles. The B
    row runs a ring width above the apex, its notch-side vertices above
    where the inset crosses the apex's level. Each leg ends at the first row,
    from two ring widths inside its tip, where the inset is half as wide as
    at the apex's level, and the trochlea's core at the first row, from two
    ring widths inside its front, half as wide as the B row. The F row is
    split in the B row's proportions, so the trochlea's quadrilaterals fan
    out over its dome. Each core boundary vertex's spoke leaves it along the
    bisector of its two sides' outward normals, and the ring between the
    perimeter and the core is relaxed as the butterfly's is (ring_blocks).

    Args:
        sheet: The femoral cartilage's sheet from fit_femoral_sheet.
        ring_width: Width of the ring as a fraction of the legs' half-width,
            the 95th percentile of the distance to the perimeter over them.

    Returns:
        (_PATCH_SAMPLES, _PATCH_SAMPLES, 2) grids of domain points indexed
        [b, a], the five core blocks (left leg, the trochlea's left, middle,
        and right, right leg) and then the ring's blocks counterclockwise from
        the left leg's end.

    Raises:
        ValueError: If ring_width is not in (0, 1), the domain is not U-shaped,
            or a block folds.
    """
    if not 0 < ring_width < 1:
        raise ValueError(f"ring_width must be in (0, 1), got {ring_width}")
    outline = _Outline(sheet.perimeter)
    apex, lid = outline.notch()
    # Rows run along u, parallel to the lid, at heights w toward the trochlea.
    u_axis = lid / np.linalg.norm(lid)
    w_axis = np.array([-u_axis[1], u_axis[0]])
    local = (sheet.perimeter - apex) @ np.column_stack((u_axis, w_axis))
    legs = local[:, 1] < 0
    half_width = float(np.percentile(outline.inside_distance(_area_samples(local, legs) @ np.vstack((u_axis, w_axis)) + apex), 95))
    inset = ring_width * half_width
    logger.debug("Femoral legs' half-width {:.3g}; ring width {:.3g}", half_width, inset)

    loops = inset_loops(sheet.perimeter, inset, _INSET_PIXEL)
    if len(loops) != 1:
        raise ValueError(f"The femoral cartilage's domain is too narrow somewhere for a ring {inset:.3g} wide. Try a smaller ring_width")
    inset_loop = _Loop((loops[0] - apex) @ np.column_stack((u_axis, w_axis)))

    def widest_row(levels: np.ndarray, accept: Callable[[list[_Crossing]], bool]) -> list[_Crossing]:
        """The crossings of the first of levels whose crossings accept takes."""
        for w in levels:
            crossings = inset_loop.crossings(float(w))
            if accept(crossings):
                return crossings
        raise ValueError("The femoral cartilage's domain is not U-shaped: a leg or the trochlea is too narrow for the ring")

    notch_level = inset_loop.crossings(0.0)
    if len(notch_level) != 4:
        raise ValueError("The femoral cartilage's domain is not U-shaped: the inset does not cross the notch's apex level four times")
    # The B row runs a ring width above the notch's apex, or just over the
    # inset's notch arc where a tilted apex lifts that arc higher.
    back = max(inset, max(point[1] for point in inset_loop.arc(notch_level[1], notch_level[2])) + _SCAN_STEP)
    outer = inset_loop.crossings(back)
    if len(outer) != 2:
        raise ValueError("The femoral cartilage's trochlea is not one piece across above the notch")
    # The legs end at the first row from two ring widths inside their tips where the
    # inset is at least _END_WIDTH of its width at the notch's apex level, which keeps
    # the core's corners out of the rounded ends without leaving a thick ring there.
    ends = []
    for sign, (first, last) in ((-1, notch_level[:2]), (1, notch_level[2:])):
        leg = local[legs & (sign * local[:, 0] > 0)]
        if len(leg) == 0:
            raise ValueError("The femoral cartilage's domain is not U-shaped: a condylar leg is missing")
        wide = _END_WIDTH * (last.u - first.u)

        def leg_end(crossings: list[_Crossing], sign: int = sign, wide: float = wide) -> bool:
            on_leg = [crossing for crossing in crossings if sign * crossing.u > 0]
            return len(on_leg) == 2 and on_leg[1].u - on_leg[0].u >= wide

        crossings = widest_row(np.arange(float(leg[:, 1].min()) + 2 * inset, 0.0, _SCAN_STEP), leg_end)
        ends.append([crossing for crossing in crossings if sign * crossing.u > 0])
    # The trochlea narrows toward its front like a dome. Its core stops at the
    # first row from two ring widths inside its front at least _FRONT_WIDTH as wide as the B row.
    back_width = outer[1].u - outer[0].u

    def front_row(crossings: list[_Crossing]) -> bool:
        return len(crossings) == 2 and crossings[1].u - crossings[0].u >= _FRONT_WIDTH * back_width

    front = widest_row(np.arange(float(local[:, 1].max()) - 2 * inset, back + inset, -_SCAN_STEP), front_row)
    level = front[0].w
    on_inset = {
        "E0": ends[0][0],
        "E1": ends[0][1],
        "E2": ends[1][0],
        "E3": ends[1][1],
        "B0": outer[0],
        "B3": outer[1],
        "F0": front[0],
        "F3": front[1],
    }
    notch_sides = {"N1": notch_level[1], "N2": notch_level[2]}
    vertices = {name: crossing.point for name, crossing in (on_inset | notch_sides).items()}
    vertices |= {
        "B1": np.array([notch_level[1].u, back]),
        "B2": np.array([notch_level[2].u, back]),
        # The front row is split in the same proportions as the B row, so the trochlea's quadrilaterals fan out.
        "F1": np.array([front[0].u + (front[1].u - front[0].u) * (notch_level[1].u - outer[0].u) / back_width, level]),
        "F2": np.array([front[0].u + (front[1].u - front[0].u) * (notch_level[2].u - outer[0].u) / back_width, level]),
    }
    domain = {name: apex + vertex @ np.vstack((u_axis, w_axis)) for name, vertex in vertices.items()}
    crossings_of = on_inset | notch_sides

    def to_domain(points: np.ndarray) -> np.ndarray:
        return apex + points @ np.vstack((u_axis, w_axis))

    # Core sides along the legs and the trochlea's flanks follow the inset; the
    # leg ends, the trochlea's front, the notch's top, and the inner sides are straight.
    curved = {
        ("E1", "B1"): np.vstack((inset_loop.arc(crossings_of["E1"], crossings_of["N1"]), vertices["B1"])),
        ("B2", "E2"): np.vstack((vertices["B2"], inset_loop.arc(crossings_of["N2"], crossings_of["E2"]))),
        ("E3", "B3"): inset_loop.arc(crossings_of["E3"], crossings_of["B3"]),
        ("B3", "F3"): inset_loop.arc(crossings_of["B3"], crossings_of["F3"]),
        ("F0", "B0"): inset_loop.arc(crossings_of["F0"], crossings_of["B0"]),
        ("B0", "E0"): inset_loop.arc(crossings_of["B0"], crossings_of["E0"]),
    }
    edges: dict[tuple[str, str], np.ndarray] = {}

    def side(start: str, end: str) -> np.ndarray:
        if (end, start) in edges:
            return edges[end, start][::-1]
        if (start, end) in curved:
            edges[start, end] = resample_open(_smooth(to_domain(curved[start, end]), _SIDE_SMOOTHING * inset), _PATCH_SAMPLES)
        elif (end, start) in curved:
            edges[end, start] = resample_open(_smooth(to_domain(curved[end, start]), _SIDE_SMOOTHING * inset), _PATCH_SAMPLES)
            return edges[end, start][::-1]
        else:
            samples = np.linspace(0.0, 1.0, _PATCH_SAMPLES)[:, np.newaxis]
            edges[start, end] = domain[start] + samples * (domain[end] - domain[start])
        return edges[start, end]

    core = [coons_block([side(a, b), side(d, c), side(a, d), side(b, c)]) for a, b, c, d in _CORE_QUADS]
    core_sides = [side(name, _CORE_LOOP[(k + 1) % len(_CORE_LOOP)]) for k, name in enumerate(_CORE_LOOP)]
    loop = [side[0] for side in core_sides]
    splits = outline.spoke_splits(loop, core_sides)
    blocks = [*core, *ring_blocks(split_perimeter(sheet.perimeter, splits), core_sides)]
    check_unfolded(blocks, "the femoral cartilage's perimeter is too concave for its ring. Try a larger perimeter_radius or another ring_width")
    return blocks


class _Outline:
    """A closed counterclockwise perimeter, with distances from it inside."""

    def __init__(self, perimeter: np.ndarray) -> None:
        self.perimeter: np.ndarray = perimeter
        closed = np.vstack((perimeter, perimeter[:1]))
        lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        dense = [
            closed[i] + np.linspace(0, 1, max(2, int(np.ceil(length / _CROSSING_STEP))), endpoint=False)[:, None] * (closed[i + 1] - closed[i])
            for i, length in enumerate(lengths)
        ]
        self._tree: KDTree = KDTree(np.vstack(dense))

    def inside_distance(self, points: np.ndarray) -> np.ndarray:
        """Distance of (N, 2) points from the perimeter, positive inside it and negative outside."""
        distance, _ = self._tree.query(points)
        return np.where(self.contains(points), distance, -distance)

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Whether each of (N, 2) points is inside the perimeter, by counting crossings of a ray along +s."""
        start, end = self.perimeter, np.roll(self.perimeter, -1, axis=0)
        x, y = points[:, 0, np.newaxis], points[:, 1, np.newaxis]
        straddles = (start[:, 1] > y) != (end[:, 1] > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            crossing = start[:, 0] + (y - start[:, 1]) * (end[:, 0] - start[:, 0]) / (end[:, 1] - start[:, 1])
        return np.count_nonzero(straddles & (x < crossing), axis=1) % 2 == 1

    def notch(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the apex of the deepest pocket between the perimeter and its convex hull, and the hull edge across it, counterclockwise."""
        hull = ConvexHull(self.perimeter).vertices
        # Hull vertices come counterclockwise; the perimeter between consecutive ones is a pocket.
        best = (0.0, 0.0, self.perimeter[0], self.perimeter[0])
        for first, second in zip(hull, np.roll(hull, -1), strict=True):
            pocket = self.perimeter[first : second + 1] if first < second else np.vstack((self.perimeter[first:], self.perimeter[: second + 1]))
            lid = self.perimeter[second] - self.perimeter[first]
            normal = np.array([-lid[1], lid[0]]) / np.linalg.norm(lid)
            depths = (pocket - self.perimeter[first]) @ normal
            deepest = int(np.argmax(depths))
            if depths[deepest] > best[0]:
                extent = float(np.ptp(self.perimeter @ normal))
                best = (float(depths[deepest]), extent, pocket[deepest], lid)
        if best[0] < _NOTCH_DEPTH * best[1]:
            raise ValueError(
                f"The femoral cartilage's domain is not U-shaped: its deepest notch is {best[0]:.3g} deep, "
                + f"under {_NOTCH_DEPTH} of its extent {best[1]:.3g} across the notch's mouth"
            )
        return best[2], best[3]

    def spoke_splits(self, loop: list[np.ndarray], sides: list[np.ndarray]) -> np.ndarray:
        """Return increasing perimeter indices where each counterclockwise core boundary vertex's spoke meets the perimeter.

        Each spoke leaves its vertex along the bisector of the outward normals of the vertex's two core sides
        there; side k runs from vertex k to vertex k + 1.

        Raises:
            ValueError: If the spokes do not meet the perimeter in the core's order.
        """
        count = len(self.perimeter)
        start, end = self.perimeter, np.roll(self.perimeter, -1, axis=0)
        indices = []
        for k, vertex in enumerate(loop):
            normals = []
            for a, b in ((sides[k - 1][-2], vertex), (vertex, sides[k][1])):
                tangent = (b - a) / np.linalg.norm(b - a)
                normals.append(np.array([tangent[1], -tangent[0]]))
            direction = normals[0] + normals[1]
            direction /= np.linalg.norm(direction)
            # Solve vertex + r * direction = start + q * (end - start) for every perimeter segment.
            segment = end - start
            denominator = direction[0] * segment[:, 1] - direction[1] * segment[:, 0]
            offset = start - vertex
            with np.errstate(divide="ignore", invalid="ignore"):
                r = (offset[:, 0] * segment[:, 1] - offset[:, 1] * segment[:, 0]) / denominator
                q = (offset[:, 0] * direction[1] - offset[:, 1] * direction[0]) / denominator
            hits = np.flatnonzero((r > 0) & (q >= 0) & (q < 1))
            if len(hits) == 0:
                raise ValueError("A spoke of the femoral cartilage's core does not meet the perimeter")
            nearest = hits[np.argmin(r[hits])]
            indices.append((int(nearest) + round(float(q[nearest]))) % count)
        splits = np.array(indices)
        splits = splits[0] + (splits - splits[0]) % count
        if np.any(np.diff(splits) <= 0):
            raise ValueError("The spokes of the femoral cartilage's core cross; its domain is too far from a U for this partition")
        return splits


@dataclass(frozen=True)
class _Crossing:
    """Where a row crosses a loop: the point, and the loop segment it lies on."""

    point: np.ndarray
    segment: int

    @property
    def u(self) -> float:
        return float(self.point[0])

    @property
    def w(self) -> float:
        return float(self.point[1])


class _Loop:
    """A closed counterclockwise loop of points in row coordinates (u, w)."""

    def __init__(self, points: np.ndarray) -> None:
        self.points: np.ndarray = points

    def crossings(self, w: float) -> list[_Crossing]:
        """Return where the row at height w crosses the loop, in increasing u."""
        start, end = self.points, np.roll(self.points, -1, axis=0)
        found = []
        for segment in np.flatnonzero((start[:, 1] >= w) != (end[:, 1] >= w)):
            fraction = float((w - start[segment, 1]) / (end[segment, 1] - start[segment, 1]))
            found.append(_Crossing(start[segment] + fraction * (end[segment] - start[segment]), int(segment)))
        return sorted(found, key=lambda crossing: crossing.u)

    def arc(self, first: _Crossing, last: _Crossing) -> np.ndarray:
        """Return the loop's points counterclockwise from one crossing to another, both included."""
        count = len(self.points)
        between = [(first.segment + 1 + offset) % count for offset in range((last.segment - first.segment) % count)]
        return np.vstack((first.point, self.points[between], last.point))


def _smooth(points: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth a polyline by a Gaussian of sigma along its length, keeping its ends and their tangents.

    Reflecting the polyline through each end before filtering keeps the
    ends fixed and straight runs straight, so a smoothed side still meets its
    neighbors at the vertices. It rounds kinks, such as where the legs' notch
    sides leave the inset, through which cubic patches would overshoot.
    """
    fine = resample_open(points, max(2, int(np.ceil(_polyline_length(points) / _CROSSING_STEP)) + 1))
    pad = min(len(fine) - 1, int(np.ceil(3 * sigma / _CROSSING_STEP)))
    padded = np.vstack((2 * fine[0] - fine[pad:0:-1], fine, 2 * fine[-1] - fine[-2 : -pad - 2 : -1]))
    smoothed = gaussian_filter1d(padded, sigma / _CROSSING_STEP, axis=0, mode="nearest")[pad : pad + len(fine)]
    smoothed[0], smoothed[-1] = fine[0], fine[-1]
    return smoothed


def _polyline_length(points: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def _area_samples(local: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a grid of points, in the same coordinates as local, filling the bounding box of local[mask] at _CROSSING_STEP * 10 spacing."""
    low, high = local[mask].min(axis=0), local[mask].max(axis=0)
    step = 10 * _CROSSING_STEP
    u, w = np.meshgrid(np.arange(low[0], high[0], step), np.arange(low[1], high[1], step))
    return np.column_stack((u.ravel(), w.ravel()))
