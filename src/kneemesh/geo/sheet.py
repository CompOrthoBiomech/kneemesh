"""B-rep generation for cartilage sheets, the lateral and medial tibial and the patellar cartilage.

Each sheet is a pair of height fields over the plane normal to its thinnest
principal direction: its articular side above and the side against its bone
below, within one smooth perimeter in that plane. Walls along the normal
join the two sides at the perimeter.
"""

from dataclasses import dataclass
from itertools import pairwise

import gmsh
import numpy as np
import SimpleITK as sitk
from loguru import logger
from scipy.interpolate import RectBivariateSpline, griddata
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.spatial import KDTree
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkFiltersCore import vtkFlyingEdges2D, vtkStripper

from kneemesh.brep import (
    BRep,
    add_closed_wire,
    add_swept_surface,
    butterfly_split_indices,
    chord_parameters,
    clamped_interpolation,
    export_brep,
    gmsh_session,
    periodic_spline,
    resample_closed,
    split_periodic_spline,
)
from kneemesh.cartilage import CartilageSides

DEFAULT_PERIMETER_RADIUS_HEXAHEDRAL = 3.0
"""Default smallest radius of curvature, in physical units, of the perimeter of a sheet built for hexahedra."""

DEFAULT_PERIMETER_RADIUS_TETRAHEDRAL = 1.0
"""Default smallest radius of curvature, in physical units, of the perimeter and hole outlines of a sheet built for tetrahedra."""

DEFAULT_MIN_THICKNESS_HEXAHEDRAL = 0.2
"""Default minimum thickness of a sheet built for hexahedra, as a fraction of its mean thickness."""

DEFAULT_MIN_THICKNESS_TETRAHEDRAL = 0.1
"""Default minimum thickness of a sheet built for tetrahedra, as a fraction of its mean thickness."""

DEFAULT_HOLE_THICKNESS = 0.05
"""Default thickness, as a fraction of the mean, below which a sheet built for tetrahedra gets a hole."""

# Points around the perimeter and hole outlines.
_OUTLINE_POINTS = 400
# Gaps between surface points closed when finding each side's footprint, in median point spacings.
_FOOTPRINT_CLOSING_SPACINGS = 4.0
# Height grid pixels per median surface point spacing.
_PIXELS_PER_SPACING = 2.0
# Gaussian sigma smoothing the rounded outlines' waviness, as a fraction of perimeter_radius.
_OUTLINE_SMOOTHING = 0.5
# Gaussian sigma, in pixels, smoothing the height grids' linear interpolation.
_HEIGHT_SMOOTHING_PIXELS = 2.0
# Width of the smooth maximum that thickens thin regions, as a fraction of the minimum thickness.
_THICKENING_BLEND = 0.5
# Points along each side of a block's patches.
_PATCH_SAMPLES = 20
# Inner block corners lie this fraction of the way from the perimeter's centroid to the split points.
_INNER_BLOCK_SCALE = 0.5
_SPLINE_ORDER = 3
# Jacobi iterations, relaxation, and convergence tolerance, relative to the perimeter's extent, of the Winslow ring grid.
_WINSLOW_ITERATIONS = 5000
_WINSLOW_RELAXATION = 0.5
_WINSLOW_TOLERANCE = 1e-7


@dataclass(frozen=True, eq=False)
class CartilageSheet:
    """A cartilage sheet as heights of its two sides over a plane, within a perimeter.

    Plane coordinates (s, t) and height z give the physical point
    origin + s * axes[0] + t * axes[1] + z * axes[2].

    Attributes:
        origin: Physical origin of the plane.
        axes: (3, 3) right-handed orthonormal rows: the plane's s and t axes,
            and its normal, which points from the bone toward the articular side.
        perimeter: (N, 2) plane points around the sheet, counterclockwise about
            the normal, without the first repeated.
        holes: (M, 2) plane points around each hole, counterclockwise.
        articular: Height of the articular side over the plane.
        bone: Height of the side against the bone, already lowered where the
            sheet was thinner than its minimum thickness.
        mean_thickness: Mean thickness along the normal within the perimeter,
            before thickening.
        min_thickness: The thickness thin regions were raised to.
    """

    origin: np.ndarray
    axes: np.ndarray
    perimeter: np.ndarray
    holes: list[np.ndarray]
    articular: RectBivariateSpline
    bone: RectBivariateSpline
    mean_thickness: float
    min_thickness: float

    def points(self, plane_points: np.ndarray, side: str) -> np.ndarray:
        """Return the physical points on a side, "articular" or "bone", over (..., 2) plane points."""
        heights = self.articular if side == "articular" else self.bone
        flat = plane_points.reshape(-1, 2)
        z = heights.ev(flat[:, 0], flat[:, 1])
        physical = self.origin + flat @ self.axes[:2] + np.outer(z, self.axes[2])
        return physical.reshape(*plane_points.shape[:-1], 3)

    @property
    def area(self) -> float:
        """The plane area inside the perimeter, less the holes."""
        return _signed_area(self.perimeter) - sum(_signed_area(hole) for hole in self.holes)


def fit_sheet(
    sides: CartilageSides,
    perimeter_radius: float = DEFAULT_PERIMETER_RADIUS_HEXAHEDRAL,
    perimeter_inset: float = 0.0,
    min_thickness: float = DEFAULT_MIN_THICKNESS_HEXAHEDRAL,
    hole_thickness: float = 0.0,
) -> CartilageSheet:
    """Fit a cartilage sheet to the two sides of a cartilage surface.

    The plane is normal to the thinnest principal direction of the side
    points. Each side's footprint is where its points project onto the plane,
    with gaps of a few point spacings closed, and the sheet covers where both
    footprints overlap, so the steep perimeter between the sides is cut off.
    The overlap is moved perimeter_inset inward and rounded by rolling a disk
    of radius perimeter_radius around it, inside and out: a morphological
    closing then opening, from exact distance maps. Its outline then curves
    no more tightly than that radius anywhere, concave or convex, which
    removes the jagged edge and the notches and spurs narrower than the
    elements that must follow it. A Gaussian of half that radius then smooths
    the outline's remaining waviness. Holes in the overlap are filled unless
    they become holes.

    Heights are interpolated linearly between each side's points on a grid,
    smoothed lightly, and fitted with bicubic splines. Where the thickness
    falls below min_thickness of the mean, the bone side is lowered to that
    thickness by a smooth maximum, leaving the articular side alone.

    Args:
        sides: Points of the cartilage's articular and bone sides from classify_sides.
        perimeter_radius: Smallest radius of curvature of the perimeter and
            hole outlines, in physical units. Keep it above about half the
            elements' size along the perimeter, or elements across a tight
            notch can invert.
        perimeter_inset: How far inside the sides' overlap to put the
            perimeter, in physical units, cutting off more of the thin rim.
        min_thickness: Minimum thickness as a fraction of the mean.
        hole_thickness: Thickness as a fraction of the mean below which the
            sheet gets a hole, smaller than min_thickness. Regions of the
            overlap with no cartilage become holes too. Only regions whose
            outlines lie at least perimeter_radius inside the perimeter
            become holes; others are thickened. Holes are rounded like the
            perimeter, so those narrower than perimeter_radius vanish. 0
            makes no holes.

    Returns:
        The fitted sheet.

    Raises:
        ValueError: If a parameter is out of range, or the sides do not overlap
            enough to leave a region inside the perimeter.
    """
    if perimeter_radius <= 0:
        raise ValueError(f"perimeter_radius must be positive, got {perimeter_radius}")
    if perimeter_inset < 0:
        raise ValueError(f"perimeter_inset must not be negative, got {perimeter_inset}")
    if not 0 < min_thickness < 1:
        raise ValueError(f"min_thickness must be in (0, 1), got {min_thickness}")
    if not 0 <= hole_thickness < min_thickness:
        raise ValueError(f"hole_thickness must be at least 0 and below min_thickness {min_thickness}, got {hole_thickness}")

    origin, axes = _sheet_frame(sides)
    articular_plane, articular_z = _plane_coordinates(sides.articular, origin, axes)
    bone_plane, bone_z = _plane_coordinates(sides.bone, origin, axes)
    spacing = float(np.median(KDTree(sides.articular).query(sides.articular, k=2)[0][:, 1]))
    closing = _FOOTPRINT_CLOSING_SPACINGS * spacing
    grid = _Grid.around(np.vstack((articular_plane, bone_plane)), spacing / _PIXELS_PER_SPACING, closing + 2 * perimeter_radius)

    # Pixels near points of both sides, shrunk back by the same distance, are
    # covered: gaps narrower than the closing distance between points close.
    near_both = _largest_component(grid.near(articular_plane, closing) & grid.near(bone_plane, closing))
    covered = grid.distance(near_both) <= -closing
    filled = sitk.GetArrayFromImage(sitk.BinaryFillhole(sitk.GetImageFromArray(covered.astype(np.uint8)))).astype(bool)
    region = grid.rounded(grid.distance(filled) <= -perimeter_inset, perimeter_radius)
    if not region.any():
        raise ValueError("The cartilage's sides do not overlap enough to leave a sheet inside the perimeter")
    outline = grid.smoothed_distance(region, _OUTLINE_SMOOTHING * perimeter_radius)
    inside = outline <= 0.0
    perimeter = _ccw(resample_closed(max(grid.contours(outline, 0.0), key=_loop_length), _OUTLINE_POINTS))

    articular = grid.heights(articular_plane, articular_z)
    bone = grid.heights(bone_plane, bone_z)
    thickness = articular - bone
    mean_thickness = float(thickness[inside].mean())
    minimum = min_thickness * mean_thickness
    logger.debug(
        "Sheet thickness within the perimeter: mean {:.3g}, minimum {:.3g}; {:.1%} of its area is below the minimum {:.3g}",
        mean_thickness,
        float(thickness[inside].min()),
        float(np.mean(thickness[inside] < minimum)),
        minimum,
    )

    holes = []
    if hole_thickness > 0:
        thin = ((thickness < hole_thickness * mean_thickness) | (filled & ~covered)) & inside
        clear_of_perimeter = outline <= -perimeter_radius
        for thin_region in _components(thin):
            hole = grid.rounded(thin_region, perimeter_radius)
            if not hole.any():
                continue
            hole_outline = grid.smoothed_distance(hole, _OUTLINE_SMOOTHING * perimeter_radius)
            loops = grid.contours(hole_outline, 0.0)
            if not loops or not np.all(clear_of_perimeter[hole_outline <= 0.0]):
                continue
            holes.append(_ccw(resample_closed(max(loops, key=_loop_length), _OUTLINE_POINTS)))
        logger.debug("Cutting {} holes thinner than {:.3g}", len(holes), hole_thickness * mean_thickness)

    blend = _THICKENING_BLEND * minimum
    thickened = 0.5 * (thickness + minimum + np.sqrt((thickness - minimum) ** 2 + blend**2))
    sheet = CartilageSheet(origin, axes, perimeter, holes, grid.spline(articular), grid.spline(articular - thickened), mean_thickness, minimum)
    logger.debug("Fitted a sheet with perimeter length {:.4g} and {} holes", _loop_length(perimeter), len(holes))
    return sheet


def sheet_to_brep(sheet: CartilageSheet) -> BRep:
    """Build a solid of a cartilage sheet for tetrahedral meshing.

    The two sides are B-spline patches over the same butterfly partition of
    the perimeter that sheet_to_hex_blocks uses, joined by walls along the
    normal at the perimeter, and sewn into one solid. Each hole is cut out
    with a prism along the normal.

    Raises:
        ValueError: If the butterfly partition folds.
    """
    blocks = _butterfly_blocks(sheet)
    with gmsh_session():
        faces = []
        for grid in blocks:
            faces.extend(_add_patch(sheet.points(grid, side)) for side in ("articular", "bone"))
        for grid in blocks[1:]:
            # Row 0 of each outer block runs along its perimeter arc.
            faces.append(_add_wall(sheet.points(grid[0], "articular"), sheet.points(grid[0], "bone")))
        shell = gmsh.model.occ.addSurfaceLoop(faces, sewing=True)
        solid = [(3, gmsh.model.occ.addVolume([shell]))]
        if sheet.holes:
            prisms = [_add_prism(sheet, hole) for hole in sheet.holes]
            solid, _ = gmsh.model.occ.cut(solid, prisms)
        gmsh.model.occ.synchronize()
        return export_brep()


def sheet_to_hex_blocks(sheet: CartilageSheet) -> BRep:
    """Build a solid of a cartilage sheet as five blocks for hexahedral meshing.

    The perimeter is split with a butterfly partition, as the meniscus's
    sections are: an inner quadrilateral surrounded by four blocks, one on
    each of four equal-length arcs of the perimeter. The splits are rotated
    away from the perimeter's sharpest convex turns and toward its concave
    ones. The inner quadrilateral's corners lie halfway from the perimeter's
    centroid to the split points, and the spokes from the split points to
    them bend around concavities of the perimeter. Each block's two sides are
    B-spline patches through the heights over a grid of plane points. Walls
    along the normal stand on every block side, and the blocks are
    fragmented so neighbors share their walls. Holes are ignored.

    Raises:
        ValueError: If the butterfly partition folds, where the perimeter is
            too concave for its blocks.
    """
    if sheet.holes:
        logger.warning("Hexahedral blocks cannot have holes; ignoring {} holes", len(sheet.holes))
    blocks = _butterfly_blocks(sheet)
    with gmsh_session():
        # Wall i of the inner block is the inner side of outer block i, shared with it.
        inner_walls = [_add_wall(sheet.points(grid[-1], "articular"), sheet.points(grid[-1], "bone")) for grid in blocks[1:]]
        spoke_walls = [_add_wall(sheet.points(grid[:, 0], "articular"), sheet.points(grid[:, 0], "bone")) for grid in blocks[1:]]
        volumes = []
        inner = blocks[0]
        faces = [_add_patch(sheet.points(inner, "articular")), _add_patch(sheet.points(inner, "bone")), *inner_walls]
        volumes.append((3, gmsh.model.occ.addVolume([gmsh.model.occ.addSurfaceLoop(faces, sewing=True)])))
        for index, grid in enumerate(blocks[1:]):
            arc_wall = _add_wall(sheet.points(grid[0], "articular"), sheet.points(grid[0], "bone"))
            faces = [
                _add_patch(sheet.points(grid, "articular")),
                _add_patch(sheet.points(grid, "bone")),
                arc_wall,
                inner_walls[index],
                spoke_walls[index],
                spoke_walls[(index + 1) % 4],
            ]
            volumes.append((3, gmsh.model.occ.addVolume([gmsh.model.occ.addSurfaceLoop(faces, sewing=True)])))
        # Sewing copies the shared walls into each block. They are identical
        # surfaces, so fragmenting merges them back into one each.
        gmsh.model.occ.fragment(volumes[:1], volumes[1:])
        gmsh.model.occ.synchronize()
        return export_brep()


@dataclass(frozen=True)
class _Grid:
    """A pixel grid over the sheet's plane, with arrays indexed [t, s]."""

    origin: np.ndarray
    pixel: float
    shape: tuple[int, int]

    @classmethod
    def around(cls, points: np.ndarray, pixel: float, margin: float) -> "_Grid":
        low = points.min(axis=0) - margin
        high = points.max(axis=0) + margin
        counts = np.ceil((high - low) / pixel).astype(int) + 1
        return cls(low, pixel, (int(counts[1]), int(counts[0])))

    @property
    def s(self) -> np.ndarray:
        return self.origin[0] + self.pixel * np.arange(self.shape[1])

    @property
    def t(self) -> np.ndarray:
        return self.origin[1] + self.pixel * np.arange(self.shape[0])

    def centers(self) -> np.ndarray:
        s, t = np.meshgrid(self.s, self.t)
        return np.column_stack((s.ravel(), t.ravel()))

    def near(self, points: np.ndarray, distance: float) -> np.ndarray:
        """Pixels within distance of any of points."""
        found, _ = KDTree(points).query(self.centers(), distance_upper_bound=distance)
        return np.isfinite(found).reshape(self.shape)

    def heights(self, points: np.ndarray, heights: np.ndarray) -> np.ndarray:
        """Heights interpolated linearly between points, nearest beyond them, then smoothed lightly."""
        centers = self.centers()
        values = griddata(points, heights, centers, method="linear")
        missing = np.isnan(values)
        values[missing] = griddata(points, heights, centers[missing], method="nearest")
        return gaussian_filter(values.reshape(self.shape), _HEIGHT_SMOOTHING_PIXELS, mode="nearest")

    def distance(self, mask: np.ndarray) -> np.ndarray:
        """Signed distance to a mask's outline in physical units, negative inside."""
        image = sitk.GetImageFromArray(mask.astype(np.uint8))
        image.SetSpacing((self.pixel, self.pixel))
        # One map alone is zero on boundary pixel centers; averaging with the inverse's centers it on their edges.
        inside = sitk.SignedMaurerDistanceMap(image, insideIsPositive=False, squaredDistance=False, useImageSpacing=True)
        outside = sitk.SignedMaurerDistanceMap(sitk.BinaryNot(image), insideIsPositive=False, squaredDistance=False, useImageSpacing=True)
        return sitk.GetArrayFromImage((inside - outside) * 0.5)

    def smoothed_distance(self, mask: np.ndarray, sigma: float) -> np.ndarray:
        """Signed distance to a mask's outline, smoothed with a Gaussian of sigma in physical units."""
        return gaussian_filter(self.distance(mask), sigma / self.pixel, mode="nearest")

    def rounded(self, mask: np.ndarray, radius: float) -> np.ndarray:
        """A mask closed and then opened by a disk of radius, so its outline curves no more tightly than radius."""
        closed = self.distance(self.distance(mask) <= radius) <= -radius
        return self.distance(self.distance(closed) <= -radius) <= radius

    def contours(self, field: np.ndarray, level: float) -> list[np.ndarray]:
        """Closed loops of plane points where field crosses level."""
        image = vtkImageData()
        image.SetDimensions(self.shape[1], self.shape[0], 1)
        image.SetSpacing(self.pixel, self.pixel, 1.0)
        image.SetOrigin(float(self.origin[0]), float(self.origin[1]), 0.0)
        image.GetPointData().SetScalars(numpy_to_vtk(np.ascontiguousarray(field, dtype=float).ravel(), deep=True))
        contour = vtkFlyingEdges2D()
        contour.SetInputData(image)
        contour.SetValue(0, level)
        stripper = vtkStripper()
        stripper.SetInputConnection(contour.GetOutputPort())
        stripper.JoinContiguousSegmentsOn()
        stripper.SetMaximumLength(field.size)
        stripper.Update()
        output = stripper.GetOutput()
        if output.GetNumberOfPoints() == 0:
            return []
        points = vtk_to_numpy(output.GetPoints().GetData())[:, :2].astype(float)
        lines = output.GetLines()
        connectivity = vtk_to_numpy(lines.GetConnectivityArray())
        offsets = vtk_to_numpy(lines.GetOffsetsArray())
        loops = []
        for start, end in pairwise(offsets):
            ids = connectivity[start:end]
            if len(ids) > 3 and ids[0] == ids[-1]:
                loops.append(points[ids[:-1]])
        return loops

    def spline(self, values: np.ndarray) -> RectBivariateSpline:
        return RectBivariateSpline(self.s, self.t, values.T, kx=3, ky=3, s=0)


def _sheet_frame(sides: CartilageSides) -> tuple[np.ndarray, np.ndarray]:
    """Return the plane's origin and axes: s and t along the principal directions, and the normal from bone to articular side."""
    points = np.vstack((sides.articular, sides.bone))
    origin = points.mean(axis=0)
    eigenvectors = np.linalg.eigh(np.cov((points - origin).T))[1]
    normal = eigenvectors[:, 0]
    if (sides.articular.mean(axis=0) - sides.bone.mean(axis=0)) @ normal < 0:
        normal = -normal
    s_axis = eigenvectors[:, 2]
    return origin, np.array([s_axis, np.cross(normal, s_axis), normal])


def _plane_coordinates(points: np.ndarray, origin: np.ndarray, axes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    local = (points - origin) @ axes.T
    return local[:, :2], local[:, 2]


def _largest_component(mask: np.ndarray) -> np.ndarray:
    components = _components(mask)
    if not components:
        raise ValueError("The cartilage's articular and bone sides do not overlap in the plane")
    return max(components, key=lambda component: int(component.sum()))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    labels = sitk.GetArrayFromImage(sitk.ConnectedComponent(sitk.GetImageFromArray(mask.astype(np.uint8))))
    return [labels == value for value in range(1, int(labels.max()) + 1)]


def _loop_length(loop: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(loop - np.roll(loop, 1, axis=0), axis=1)))


def _signed_area(loop: np.ndarray) -> float:
    s, t = loop[:, 0], loop[:, 1]
    return 0.5 * float(np.sum(s * np.roll(t, -1) - np.roll(s, -1) * t))


def _ccw(loop: np.ndarray) -> np.ndarray:
    return loop if _signed_area(loop) > 0 else loop[::-1].copy()


def _centroid(loop: np.ndarray) -> np.ndarray:
    s, t = loop[:, 0], loop[:, 1]
    cross = s * np.roll(t, -1) - np.roll(s, -1) * t
    area = cross.sum() / 2
    return np.array([np.sum((s + np.roll(s, -1)) * cross), np.sum((t + np.roll(t, -1)) * cross)]) / (6 * area)


def _butterfly_blocks(sheet: CartilageSheet) -> list[np.ndarray]:
    """Return the plane points of each butterfly block on a grid, the inner block first.

    Each block is a (_PATCH_SAMPLES, _PATCH_SAMPLES, 2) array indexed [b, a].
    Outer block i runs along arc i of the perimeter in a at b = 0, toward the
    inner block at b = 1. Its side at a = 1 is the side at a = 0 of block
    i + 1. The inner block is a bilinear quadrilateral from corner 0 toward
    corner 1 in a and toward corner 3 in b, where outer block i's inner side
    runs straight from corner i to corner i + 1.

    The four outer blocks form a ring between the perimeter and the inner
    block's straight sides. The ring's interior starts from a discrete
    harmonic (Laplace) grid, is relaxed by Winslow's equations, and is then
    respaced evenly along each column, so the spokes bend around concavities
    of the perimeter. Straight spokes with transfinite interpolation fold the
    blocks at the real lateral tibial cartilage's notch. Letting the inner
    block's sides move too rounds its corners to nearly 180 degrees, and a
    Laplace grid alone folds at its corners.

    Raises:
        ValueError: If any block's map folds.
    """
    count = _PATCH_SAMPLES
    perimeter = sheet.perimeter
    # A split at a convex corner halves its small angle between two blocks,
    # but one at a concave notch splits its reflex angle, which a single
    # block would otherwise have to bend around.
    splits = butterfly_split_indices([perimeter], _signed_turning_angles(perimeter))
    points = np.roll(perimeter, -int(splits[0]), axis=0)
    parameters = np.interp(splits - splits[0], np.arange(len(points) + 1), chord_parameters(points))
    samples = np.linspace(0.0, 1.0, count)
    arcs = [curve(curve.t[0] + samples * (curve.t[-1] - curve.t[0])) for curve in split_periodic_spline(periodic_spline(points, 3), parameters)]
    centroid = _centroid(perimeter)
    corners = [centroid + _INNER_BLOCK_SCALE * (arc[0] - centroid) for arc in arcs]
    inner_sides = [corners[i] + samples[:, np.newaxis] * (corners[(i + 1) % 4] - corners[i]) for i in range(4)]

    # Ring rows run around the perimeter, row 0 on it and the last row on the
    # inner block's sides, with columns that wrap.
    columns = 4 * (count - 1)
    ring = np.zeros((count, columns, 2))
    ring[0] = np.vstack([arc[:-1] for arc in arcs])
    ring[-1] = np.vstack([side[:-1] for side in inner_sides])
    rows = np.arange(1, count - 1)[:, np.newaxis, np.newaxis] / (count - 1)
    ring[1:-1] = (1 - rows) * ring[0] + rows * ring[-1]
    index = np.arange((count - 2) * columns).reshape(count - 2, columns)
    edges = np.vstack(
        (np.column_stack((index.ravel(), np.roll(index, -1, axis=1).ravel())), np.column_stack((index[:-1].ravel(), index[1:].ravel())))
    )
    adjacency = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(index.size, index.size)).tocsr()
    adjacency = adjacency + adjacency.T
    # Each free point has four neighbors; those in the first and last free rows reach the fixed rows.
    laplacian = (diags(np.full(index.size, 4.0)) - adjacency).tocsc()
    right = np.zeros((count - 2, columns, 2))
    right[0] += ring[0]
    right[-1] += ring[-1]
    solved = spsolve(laplacian, right.reshape(-1, 2))
    ring[1:-1] = np.asarray(solved).reshape(count - 2, columns, 2)
    ring = _even_columns(_winslow(ring))

    b = samples[:, np.newaxis, np.newaxis]
    bottom, top = inner_sides[0][np.newaxis], inner_sides[2][::-1][np.newaxis]
    blocks = [(1 - b) * bottom + b * top]
    side = np.arange(count)
    blocks.extend(ring[:, (block * (count - 1) + side) % columns] for block in range(4))
    for block_index, grid in enumerate(blocks):
        # Every cell must keep the counterclockwise orientation of the unit square.
        da, db = grid[:-1, 1:] - grid[:-1, :-1], grid[1:, :-1] - grid[:-1, :-1]
        if np.any(da[..., 0] * db[..., 1] - da[..., 1] * db[..., 0] <= 0):
            raise ValueError(f"Butterfly block {block_index} folds; the perimeter is too concave. Try a larger perimeter_radius")
    return blocks


def _signed_turning_angles(loop: np.ndarray) -> np.ndarray:
    """Return the angle a counterclockwise loop of plane points turns at each point, negative where it is concave."""
    incoming = loop - np.roll(loop, 1, axis=0)
    outgoing = np.roll(loop, -1, axis=0) - loop
    cross = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
    return np.arctan2(cross, np.sum(incoming * outgoing, axis=1))


def _winslow(ring: np.ndarray) -> np.ndarray:
    """Relax the free rows of a ring grid by Winslow's equations, keeping its first and last rows.

    Winslow's equations make the grid's computational coordinates harmonic
    functions of position, the inverse of a Laplace grid, which keeps grid
    lines from crossing near the inner block's corners, where the ring turns
    through 270 degrees and a Laplace grid folds. Columns wrap around.
    """
    grid = ring.copy()
    for _ in range(_WINSLOW_ITERATIONS):
        east, west = np.roll(grid, -1, axis=1)[1:-1], np.roll(grid, 1, axis=1)[1:-1]
        north, south = grid[2:], grid[:-2]
        north_east, north_west = np.roll(grid, -1, axis=1)[2:], np.roll(grid, 1, axis=1)[2:]
        south_east, south_west = np.roll(grid, -1, axis=1)[:-2], np.roll(grid, 1, axis=1)[:-2]
        along, across = (east - west) / 2, (north - south) / 2
        alpha = np.sum(across**2, axis=-1, keepdims=True)
        beta = np.sum(along * across, axis=-1, keepdims=True)
        gamma = np.sum(along**2, axis=-1, keepdims=True)
        cross = (north_east - north_west - south_east + south_west) / 4
        updated = (alpha * (east + west) + gamma * (north + south) - 2 * beta * cross) / (2 * (alpha + gamma))
        change = float(np.max(np.abs(updated - grid[1:-1])))
        grid[1:-1] += _WINSLOW_RELAXATION * (updated - grid[1:-1])
        if change < _WINSLOW_TOLERANCE * float(np.max(np.ptp(ring[0], axis=0))):
            break
    return grid


def _even_columns(ring: np.ndarray) -> np.ndarray:
    """Respace the points along each column of a ring grid evenly by arc length, keeping its ends.

    Winslow's equations crowd the rows toward the inner block, by more than
    ten times along the real cartilage's spokes, and cubic patches through
    such uneven rows at uniform parameters overshoot and fold.
    """
    lengths = np.concatenate((np.zeros((1, ring.shape[1])), np.cumsum(np.linalg.norm(np.diff(ring, axis=0), axis=-1), axis=0)))
    fractions = np.linspace(0.0, 1.0, ring.shape[0])
    even = np.empty_like(ring)
    for column in range(ring.shape[1]):
        targets = fractions * lengths[-1, column]
        for axis in range(2):
            even[:, column, axis] = np.interp(targets, lengths[:, column], ring[:, column, axis])
    return even


def _add_patch(points: np.ndarray) -> int:
    """Add a B-spline surface through a (rows, columns, 3) grid of points, returning its tag.

    Rows and columns are interpolated at uniform parameters, so a patch's
    edge depends only on that edge's points and matches its neighbor's and
    its wall's exactly.
    """
    parameters = np.linspace(0.0, 1.0, points.shape[1])
    rows = [clamped_interpolation(parameters, row, _SPLINE_ORDER) for row in points]
    return add_swept_surface(rows, _SPLINE_ORDER, np.linspace(0.0, 1.0, points.shape[0]), _SPLINE_ORDER)


def _add_wall(articular: np.ndarray, bone: np.ndarray) -> int:
    """Add a ruled surface between matching edges of the two sides, returning its tag."""
    parameters = np.linspace(0.0, 1.0, len(articular))
    rows = [clamped_interpolation(parameters, edge, _SPLINE_ORDER) for edge in (articular, bone)]
    return add_swept_surface(rows, _SPLINE_ORDER, np.array([0.0, 1.0]), 1)


def _add_prism(sheet: CartilageSheet, hole: np.ndarray) -> tuple[int, int]:
    """Add a prism along the normal through the whole sheet over a hole's outline, returning its dimension and tag."""
    heights = np.concatenate((sheet.articular.ev(hole[:, 0], hole[:, 1]), sheet.bone.ev(hole[:, 0], hole[:, 1])))
    margin = sheet.mean_thickness
    base = sheet.origin + hole @ sheet.axes[:2] + (heights.min() - margin) * sheet.axes[2]
    wire = add_closed_wire(periodic_spline(base, _SPLINE_ORDER), len(base))
    face = gmsh.model.occ.addPlaneSurface([wire])
    height = float(np.ptp(heights)) + 2 * margin
    extruded = gmsh.model.occ.extrude([(2, face)], *(height * sheet.axes[2]).tolist())
    return next((dim, tag) for dim, tag in extruded if dim == 3)
