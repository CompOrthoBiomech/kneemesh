"""B-rep generation for cartilage sheets, the lateral and medial tibial and the patellar cartilage.

Each sheet is a pair of height fields over a two-dimensional domain: its
articular side above and the side against its bone below, within one smooth
perimeter in that domain. A frame maps domain points and heights to physical
points. For these cartilages it is the plane normal to the sheet's thinnest
principal direction; the femoral cartilage uses the same machinery over a
cylinder (geo.femur). Straight walls join the two sides at the perimeter,
tilted out of the height lines so they meet both sides closer to right angles.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import override

import gmsh
import numpy as np
import SimpleITK as sitk
from loguru import logger
from scipy.interpolate import RectBivariateSpline, griddata
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.spatial import KDTree
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkFiltersCore import vtkFlyingEdges2D, vtkStripper

from kneemesh.brep import (
    BRep,
    add_closed_wire,
    export_brep,
    gmsh_session,
    periodic_spline,
    resample_closed,
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

# Neighboring points that estimate a surface normal when fitting a CylinderFrame.
_NORMAL_NEIGHBORS = 16
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
_SPLINE_ORDER = 3
# Spacing, in physical units, of the domain grid each side's surface interpolates, and the grid's margin beyond the perimeter in spacings.
_SURFACE_SPACING = 0.5
_SURFACE_MARGIN = 3
# Largest wall tilt, as a fraction of the local thickness and of perimeter_radius, and the tilt's smoothing along the perimeter, as a fraction of perimeter_radius.
_MAX_WALL_TILT = 0.25
_MAX_TILT_REACH = 0.25
_TILT_SMOOTHING = 0.5


class SheetFrame(ABC):
    """Coordinates of a sheet: domain points (s, t) and a height along each domain point's straight line.

    Domain lengths approximate physical lengths, and heights grow from the
    bone toward the articular side. Turning from s toward t is counterclockwise
    seen from the articular side.
    """

    @abstractmethod
    def local(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the (N, 2) domain points and (N,) heights of (N, 3) physical points."""

    @abstractmethod
    def physical(self, domain_points: np.ndarray, heights: np.ndarray) -> np.ndarray:
        """Return the (N, 3) physical points at (N, 2) domain points and (N,) heights."""


@dataclass(frozen=True, eq=False)
class PlaneFrame(SheetFrame):
    """A plane: domain point (s, t) and height z give origin + s * axes[0] + t * axes[1] + z * axes[2].

    Attributes:
        origin: Physical origin of the plane.
        axes: (3, 3) right-handed orthonormal rows: the plane's s and t axes,
            and its normal, which points from the bone toward the articular side.
    """

    origin: np.ndarray
    axes: np.ndarray

    @classmethod
    def fit(cls, sides: CartilageSides) -> "PlaneFrame":
        """Return the plane through the sides' points normal to their thinnest principal direction, s along the longest."""
        points = np.vstack((sides.articular, sides.bone))
        origin = points.mean(axis=0)
        eigenvectors = np.linalg.eigh(np.cov((points - origin).T))[1]
        normal = eigenvectors[:, 0]
        if (sides.articular.mean(axis=0) - sides.bone.mean(axis=0)) @ normal < 0:
            normal = -normal
        s_axis = eigenvectors[:, 2]
        return cls(origin, np.array([s_axis, np.cross(normal, s_axis), normal]))

    @override
    def local(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        local = (points - self.origin) @ self.axes.T
        return local[:, :2], local[:, 2]

    @override
    def physical(self, domain_points: np.ndarray, heights: np.ndarray) -> np.ndarray:
        return self.origin + domain_points @ self.axes[:2] + np.outer(heights, self.axes[2])


@dataclass(frozen=True, eq=False)
class CylinderFrame(SheetFrame):
    """A cylinder unrolled: s runs along its axis, t is arc length at a reference radius around it, and heights are distances from the axis.

    The physical point at (s, t) and height r is center + s * axes[0] +
    r * (cos(theta) * axes[1] + sin(theta) * axes[2]), with theta = -t / radius.
    Heights lines are rays from the axis, so the frame suits a sheet that
    wraps around a convex bone, as the femoral cartilage wraps the condyles.

    Attributes:
        center: Physical point on the axis at s = 0.
        axes: (3, 3) right-handed orthonormal rows: the axis, the direction
            from it of t = 0, and their cross product.
        radius: Reference radius that turns angles into arc lengths.
    """

    center: np.ndarray
    axes: np.ndarray
    radius: float

    @classmethod
    def fit(cls, sides: CartilageSides) -> "CylinderFrame":
        """Return the cylinder the bone side's points wrap, with t = 0 in the middle of the arc they span.

        The axis is the direction the bone side's surface normals, estimated
        from neighboring points, vary least along, as they all would be
        normal to the axis on a cylinder. Its position is the center of the
        circle fitted to the bone side projected along it, and the reference
        radius the mean distance of both sides' points from it.

        Raises:
            ValueError: If the sides wrap all the way around, leaving no gap of at least 30 degrees.
        """
        bone = sides.bone
        _, neighbors = KDTree(bone).query(bone, k=_NORMAL_NEIGHBORS)
        offsets = bone[neighbors] - bone[neighbors].mean(axis=1, keepdims=True)
        normals = np.linalg.eigh(np.einsum("nki,nkj->nij", offsets, offsets))[1][:, :, 0]
        axis = np.linalg.eigh(normals.T @ normals)[1][:, 0]
        axis = axis if axis[np.argmax(np.abs(axis))] > 0 else -axis
        first = np.cross(axis, np.eye(3)[np.argmin(np.abs(axis))])
        first /= np.linalg.norm(first)
        second = np.cross(axis, first)

        points = np.vstack((sides.articular, bone))
        origin = points.mean(axis=0)
        x, y = (bone - origin) @ first, (bone - origin) @ second
        circle = np.linalg.lstsq(np.column_stack((x, y, np.ones_like(x))), x**2 + y**2, rcond=None)[0]
        center = origin + 0.5 * circle[0] * first + 0.5 * circle[1] * second

        offsets = points - center
        angles = np.sort(np.arctan2(offsets @ second, offsets @ first))
        gaps = np.diff(np.concatenate((angles, [angles[0] + 2 * np.pi])))
        widest = int(np.argmax(gaps))
        if gaps[widest] < np.radians(30.0):
            raise ValueError(f"The cartilage wraps {np.degrees(2 * np.pi - gaps[widest]):.0f} degrees around its axis, too far to unroll")
        middle = angles[widest] + 0.5 * gaps[widest] + np.pi
        start = np.cos(middle) * first + np.sin(middle) * second
        radial = offsets - np.outer(offsets @ axis, axis)
        frame = cls(center, np.array([axis, start, np.cross(axis, start)]), float(np.linalg.norm(radial, axis=1).mean()))
        logger.debug(
            "Fitted a cylinder of radius {:.4g} along {}; the sides span {:.0f} degrees around it",
            frame.radius,
            np.round(axis, 3).tolist(),
            np.degrees(2 * np.pi - gaps[widest]),
        )
        return frame

    @override
    def local(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        local = (points - self.center) @ self.axes.T
        theta = np.arctan2(local[:, 2], local[:, 1])
        return np.column_stack((local[:, 0], -self.radius * theta)), np.hypot(local[:, 1], local[:, 2])

    @override
    def physical(self, domain_points: np.ndarray, heights: np.ndarray) -> np.ndarray:
        theta = -domain_points[:, 1] / self.radius
        directions = np.outer(np.cos(theta), self.axes[1]) + np.outer(np.sin(theta), self.axes[2])
        return self.center + np.outer(domain_points[:, 0], self.axes[0]) + heights[:, np.newaxis] * directions


@dataclass(frozen=True, eq=False)
class CartilageSheet:
    """A cartilage sheet as heights of its two sides over a domain, within a perimeter.

    Attributes:
        frame: Maps domain points and heights to physical points.
        perimeter: (N, 2) domain points around the sheet, counterclockwise,
            without the first repeated.
        holes: (M, 2) domain points around each hole, counterclockwise.
        articular: Height of the articular side over the domain.
        bone: Height of the side against the bone, already lowered where the
            sheet was thinner than its minimum thickness.
        mean_thickness: Mean thickness along the height lines within the
            perimeter, before thickening.
        min_thickness: The thickness thin regions were raised to.
        tilt: How far outward the wall's articular edge lies from its bone
            edge in the domain at each perimeter point. Each side's domain
            points move half of it along the perimeter's outward normal,
            the articular side outward and the bone side inward, fading
            out at tilt_reach inside the perimeter.
        tilt_reach: How far inside the perimeter the tilt fades out.
    """

    frame: SheetFrame
    perimeter: np.ndarray
    holes: list[np.ndarray]
    articular: RectBivariateSpline
    bone: RectBivariateSpline
    mean_thickness: float
    min_thickness: float
    tilt: np.ndarray
    tilt_reach: float

    def points(self, domain_points: np.ndarray, side: str) -> np.ndarray:
        """Return the physical points on a side, "articular" or "bone", over (..., 2) domain points, moved by half the tilt near the perimeter."""
        heights = self.articular if side == "articular" else self.bone
        flat = self.tilted(domain_points.reshape(-1, 2), side)
        physical = self.frame.physical(flat, heights.ev(flat[:, 0], flat[:, 1]))
        return physical.reshape(*domain_points.shape[:-1], 3)

    def tilted(self, domain_points: np.ndarray, side: str) -> np.ndarray:
        """Return (N, 2) domain points of a side, moved along the perimeter's outward normal by half the tilt, out for the articular side and in for the bone side.

        The move fades as the square of the distance from the perimeter, out
        to tilt_reach inside it, at the nearest point of the perimeter.
        Outside, the same square of the signed distance continues to grow, up
        to four times the tilt at tilt_reach, so a side's surface, which
        reaches past the perimeter to be trimmed, has no crease along it.
        """
        if not self.tilt.any():
            return domain_points
        distance, segment, fraction = _nearest_on_loop(self.perimeter, domain_points)
        following = (segment + 1) % len(self.perimeter)
        tilt = (1 - fraction) * self.tilt[segment] + fraction * self.tilt[following]
        normals = _outward_normals(self.perimeter)
        normal = (1 - fraction)[:, np.newaxis] * normals[segment] + fraction[:, np.newaxis] * normals[following]
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        nearest = (1 - fraction)[:, np.newaxis] * self.perimeter[segment] + fraction[:, np.newaxis] * self.perimeter[following]
        inside = np.einsum("ij,ij->i", domain_points - nearest, normal) <= 0
        weight = (1 - np.clip(np.where(inside, distance, -distance) / self.tilt_reach, -1.0, 1.0)) ** 2
        sign = 0.5 if side == "articular" else -0.5
        return domain_points + (sign * weight * tilt)[:, np.newaxis] * normal

    @property
    def area(self) -> float:
        """The domain area inside the perimeter, less the holes."""
        return _signed_area(self.perimeter) - sum(_signed_area(hole) for hole in self.holes)


def fit_sheet(
    sides: CartilageSides,
    perimeter_radius: float = DEFAULT_PERIMETER_RADIUS_HEXAHEDRAL,
    perimeter_inset: float = 0.0,
    min_thickness: float = DEFAULT_MIN_THICKNESS_HEXAHEDRAL,
    hole_thickness: float = 0.0,
    frame: SheetFrame | None = None,
) -> CartilageSheet:
    """Fit a cartilage sheet to the two sides of a cartilage surface.

    By default the domain is the plane normal to the thinnest principal
    direction of the side points. Each side's footprint is where its points
    project onto the domain,
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

    The wall at the perimeter is tilted so it leaves both sides at about
    equal angles: where the sheet curves down at its rim, a wall along the
    height lines meets the articular side at up to 130 degrees or more. A
    straight wall cannot meet both sides of a tapering rim at right angles,
    so the tilt balances the two, limited to _MAX_WALL_TILT of the local
    thickness and _MAX_TILT_REACH of perimeter_radius, and smoothed along
    the perimeter. Its band fades out at perimeter_radius inside the perimeter.

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

    frame = frame or PlaneFrame.fit(sides)
    articular_plane, articular_z = frame.local(sides.articular)
    bone_plane, bone_z = frame.local(sides.bone)
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
    articular_spline, bone_spline = grid.spline(articular), grid.spline(articular - thickened)
    tilt = _wall_tilt(perimeter, articular_spline, bone_spline, perimeter_radius)
    sheet = CartilageSheet(frame, perimeter, holes, articular_spline, bone_spline, mean_thickness, minimum, tilt, perimeter_radius)
    logger.debug(
        "Fitted a sheet with perimeter length {:.4g}, {} holes, and walls tilted by {:.3g} at the median and up to {:.3g}",
        _loop_length(perimeter),
        len(holes),
        float(np.median(np.abs(tilt))),
        float(np.abs(tilt).max()),
    )
    return sheet


def sheet_to_brep(sheet: CartilageSheet) -> BRep:
    """Build a solid of a cartilage sheet: its two sides, the wall between them at the perimeter, and the walls of its holes.

    Each side is one bicubic B-spline surface interpolating the side's points
    over a grid of domain points, trimmed by the perimeter and the hole
    outlines drawn in its parameter space, so both sides share one
    parameterization: the surface parameters are the domain coordinates
    (s, t). Each wall is ruled between matching edges of the two sides, so it
    is straight from side to side, tilted at the perimeter as fit_sheet
    describes and along the height lines at the holes. Hexahedral meshing
    relies on both: mesh.sheet.hexahedral_mesh pairs the sides' points at
    equal parameters and follows the walls' straight lines.

    Returns:
        The solid as one volume.
    """
    with gmsh_session():
        loops = [sheet.perimeter, *sheet.holes]
        faces = [_add_side(sheet, side, loops) for side in ("articular", "bone")]
        gmsh.model.occ.synchronize()
        rims = [_rims_by_loop(face, loops) for face in faces]
        walls: list[int] = []
        for articular_rim, bone_rim in zip(*rims, strict=True):
            wires = [gmsh.model.occ.addWire([articular_rim]), gmsh.model.occ.addWire([bone_rim])]
            walls.extend(tag for dim, tag in gmsh.model.occ.addThruSections(wires, makeSolid=False, makeRuled=True) if dim == 2)
        shell = gmsh.model.occ.addSurfaceLoop([*faces, *walls], sewing=True)
        gmsh.model.occ.addVolume([shell])
        gmsh.model.occ.synchronize()
        return export_brep()


def _add_side(sheet: CartilageSheet, side: str, loops: Sequence[np.ndarray]) -> int:
    """Add one side of a sheet as a B-spline surface over the domain, trimmed by loops in its parameter space, returning its tag.

    The surface interpolates the side's points on a grid of domain points
    reaching _SURFACE_MARGIN spacings beyond the perimeter, with knots at the
    grid's s and t coordinates, so its parameters are the domain coordinates.
    """
    low = sheet.perimeter.min(axis=0) - _SURFACE_MARGIN * _SURFACE_SPACING
    high = sheet.perimeter.max(axis=0) + _SURFACE_MARGIN * _SURFACE_SPACING
    s, t = (np.linspace(low[axis], high[axis], int(np.ceil((high[axis] - low[axis]) / _SURFACE_SPACING)) + 1) for axis in range(2))
    points = sheet.points(np.stack(np.meshgrid(s, t, indexing="ij"), axis=-1), side)
    splines = [RectBivariateSpline(s, t, points[..., axis], kx=_SPLINE_ORDER, ky=_SPLINE_ORDER, s=0) for axis in range(3)]
    knots_s, knots_t = splines[0].get_knots()
    count_s, count_t = len(knots_s) - _SPLINE_ORDER - 1, len(knots_t) - _SPLINE_ORDER - 1
    poles = np.stack([spline.get_coeffs().reshape(count_s, count_t) for spline in splines], axis=-1)
    # gmsh takes the poles with u, here s, varying fastest.
    tags = [gmsh.model.occ.addPoint(*pole) for pole in poles.transpose(1, 0, 2).reshape(-1, 3)]
    unique_s, multiplicities_s = np.unique(knots_s, return_counts=True)
    unique_t, multiplicities_t = np.unique(knots_t, return_counts=True)
    wires = [add_closed_wire(periodic_spline(np.column_stack((loop, np.zeros(len(loop)))), _SPLINE_ORDER), len(loop)) for loop in loops]
    return gmsh.model.occ.addBSplineSurface(
        tags,
        count_s,
        degreeU=_SPLINE_ORDER,
        degreeV=_SPLINE_ORDER,
        knotsU=unique_s.tolist(),
        knotsV=unique_t.tolist(),
        multiplicitiesU=multiplicities_s.tolist(),
        multiplicitiesV=multiplicities_t.tolist(),
        wireTags=wires,
        wire3D=False,
    )


def _rims_by_loop(face: int, loops: Sequence[np.ndarray]) -> list[int]:
    """Return the boundary curve of a trimmed side over each loop, in the loops' order, matched by its points' parameters."""
    rims = [-1] * len(loops)
    for _, curve in gmsh.model.getBoundary([(2, face)], combined=False, oriented=False):
        low, high = gmsh.model.getParametrizationBounds(1, abs(curve))
        point = gmsh.model.getValue(1, abs(curve), [0.5 * (low[0] + high[0])])
        domain_point = np.reshape(gmsh.model.getParametrization(2, face, point), (1, 2))
        rims[int(np.argmin([_nearest_on_loop(loop, domain_point)[0][0] for loop in loops]))] = abs(curve)
    return rims


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

    def inside(self, loop: np.ndarray) -> np.ndarray:
        """Pixels whose centers lie inside a closed loop of points, filled row by row between its crossings."""
        start, end = loop, np.roll(loop, -1, axis=0)
        s = self.s
        mask = np.zeros(self.shape, dtype=bool)
        for row, t in enumerate(self.t):
            straddles = (start[:, 1] > t) != (end[:, 1] > t)
            first, second = start[straddles], end[straddles]
            crossings = np.sort(first[:, 0] + (t - first[:, 1]) * (second[:, 0] - first[:, 0]) / (second[:, 1] - first[:, 1]))
            for low, high in crossings.reshape(-1, 2):
                mask[row] |= (s > low) & (s < high)
        return mask

    def spline(self, values: np.ndarray) -> RectBivariateSpline:
        return RectBivariateSpline(self.s, self.t, values.T, kx=_SPLINE_ORDER, ky=_SPLINE_ORDER, s=0)


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


def _outward_normals(loop: np.ndarray) -> np.ndarray:
    """Return the unit outward normal at each point of a counterclockwise loop of plane points, from its neighbors."""
    tangents = np.roll(loop, -1, axis=0) - np.roll(loop, 1, axis=0)
    return np.column_stack((tangents[:, 1], -tangents[:, 0])) / np.linalg.norm(tangents, axis=1, keepdims=True)


def _nearest_on_loop(loop: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return each point's distance to a closed polyline, the index of the nearest segment's first point, and the fraction along it."""
    following = np.roll(loop, -1, axis=0)
    _, nearest = KDTree(loop).query(points)
    best = (np.full(len(points), np.inf), np.zeros(len(points), dtype=np.int64), np.zeros(len(points)))
    # The nearest point of the polyline lies on one of the two segments at its nearest vertex.
    for segment in (nearest, (nearest - 1) % len(loop)):
        start, direction = loop[segment], following[segment] - loop[segment]
        fraction = np.clip(np.einsum("ij,ij->i", points - start, direction) / np.einsum("ij,ij->i", direction, direction), 0.0, 1.0)
        distance = np.linalg.norm(points - start - fraction[:, np.newaxis] * direction, axis=1)
        closer = distance < best[0]
        best = (np.where(closer, distance, best[0]), np.where(closer, segment, best[1]), np.where(closer, fraction, best[2]))
    return best


def _wall_tilt(perimeter: np.ndarray, articular: RectBivariateSpline, bone: RectBivariateSpline, reach: float) -> np.ndarray:
    """Return the tilt at each perimeter point that leaves the wall at equal angles to both sides, within its limits.

    In the plane of the perimeter's outward normal and the heights, a wall
    whose articular edge lies d outward of its bone edge, over thickness h,
    is perpendicular to a side of slope g when d = -h g. The tilt takes the
    mean over the two sides, which balances the two angles.
    """
    outward = _outward_normals(perimeter)
    step = 1e-3 * reach

    def slope(heights: RectBivariateSpline) -> np.ndarray:
        ahead, behind = perimeter + step * outward, perimeter - step * outward
        return (heights.ev(ahead[:, 0], ahead[:, 1]) - heights.ev(behind[:, 0], behind[:, 1])) / (2 * step)

    thickness = articular.ev(perimeter[:, 0], perimeter[:, 1]) - bone.ev(perimeter[:, 0], perimeter[:, 1])
    limit = np.minimum(_MAX_WALL_TILT * thickness, _MAX_TILT_REACH * reach)
    tilt = np.clip(-thickness * (slope(articular) + slope(bone)) / 2, -limit, limit)
    # Smoothing spreads the tilt of thick stretches into thin ones, so it is limited again afterward.
    spacing = _loop_length(perimeter) / len(perimeter)
    return np.clip(gaussian_filter1d(tilt, _TILT_SMOOTHING * reach / spacing, mode="wrap"), -limit, limit)
