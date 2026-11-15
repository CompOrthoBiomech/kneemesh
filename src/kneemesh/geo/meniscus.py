"""B-rep generation for the meniscus, a curved tube lofted through sections normal to its centerline."""

import math
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Literal

import gmsh
import numpy as np
import SimpleITK as sitk
from loguru import logger
from scipy.interpolate import BSpline, make_splprep
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import KDTree
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkCommonDataModel import vtkPlane, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkCutter, vtkStripper

from kneemesh.brep import (
    MAX_SPLINE_ORDER,
    BRep,
    add_bspline_curve,
    add_closed_points,
    add_closed_wire,
    add_periodic_swept_surface,
    add_swept_surface,
    butterfly_split_indices,
    chord_parameters,
    clamped_interpolation,
    export_brep,
    gmsh_session,
    open_chord_parameters,
    periodic_poles_and_knots,
    periodic_spline,
    resample_closed,
    sample_closed,
    split_periodic_spline,
)
from kneemesh.vol import ContourSlice, require_3d_label_map

_MIN_LABEL_POINTS = 4
_REFINEMENT_ITERATIONS = 3
_ARC_LENGTH_SAMPLES = 2000
_STATION_NEIGHBORS = 2
_END_EXTENSION_STATIONS = 4
_MIN_SECTION_RATIO = 0.5
_SKELETON_TOLERANCE_VOXELS = 0.5
_CENTROID_TOLERANCE_VOXELS = 0.25
_CENTROID_TOLERANCE_HALF_WIDTHS = 0.05
_MIN_END_AREA_RATIO = 0.2
_MAX_CONTOUR_SEGMENTS = 1_000_000
# Corner tracking: contours are first sampled at _DENSE_POINTS, and turning is summed over a Gaussian
# window of _CORNER_WINDOW of the perimeter. A track may drift about _CORNER_DRIFT of the perimeter per
# slice, at a cost of _CORNER_DRIFT_WEIGHT radians per drift squared. Up to _MAX_CORNERS tracks are kept
# whose turning averages _MIN_CORNER_TURN, each at least _CORNER_SEPARATION of the perimeter from the others.
_DENSE_POINTS = 512
_CORNER_WINDOW = 1 / 48
_CORNER_DRIFT = 0.03
_CORNER_DRIFT_WEIGHT = 0.1
_MIN_CORNER_TURN = math.radians(35.0)
_MAX_CORNERS = 4
_CORNER_SEPARATION = 1 / 8
# Inner edge clipping: chord heights at _CLIP_STATIONS distances across each section, a cut at most
# _MAX_CLIP_HEIGHT of the section's greatest height, and tracked corners kept only beyond _CLIP_CORNER_MARGIN
# of the curved part's length from the cut's corners.
_CLIP_STATIONS = 400
_MAX_CLIP_HEIGHT = 0.5
_CLIP_CORNER_MARGIN = 0.05

_DEFAULT_CAP_OFFSET_RADII = 0.25
_PIECE_SAMPLES = 20
_MIN_HEX_POINTS = 8
# Inner block corners lie this fraction of the way from the section's centroid to the split points.
_INNER_BLOCK_SCALE = 0.5


def centerline(label_map: sitk.Image, label: int = 1) -> BSpline:
    """Fit a smooth curve through the centers of a label's cross-sections.

    BinaryThinning only thins in 2D, so the label is projected along the image
    axis closest to its thinnest direction and the longest path through that
    skeleton gives a first estimate of the curve. The curve is then repeatedly
    moved onto the centroids of the label's cross-sections normal to it.

    Args:
        label_map: 3D LabelMap of sitkUInt8, sitkUInt16, or sitkInt16 pixels.
        label: Value of the structure to fit.

    Returns:
        A cubic B-spline over u in [0, 1]. Evaluating it at N parameters gives
        an (N, 3) array of physical points. Which end of the structure is
        u = 0 is unspecified.

    Raises:
        ValueError: If the image is not a 3D LabelMap of an accepted pixel
            type, or the label is absent or too small to fit.
    """
    require_3d_label_map(label_map)

    mask = sitk.GetArrayViewFromImage(label_map) == label
    points = _physical_points(label_map, np.argwhere(mask))
    if len(points) == 0:
        raise ValueError(f"Label {label} is not present in the image")
    if len(points) < _MIN_LABEL_POINTS:
        raise ValueError("Label is too small to fit a curve")

    voxel_size = max(label_map.GetSpacing())
    path, half_width = _skeleton_path(label_map, mask, points)
    # Tying section spacing and smoothing to the structure's width, not just
    # the voxel size, keeps their behavior at the tips the same at any resolution.
    step = max(2 * voxel_size, half_width / 2)
    centroid_tolerance = max(
        _CENTROID_TOLERANCE_VOXELS * voxel_size,
        _CENTROID_TOLERANCE_HALF_WIDTHS * half_width,
    )

    spline = _fit(path, _SKELETON_TOLERANCE_VOXELS * voxel_size)
    for _ in range(_REFINEMENT_ITERATIONS):
        centroids = _section_centroids(spline, points, step)
        spline = _fit(centroids, centroid_tolerance)
    logger.debug("Centerline has {} control points; sections every {:.3g} with half-width {:.3g}", len(spline.c), step, half_width)
    return spline


def slice_isocontour(
    mesh: vtkPolyData,
    curve: BSpline,
    spacing: float,
    points_per_slice: int = 32,
    clip_height: float | None = None,
    clip_ratio: float = 0.0,
) -> list[ContourSlice]:
    """Slice a mesh with planes normal to a curve at uniform arc-length spacing.

    Slices are spaced exactly spacing apart along the curve, with any remainder
    split equally between its two ends, and are ordered along the curve from
    u = 0. Each cut is resampled to ordered points with a closed Kochanek
    spline: the section's corners, tracked through the slices by their
    turning, at the same point indices on every slice, and points evenly
    spaced between them (_follow_corners). Slices are removed from either end
    of the curve until one encloses at least 20% of the average area.

    In-plane axes are defined from the curve's best-fit plane, so they do not
    twist along a roughly planar curve.

    With clip_height or clip_ratio, each section's thin inner edge, its
    sharpest corner, is cut off straight where the section first reaches
    that height, measured across the direction from its centroid to the
    corner (see _clip_inner_edges). The contour is then its curved part,
    from one corner of the cut to the other, and a straight run back.

    Args:
        mesh: Closed surface from extract_isocontour or smooth_isocontour.
        curve: Curve from centerline.
        spacing: Arc-length distance between slices, in physical units.
        points_per_slice: Number of points on each contour.
        clip_height: Height in physical units at which to cut off the
            sections' thin inner edges, or None to use clip_ratio.
        clip_ratio: Height at which to cut off the sections' thin inner edges,
            as a fraction of the mean of the sections' greatest heights,
            those of their outer sides. 0 keeps the inner edges.

    Returns:
        The retained ContourSlices.

    Raises:
        ValueError: If spacing is not positive, points_per_slice is below 3,
            a clipping height or ratio is negative, or a retained plane does
            not cut the mesh in a closed contour.
    """
    if spacing <= 0:
        raise ValueError("Slice spacing must be positive")
    if points_per_slice < 3:
        raise ValueError("Slices need at least 3 points")
    if (clip_height is not None and clip_height < 0) or clip_ratio < 0:
        raise ValueError(f"Clipping heights must not be negative, got {clip_height} and {clip_ratio}")

    curve_u, lengths = _arc_lengths(curve)
    count = math.floor(lengths[-1] / spacing) + 1
    first_station = (lengths[-1] - (count - 1) * spacing) / 2
    stations_u = np.interp(first_station + spacing * np.arange(count), lengths, curve_u)
    centers = curve(stations_u)
    normals = curve(stations_u, nu=1)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    reference = np.linalg.eigh(np.cov(curve(curve_u).T))[1][:, 0]
    u_axes = np.cross(reference, normals)
    u_axes /= np.linalg.norm(u_axes, axis=1, keepdims=True)
    v_axes = np.cross(normals, u_axes)

    slices = [
        _slice_at(mesh, center, u_axis, v_axis, normal, _DENSE_POINTS) for center, u_axis, v_axis, normal in zip(centers, u_axes, v_axes, normals)
    ]
    areas = np.array([0.0 if contour is None else contour.area for contour in slices])
    if not areas.any():
        raise ValueError("The curve's planes do not cut the mesh in any closed contour")

    keep = np.flatnonzero(areas >= _MIN_END_AREA_RATIO * areas.mean())
    retained = slices[keep[0] : keep[-1] + 1]
    if any(contour is None for contour in retained):
        raise ValueError("A plane along the curve does not cut the mesh in a closed contour")
    logger.debug("Kept {} of {} slices after clipping ends below {:.0%} of the mean area", len(retained), len(slices), _MIN_END_AREA_RATIO)
    aligned = _align_with_previous(retained)  # pyright: ignore[reportArgumentType]
    if clip_height or clip_ratio:
        return _clip_inner_edges(aligned, points_per_slice, clip_height, clip_ratio)
    return _follow_corners(aligned, points_per_slice)


def contours_to_brep(
    slices: list[ContourSlice],
    spline_order: int = 3,
    cap: Literal["none", "planar", "smooth"] = "none",
    cap_offset: float | None = None,
) -> BRep:
    """Loft a B-spline surface through contour slices, optionally capping its ends.

    Each contour becomes a periodic B-spline through all of its points, at
    parameters shared by every contour, and the surface interpolates those
    curves in the order given, at parameters proportional to the distance
    between slice centers. Without caps it spans only the first to last slice.

    Args:
        slices: Contours from slice_isocontour, in order along the curve.
        spline_order: Polynomial degree of the contour splines. The degree
            along the curve is at most 3. 3 is cubic, and 1 gives a ruled surface.
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
            1 to 25, a contour has too few points for spline_order, contours
            have unequal point counts, cap is unknown, or cap_offset is not
            positive.
    """
    if cap not in ("none", "planar", "smooth"):
        # Unreachable for type-checked callers, but configs and scripts can pass any string.
        raise ValueError(f"cap must be 'none', 'planar', or 'smooth', got {cap!r}")  # pyright: ignore[reportUnreachable]
    if cap_offset is not None and cap_offset <= 0:
        raise ValueError(f"cap_offset must be positive, got {cap_offset}")
    _require_loftable(slices, spline_order)

    if len({len(contour.points) for contour in slices}) > 1:
        raise ValueError("Lofting needs the same number of points on every contour")
    if len({contour.straight_from for contour in slices}) > 1:
        raise ValueError("Lofting needs every contour clipped alike, or none")
    if slices[0].straight_from is not None:
        if cap == "smooth":
            raise ValueError("Contours with clipped inner edges take planar caps or none, not smooth ones")
        return _clipped_loft(slices, spline_order, cap == "planar")

    point_count = len(slices[0].points)
    projected = [_project_onto_plane(contour) for contour in slices]
    # Shared parameters give every contour spline the same knot vector, so the
    # surface's poles form a grid. Contour points are evenly spaced, so the
    # mean chord parameters stay close to each contour's own.
    parameters = np.mean([chord_parameters(points) for points in projected], axis=0)
    splines = [periodic_spline(points, spline_order, parameters) for points in projected]
    along = open_chord_parameters(np.array([contour.center for contour in slices]))
    along_degree = min(spline_order, 3, len(slices) - 1)

    with gmsh_session():
        loft = add_periodic_swept_surface(splines, point_count, along, along_degree)
        ends = [(slices[0], splines[0], -1.0), (slices[-1], splines[-1], 1.0)]
        caps = None
        if cap == "planar":
            caps = [gmsh.model.occ.addPlaneSurface([add_closed_wire(spline, point_count)]) for _, spline, _ in ends]
        elif cap == "smooth":
            caps = [_add_smooth_cap(spline, contour, sign * contour.normal, cap_offset) for contour, spline, sign in ends]
        if caps is not None:
            # The caps' rims are the loft's end sections, so sewing joins them.
            shell = gmsh.model.occ.addSurfaceLoop([loft, *caps], sewing=True)
            gmsh.model.occ.addVolume([shell])
        gmsh.model.occ.synchronize()
        return export_brep()


def _clipped_loft(slices: list[ContourSlice], spline_order: int, planar_caps: bool) -> BRep:
    """Loft contours with clipped inner edges: a surface through their curved parts and a ruled one through their cuts.

    Each curved part, from points[0] to points[straight_from], is a clamped
    B-spline at parameters shared by every contour, and the cut a straight
    line between the same two corners. Both surfaces interpolate their
    sections' poles along the curve at the same parameters, so their edges,
    the inner edge's top and bottom, depend only on the corners and match
    exactly.
    """
    end = slices[0].straight_from
    assert end is not None
    projected = [_project_onto_plane(contour) for contour in slices]
    curved = [points[: end + 1] for points in projected]
    parameters = np.mean([open_chord_parameters(points) for points in curved], axis=0)
    degree = min(spline_order, 3, end)
    rows = [clamped_interpolation(parameters, points, degree) for points in curved]
    lines = [(np.array([0.0, 0.0, 1.0, 1.0]), np.array([points[-1], points[0]])) for points in curved]
    along = open_chord_parameters(np.array([contour.center for contour in slices]))
    along_degree = min(spline_order, 3, len(slices) - 1)

    with gmsh_session():
        faces = [add_swept_surface(rows, degree, along, along_degree), add_swept_surface(lines, 1, along, along_degree)]
        if planar_caps:
            for (knots, poles), (_, ends) in ((rows[0], lines[0]), (rows[-1], lines[-1])):
                wire = gmsh.model.occ.addCurveLoop(
                    [add_bspline_curve(knots, poles, degree), add_bspline_curve(np.array([0.0, 0.0, 1.0, 1.0]), ends, 1)]
                )
                faces.append(gmsh.model.occ.addPlaneSurface([wire]))
            gmsh.model.occ.addVolume([gmsh.model.occ.addSurfaceLoop(faces, sewing=True)])
        else:
            gmsh.model.occ.addSurfaceLoop(faces, sewing=True)
        gmsh.model.occ.synchronize()
        return export_brep()


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
    meet exactly. contours_to_brep builds its surface the same way, but at
    shared contour parameters rather than split at the block corners, so the
    two differ slightly.

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

    splits = butterfly_split_indices([contour.points for contour in slices])
    samples = np.linspace(0.0, 1.0, _PIECE_SAMPLES)
    # pieces[slice][arc] holds points along the four boundary arcs; arc i runs from split i to split i + 1.
    pieces, inner_corners = [], []
    for contour in slices:
        points = np.roll(_project_onto_plane(contour), -int(splits[0]), axis=0)
        parameters = np.interp(splits - splits[0], np.arange(len(points) + 1), chord_parameters(points))
        curves = split_periodic_spline(periodic_spline(points, spline_order), parameters)
        slice_pieces = [curve(curve.t[0] + samples * (curve.t[-1] - curve.t[0])) for curve in curves]
        centroid = _section_centroid(contour)
        pieces.append(slice_pieces)
        inner_corners.append([centroid + _INNER_BLOCK_SCALE * (piece[0] - centroid) for piece in slice_pieces])

    centers = np.array([contour.center for contour in slices])
    along = open_chord_parameters(centers)
    along_degree = min(spline_order, 3, len(slices) - 1)
    line_knots = np.array([0.0, 0.0, 1.0, 1.0])
    arc_rows = [[clamped_interpolation(samples, slice_pieces[arc], spline_order) for slice_pieces in pieces] for arc in range(4)]
    # Spoke i runs from split i to inner corner i, and inner side i from inner corner i to i + 1.
    spoke_rows = [
        [(line_knots, np.array([slice_pieces[i][0], corners[i]])) for slice_pieces, corners in zip(pieces, inner_corners)] for i in range(4)
    ]
    inner_rows = [[(line_knots, np.array([corners[i], corners[(i + 1) % 4]])) for corners in inner_corners] for i in range(4)]

    with gmsh_session():
        arcs = [add_swept_surface(rows, spline_order, along, along_degree) for rows in arc_rows]
        spokes = [add_swept_surface(rows, 1, along, along_degree) for rows in spoke_rows]
        inner = [add_swept_surface(rows, 1, along, along_degree) for rows in inner_rows]
        caps = []
        for end in (0, -1):
            arc_curves = [add_bspline_curve(*arc_rows[i][end], spline_order) for i in range(4)]
            spoke_curves = [add_bspline_curve(*spoke_rows[i][end], 1) for i in range(4)]
            inner_curves = [add_bspline_curve(*inner_rows[i][end], 1) for i in range(4)]
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
        return export_brep()


def _slice_at(
    mesh: vtkPolyData,
    center: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    normal: np.ndarray,
    point_count: int,
) -> ContourSlice | None:
    """Return the contour nearest center in the plane there, or None if there is none."""
    loops = _closed_loops(mesh, center, normal)
    if not loops:
        return None
    # The plane can also cut the far limb of a C-shaped structure.
    loop = min(loops, key=lambda candidate: float(np.linalg.norm(candidate.mean(axis=0) - center)))

    offsets = loop - loop.mean(axis=0)
    u = offsets @ u_axis
    v = offsets @ v_axis
    signed_area = 0.5 * float(np.sum(u * np.roll(v, -1) - np.roll(u, -1) * v))
    if signed_area < 0:
        loop, u, v = loop[::-1], u[::-1], v[::-1]
    loop = np.roll(loop, -int(np.argmin(np.abs(np.arctan2(v, u)))), axis=0)

    return ContourSlice(resample_closed(loop, point_count), center, u_axis, v_axis, normal, abs(signed_area))


def _align_with_previous(slices: list[ContourSlice]) -> list[ContourSlice]:
    """Rotate each contour's point order to best match the slice before it.

    A loft joins contours by parameter, so a start point that drifts between
    slices mismatches their corners and makes the surface bulge between them.
    """
    aligned = [slices[0]]
    for contour in slices[1:]:
        target = _plane_coordinates(aligned[-1])
        current = _plane_coordinates(contour)
        costs = [np.sum((target - np.roll(current, -shift, axis=0)) ** 2) for shift in range(len(current))]
        aligned.append(replace(contour, points=np.roll(contour.points, -int(np.argmin(costs)), axis=0)))
    return aligned


def _follow_corners(slices: list[ContourSlice], count: int) -> list[ContourSlice]:
    """Resample aligned dense contours to count points each, with every tracked corner at the same point index on every slice.

    A loft joins contours at shared parameters, so the corners' ridges then
    follow lines of the loft's parameter around it rather than drifting
    across them. Each stretch between corners gets its share of count by
    its mean fraction of the perimeter, and each contour starts at its
    sharpest corner. A corner is pinned only on slices where it turns at
    least _MIN_CORNER_TURN; elsewhere, as where the rounded outer corners
    fade toward the horns, it lies where points are evenly spaced between
    the pinned corners on either side, since a stretch's spacing changes at
    a corner and the loft overshoots where that change meets a smooth
    stretch of the contour. Without corners, points are evenly spaced from
    the aligned start.
    """
    dense = len(slices[0].points)
    tracks, pinned = (array[:, : min(_MAX_CORNERS, count)] for array in _corner_tracks(slices))
    if not tracks.shape[1]:
        return [replace(contour, points=resample_closed(contour.points, count)) for contour in slices]
    # Order the tracks around the contour from the sharpest, the first, which starts every contour.
    order = np.argsort((tracks - tracks[:, :1]) % dense, axis=1)
    tracks, pinned = (np.take_along_axis(array, order, axis=1) for array in (tracks, pinned))
    pinned[:, 0] = True
    # Each stretch runs from its corner to the next around the contour, wrapping past the start; a lone corner's spans the whole.
    lengths = ((np.roll(tracks, -1, axis=1) - tracks - 1) % dense + 1) / dense
    shares = _largest_remainders(lengths.mean(axis=0), count)
    starts = np.concatenate(([0], np.cumsum(shares)))
    logger.debug(
        "Tracked {} corners through {} slices, with {} points between them, pinned on {} of their slices",
        tracks.shape[1],
        len(slices),
        shares.tolist(),
        pinned.sum(axis=0).tolist(),
    )
    resampled = []
    for contour, corners, strong in zip(slices, tracks, pinned, strict=True):
        # Pinned corners' fractions of the perimeter after the first, closing with the first one turn on.
        anchors = np.append(np.flatnonzero(strong), len(corners))
        positions = np.append(((corners - corners[0]) % dense) / dense, 1.0)[anchors]
        # Points sit half a spacing either side of each corner: an interpolating spline through a sharp corner's own point overshoots.
        fractions = corners[0] / dense + np.interp(np.arange(count) + 0.5, starts[anchors], positions)
        resampled.append(replace(contour, points=sample_closed(contour.points, fractions)))
    return resampled


def _clip_inner_edges(slices: list[ContourSlice], count: int, clip_height: float | None, clip_ratio: float) -> list[ContourSlice]:
    """Cut off each aligned dense contour's thin inner edge straight, and resample it to count points.

    The inner edge is the sharpest tracked corner. Each section's height at a
    distance r along the direction from its area centroid to that corner is
    the length of its chord across that direction there; the cut is the
    chord nearest the corner at least the clipping height, clip_height or
    clip_ratio times the mean of the sections' greatest heights, but at most
    _MAX_CLIP_HEIGHT of the section's own greatest height, so small sections
    toward the horns keep their shape. The cut's corners are exact points,
    shared by index across slices: the curved part runs from points[0] to
    points[straight_from], with the other tracked corners between them as
    _follow_corners places them, and the straight part's points are evenly
    spaced on the cut.
    """
    tracks, pinned = _corner_tracks(slices)
    if not tracks.shape[1]:
        logger.warning("Found no inner edge to clip, so the contours keep theirs")
        return _follow_corners(slices, count)
    sections = [_section_heights(contour, int(tip)) for contour, tip in zip(slices, tracks[:, 0], strict=True)]
    target = clip_height if clip_height is not None else clip_ratio * float(np.mean([heights.max() for _, heights, _ in sections]))
    cuts = []
    for contour, (stations, heights, frame), tip in zip(slices, sections, tracks[:, 0], strict=True):
        height = min(target, _MAX_CLIP_HEIGHT * float(heights.max()))
        cuts.append(_cut(contour, frame, float(stations[int(np.argmax(heights >= height))]), int(tip)))

    # Arc-length fractions of the other tracked corners along each curved part, kept where they stay well inside it on every slice.
    others = []
    for track in range(1, tracks.shape[1]):
        fractions = np.array([cut.fraction_of(int(index)) for cut, index in zip(cuts, tracks[:, track], strict=True)])
        if np.all((fractions > _CLIP_CORNER_MARGIN) & (fractions < 1 - _CLIP_CORNER_MARGIN)):
            others.append((fractions, pinned[:, track]))
    others.sort(key=lambda other: float(other[0].mean()))
    straight = np.mean([cut.straight_length / (cut.curved_length + cut.straight_length) for cut in cuts])
    straight_count = min(count - 2 - len(others), max(1, round(straight * count) - 1))
    curved_count = count - straight_count
    bounds = np.array([0.0, *(float(fractions.mean()) for fractions, _ in others), 1.0])
    shares = _largest_remainders(np.diff(bounds), curved_count - 1)
    starts = np.concatenate(([0], np.cumsum(shares)))
    logger.debug(
        "Clipped the inner edges at height {:.3g}, cutting {:.1%} of the perimeter on average, with {} corners between the cut's, "
        + "and {} points on the curved part and {} on the cut",
        target,
        straight,
        len(others),
        curved_count,
        straight_count,
    )
    clipped = []
    for index, (contour, cut) in enumerate(zip(slices, cuts, strict=True)):
        anchors = [0, *(position + 1 for position, (_, strong) in enumerate(others) if strong[index]), len(others) + 1]
        positions = np.array([0.0, *(fractions[index] for fractions, _ in others), 1.0])[anchors]
        # The cut's corners are exact points; the other corners lie between points, as in _follow_corners.
        points_at = np.array([starts[anchor] - (0.5 if 0 < anchor <= len(others) else 0.0) for anchor in anchors])
        curved = cut.along(np.interp(np.arange(curved_count), points_at, positions))
        on_cut = curved[-1] + np.outer(np.arange(1, straight_count + 1) / (straight_count + 1), curved[0] - curved[-1])
        points = np.vstack((curved, on_cut))
        clipped.append(replace(contour, points=points, area=_polygon_area(contour, points), straight_from=curved_count - 1))
    return clipped


@dataclass(frozen=True)
class _Cut:
    """A dense contour with its thin inner edge cut off: the curved part from one corner of the cut to the other."""

    curved: np.ndarray  # (M, 3) dense points from the cut's corner after the tip, around, to the one before it
    lengths: np.ndarray  # (M,) arc length from curved[0]
    indices: np.ndarray  # (M,) the dense contour's index of each point, -1 for the two corners
    straight_length: float

    @property
    def curved_length(self) -> float:
        return float(self.lengths[-1])

    def along(self, fractions: np.ndarray) -> np.ndarray:
        """Return points at fractions of the curved part's length, linearly between its dense points."""
        distances = np.asarray(fractions) * self.lengths[-1]
        return np.column_stack([np.interp(distances, self.lengths, self.curved[:, axis]) for axis in range(3)])

    def fraction_of(self, index: int) -> float:
        """Return the fraction of the curved part's length at which a dense point lies, or -1 if the cut removed it."""
        where = np.flatnonzero(self.indices == index)
        return float(self.lengths[where[0]] / self.lengths[-1]) if len(where) else -1.0


def _section_heights(contour: ContourSlice, tip: int) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return distances from the area centroid toward a contour's tip, from the tip inward, the cut's length at each, and the frame (centroid, direction, across).

    The cut at each distance is the chord _cut would make there, between
    the first crossings walking each way around the contour from the tip,
    so a far crossing of a blunt section cannot pass for a tall cut.
    """
    centroid = _section_centroid(contour)
    direction = contour.points[tip] - centroid
    direction -= (direction @ contour.normal) * contour.normal
    direction /= np.linalg.norm(direction)
    across = np.cross(contour.normal, direction)
    radial = (contour.points - centroid) @ direction
    stations = np.linspace(radial.max(), radial.min(), _CLIP_STATIONS)[1:-1]
    ends = [_crossings(contour, radial, tip, stations, step) for step in (1, -1)]
    return stations, np.linalg.norm(ends[0] - ends[1], axis=1), (centroid, direction, across)


def _crossings(contour: ContourSlice, radial: np.ndarray, tip: int, stations: np.ndarray, step: int) -> np.ndarray:
    """Return where walking around a contour from its tip, one way, first crosses below each station, (S, 3)."""
    count = len(contour.points)
    walk = (tip + step * np.arange(count)) % count
    below = radial[walk][None] < stations[:, None]
    below[:, 0] = False
    inside = np.argmax(below, axis=1)
    inside[~below.any(axis=1)] = count - 1
    previous = walk[inside - 1]
    current = walk[inside]
    weights = (stations - radial[previous]) / (radial[current] - radial[previous])
    return contour.points[previous] + weights[:, None] * (contour.points[current] - contour.points[previous])


def _cut(contour: ContourSlice, frame: tuple[np.ndarray, np.ndarray, np.ndarray], station: float, tip: int) -> _Cut:
    """Cut a dense contour across its frame's direction at station, keeping the side away from the tip."""
    centroid, direction, _ = frame
    count = len(contour.points)
    radial = (contour.points - centroid) @ direction
    # Walk from the tip each way to the first point inside the cut; the edges just before them cross it.
    after = next(step for step in range(1, count) if radial[(tip + step) % count] < station)
    before = next(step for step in range(1, count) if radial[(tip - step) % count] < station)
    first, last = (tip + after) % count, (tip - before) % count
    start, end = (_crossings(contour, radial, tip, np.array([station]), step)[0] for step in (1, -1))
    indices = np.concatenate(([-1], (first + np.arange((last - first) % count + 1)) % count, [-1]))
    curved = np.vstack((start, contour.points[indices[1:-1]], end))
    lengths = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(curved, axis=0), axis=1))))
    return _Cut(curved, lengths, indices, float(np.linalg.norm(start - end)))


def _polygon_area(contour: ContourSlice, points: np.ndarray) -> float:
    """Return the area a closed polygon of points encloses in a contour's plane."""
    offsets = points - contour.center
    u, v = offsets @ contour.u_axis, offsets @ contour.v_axis
    return abs(0.5 * float(np.sum(u * np.roll(v, -1) - np.roll(u, -1) * v)))


def _corner_tracks(slices: list[ContourSlice]) -> tuple[np.ndarray, np.ndarray]:
    """Return (slices, K) dense point indices of the corners tracked through every slice, sharpest track first, and where each turns at least _MIN_CORNER_TURN.

    Each slice's turning at each point is summed over a Gaussian window,
    positive where the contour turns counterclockwise about its normal.
    Each track is the path through the slices maximizing its turning less
    a penalty on drift between slices (Viterbi); points near a found track
    are excluded from later ones. Tracks are kept while their turning
    averages at least _MIN_CORNER_TURN and their order around the contour
    stays the same on every slice.
    """
    dense = len(slices[0].points)
    width = _CORNER_WINDOW * dense
    turning = np.array([gaussian_filter1d(_signed_turns(contour), width, mode="wrap") * width * math.sqrt(2 * math.pi) for contour in slices])
    offsets = np.arange(dense)
    cyclic = np.abs(offsets[:, None] - offsets[None]) / dense
    cyclic = np.minimum(cyclic, 1.0 - cyclic)
    drift_cost = _CORNER_DRIFT_WEIGHT * (cyclic / _CORNER_DRIFT) ** 2
    available = np.ones_like(turning, dtype=bool)
    tracks: list[np.ndarray] = []
    while len(tracks) < _MAX_CORNERS:
        score = np.where(available, turning, -np.inf)
        best, back = score[0], np.zeros((len(slices), dense), dtype=np.int64)
        for index in range(1, len(slices)):
            candidates = best[:, None] - drift_cost
            back[index] = np.argmax(candidates, axis=0)
            best = score[index] + candidates[back[index], offsets]
        if not np.isfinite(best).any():
            break
        path = np.empty(len(slices), dtype=np.int64)
        path[-1] = int(np.argmax(best))
        for index in range(len(slices) - 1, 0, -1):
            path[index - 1] = back[index, path[index]]
        if turning[np.arange(len(slices)), path].mean() < _MIN_CORNER_TURN:
            break
        candidate = [*tracks, path]
        relative = (np.column_stack(candidate) - candidate[0][:, None]) % dense
        if len({tuple(np.argsort(row)) for row in relative}) > 1:
            break
        tracks = candidate
        available &= cyclic[path] > _CORNER_SEPARATION
    if not tracks:
        return np.zeros((len(slices), 0), dtype=np.int64), np.zeros((len(slices), 0), dtype=bool)
    corners = np.column_stack(tracks)
    return corners, turning[np.arange(len(slices))[:, None], corners] >= _MIN_CORNER_TURN


def _signed_turns(contour: ContourSlice) -> np.ndarray:
    """Return the angle a closed contour turns at each point, positive counterclockwise about its normal."""
    incoming = contour.points - np.roll(contour.points, 1, axis=0)
    outgoing = np.roll(incoming, -1, axis=0)
    return np.arctan2(np.cross(incoming, outgoing) @ contour.normal, np.einsum("ij,ij->i", incoming, outgoing))


def _largest_remainders(fractions: np.ndarray, count: int) -> np.ndarray:
    """Split count into whole shares, at least 1 each, proportional to fractions."""
    extra = count - len(fractions)
    ideal = fractions / fractions.sum() * extra
    shares = np.floor(ideal).astype(np.int64)
    shares[np.argsort(shares - ideal)[: extra - int(shares.sum())]] += 1
    return shares + 1


def _plane_coordinates(contour: ContourSlice) -> np.ndarray:
    offsets = contour.points - contour.center
    return np.column_stack((offsets @ contour.u_axis, offsets @ contour.v_axis))


def _closed_loops(mesh: vtkPolyData, origin: np.ndarray, normal: np.ndarray) -> list[np.ndarray]:
    plane = vtkPlane()
    plane.SetOrigin(origin.tolist())
    plane.SetNormal(normal.tolist())

    cutter = vtkCutter()
    cutter.SetInputData(mesh)
    cutter.SetCutFunction(plane)

    stripper = vtkStripper()
    stripper.SetInputConnection(cutter.GetOutputPort())
    stripper.JoinContiguousSegmentsOn()
    stripper.SetMaximumLength(_MAX_CONTOUR_SEGMENTS)
    stripper.Update()

    output = stripper.GetOutput()
    if output.GetNumberOfPoints() == 0:
        return []
    points = vtk_to_numpy(output.GetPoints().GetData()).astype(float)
    lines = output.GetLines()
    connectivity = vtk_to_numpy(lines.GetConnectivityArray())
    offsets = vtk_to_numpy(lines.GetOffsetsArray())

    loops = []
    for start, end in pairwise(offsets):
        ids = connectivity[start:end]
        if len(ids) > 3 and ids[0] == ids[-1]:
            loops.append(points[ids[:-1]])
    return loops


def _physical_points(image: sitk.Image, zyx_indices: np.ndarray) -> np.ndarray:
    direction = np.reshape(image.GetDirection(), (3, 3))
    offsets = direction @ (zyx_indices[:, ::-1] * image.GetSpacing()).T
    return image.GetOrigin() + offsets.T


def _skeleton_path(image: sitk.Image, mask: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, float]:
    """Return ordered points along the 2D skeleton and the label's mean half-width.

    The skeleton forks toward the label's corners at each end, so the path is
    shortened by one half-width at both ends. Section refinement then grows the
    curve back out to the label's true tips.
    """
    depth_axis = _thinnest_array_axis(image, points)
    projection = mask.any(axis=depth_axis)
    # Thinning treats the image edge as an open end, so a label cut off by the
    # edge would skeletonize into separate branches running off it. A border of
    # background keeps the skeleton inside the label and connected.
    padded = np.pad(projection, 1).astype(np.uint8)
    skeleton = sitk.GetArrayFromImage(sitk.BinaryThinning(sitk.GetImageFromArray(padded)))
    pixels = np.argwhere(skeleton) - 1

    coordinates = np.arange(mask.shape[depth_axis]).reshape([-1 if axis == depth_axis else 1 for axis in range(3)])
    pixel_key = tuple(pixels.T)
    depth_sum = (mask * coordinates).sum(axis=depth_axis)[pixel_key]
    depths = depth_sum / mask.sum(axis=depth_axis)[pixel_key]
    nodes = _physical_points(image, np.insert(pixels.astype(float), depth_axis, depths, axis=1))
    path = nodes[_longest_path(pixels, nodes)]

    lengths = _cumulative_lengths(path)
    pixel_area = np.prod(image.GetSpacing()) / image.GetSpacing()[2 - depth_axis]
    half_width = float(projection.sum() * pixel_area / (2 * lengths[-1]))

    keep = (lengths >= half_width) & (lengths <= lengths[-1] - half_width)
    return (path[keep] if keep.sum() >= 4 else path), half_width


def _thinnest_array_axis(image: sitk.Image, points: np.ndarray) -> int:
    """Return the NumPy axis closest to the direction in which the points are thinnest."""
    _, eigenvectors = np.linalg.eigh(np.cov(points.T))
    direction = np.reshape(image.GetDirection(), (3, 3))
    image_axis = int(np.argmax(np.abs(direction.T @ eigenvectors[:, 0])))
    return 2 - image_axis


def _longest_path(pixels: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Return node indices of the longest path in the largest skeleton component."""
    pairs = KDTree(pixels).query_pairs(1.5, output_type="ndarray")
    if len(pairs) == 0:
        raise ValueError("Label is too small to fit a curve")
    weights = np.linalg.norm(nodes[pairs[:, 0]] - nodes[pairs[:, 1]], axis=1)
    graph = coo_matrix((weights, (pairs[:, 0], pairs[:, 1])), shape=(len(nodes), len(nodes)))

    _, components = connected_components(graph, directed=False)
    start = int(np.argmax(components == np.bincount(components).argmax()))

    distances = dijkstra(graph, directed=False, indices=start)
    first_end = int(np.argmax(np.where(np.isfinite(distances), distances, -1.0)))
    distances, predecessors = dijkstra(graph, directed=False, indices=first_end, return_predecessors=True)
    node = int(np.argmax(np.where(np.isfinite(distances), distances, -1.0)))

    path = [node]
    while predecessors[node] >= 0:
        node = int(predecessors[node])
        path.append(node)
    return np.array(path)


def _cumulative_lengths(curve: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))))


def _fit(samples: np.ndarray, rms_tolerance: float) -> BSpline:
    if len(samples) < 4:
        raise ValueError("Label is too small to fit a curve")
    spline, _ = make_splprep(list(samples.T), k=3, s=len(samples) * rms_tolerance**2)  # pyright: ignore[reportArgumentType]
    # make_splprep evaluates to shape (3, N); rebuild so points are (N, 3) rows.
    return BSpline(spline.t, spline.c, spline.k)


def _section_centroids(spline: BSpline, points: np.ndarray, step: float) -> np.ndarray:
    """Return centroids of slabs of the label normal to the curve at even arc-length steps."""
    stations_u = _arc_length_stations(spline, step)
    tangents = spline(stations_u, nu=1)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    stations, tangents = _extend_past_ends(spline(stations_u), tangents, step)

    # A plane through a curved structure can cut it twice, so each slab only
    # takes points whose nearest station is close to its own.
    _, nearest = KDTree(stations).query(points)
    counts = np.zeros(len(stations), dtype=int)
    centroids = np.zeros_like(stations)
    for index, (station, tangent) in enumerate(zip(stations, tangents)):
        in_slab = np.abs((points - station) @ tangent) <= step / 2
        on_this_side = np.abs(nearest - index) <= _STATION_NEIGHBORS
        section = points[in_slab & on_this_side]
        counts[index] = len(section)
        if len(section) > 0:
            centroids[index] = section.mean(axis=0)

    first, last = _complete_sections(counts)
    return centroids[first : last + 1]


def _arc_lengths(spline: BSpline) -> tuple[np.ndarray, np.ndarray]:
    """Return sampled parameters and the arc length from u = 0 at each."""
    u = np.linspace(0.0, 1.0, _ARC_LENGTH_SAMPLES)
    return u, _cumulative_lengths(spline(u))


def _arc_length_stations(spline: BSpline, step: float) -> np.ndarray:
    u, lengths = _arc_lengths(spline)
    count = max(int(lengths[-1] / step), 4)
    return np.interp(np.linspace(0.0, lengths[-1], count), lengths, u)


def _extend_past_ends(stations: np.ndarray, tangents: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Add straight runs of stations beyond both ends so the curve can reach the tips."""
    distances = step * np.arange(1, _END_EXTENSION_STATIONS + 1)[:, np.newaxis]
    before = stations[0] - distances[::-1] * tangents[0]
    after = stations[-1] + distances * tangents[-1]
    extended_tangents = np.vstack(
        (
            np.tile(tangents[0], (_END_EXTENSION_STATIONS, 1)),
            tangents,
            np.tile(tangents[-1], (_END_EXTENSION_STATIONS, 1)),
        )
    )
    return np.vstack((before, stations, after)), extended_tangents


def _complete_sections(counts: np.ndarray) -> tuple[int, int]:
    """Return the inclusive station range before section size collapses at either tip."""
    first = last = int(np.argmax(counts))
    while first > 0 and counts[first - 1] >= _MIN_SECTION_RATIO * counts[first]:
        first -= 1
    while last < len(counts) - 1 and counts[last + 1] >= _MIN_SECTION_RATIO * counts[last]:
        last += 1
    return first, last


def _require_loftable(slices: list[ContourSlice], spline_order: int) -> None:
    if len(slices) < 2:
        raise ValueError("Lofting needs at least 2 contour slices")
    if not 1 <= spline_order <= MAX_SPLINE_ORDER:
        raise ValueError(f"spline_order must be between 1 and {MAX_SPLINE_ORDER}, got {spline_order}")
    if any(len(contour.points) <= spline_order for contour in slices):
        raise ValueError(f"Contours need more than {spline_order} points for spline_order {spline_order}")


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


def _section_centroid(contour: ContourSlice) -> np.ndarray:
    """Return the area centroid of a contour's polygon, in physical coordinates."""
    offsets = contour.points - contour.center
    u, v = offsets @ contour.u_axis, offsets @ contour.v_axis
    cross = u * np.roll(v, -1) - np.roll(u, -1) * v
    area = cross.sum() / 2
    centroid_u = np.sum((u + np.roll(u, -1)) * cross) / (6 * area)
    centroid_v = np.sum((v + np.roll(v, -1)) * cross) / (6 * area)
    return contour.center + centroid_u * contour.u_axis + centroid_v * contour.v_axis


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
    poles, knots = periodic_poles_and_knots(spline, len(contour.points))
    height = _DEFAULT_CAP_OFFSET_RADII * float(np.sqrt(contour.area / np.pi)) if offset is None else offset
    lift = height * outward
    apex = gmsh.model.occ.addPoint(*(contour.center + lift).tolist())
    rows = [*add_closed_points(poles), *add_closed_points(poles + lift), *[apex] * (len(poles) + 1)]
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
