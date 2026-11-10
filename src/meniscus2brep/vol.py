"""Operations on segmented LabelMaps."""

import math
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import SimpleITK as sitk
from scipy.interpolate import BSpline, make_splprep
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import KDTree
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonComputationalGeometry import vtkKochanekSpline, vtkParametricSpline
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPlane, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkCutter, vtkFlyingEdges3D, vtkStripper

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


def centerline(label_map: sitk.Image, label: int = 1) -> BSpline:
    """Fit a smooth curve through the centers of a label's cross-sections.

    BinaryThinning only thins in 2D, so the label is projected along the image
    axis closest to its thinnest direction and the longest path through that
    skeleton gives a first estimate of the curve. The curve is then repeatedly
    moved onto the centroids of the label's cross-sections normal to it.

    Args:
        label_map: 3D sitkUInt8 LabelMap.
        label: Value of the structure to fit.

    Returns:
        A cubic B-spline over u in [0, 1]. Evaluating it at N parameters gives
        an (N, 3) array of physical points. Which end of the structure is
        u = 0 is unspecified.

    Raises:
        ValueError: If the image is not a 3D sitkUInt8 LabelMap, or the label
            is absent or too small to fit.
    """
    _require_3d_label_map(label_map)

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
    return spline


@dataclass(frozen=True, eq=False)
class ContourSlice:
    """An ordered closed contour of a surface in a plane normal to a curve.

    Attributes:
        points: (N, 3) physical points, evenly spaced around the contour and
            counterclockwise about normal. The closing point is not repeated.
            The first point lies in the +u_axis direction from the contour's
            center, so corresponding points line up from slice to slice.
        center: Point on the curve where the plane is centered.
        u_axis: Unit vector in the plane.
        v_axis: Unit vector in the plane. (u_axis, v_axis, normal) is right-handed.
        normal: Unit tangent of the curve at center.
        area: Area enclosed by the contour.
    """

    points: np.ndarray
    center: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray
    normal: np.ndarray
    area: float


def signed_distance_field(label_map: sitk.Image, label: int = 1) -> sitk.Image:
    """Compute the signed distance to a label's boundary, in physical units.

    Args:
        label_map: 3D sitkUInt8 LabelMap.
        label: Value of the structure.

    Returns:
        A float image that is negative inside the label and positive outside.
        Its zero level set lies on the voxel edges of the label's boundary.

    Raises:
        ValueError: If the image is not a 3D sitkUInt8 LabelMap or the label
            is absent.
    """
    _require_3d_label_map(label_map)
    mask = sitk.BinaryThreshold(label_map, label, label, 1, 0)
    if not sitk.GetArrayViewFromImage(mask).any():
        raise ValueError(f"Label {label} is not present in the image")

    # One map alone is zero on the centers of the boundary voxels, half a voxel
    # inside the label's edge. Averaging with the inverse's map centers it.
    inside = _maurer_distance(mask)
    outside = _maurer_distance(sitk.BinaryNot(mask))
    return (inside - outside) * 0.5


def extract_isocontour(distance_field: sitk.Image, level: float = 0.0) -> vtkPolyData:
    """Extract a triangle mesh at a given distance from a label's boundary.

    Args:
        distance_field: Float image from signed_distance_field.
        level: Distance of the surface from the label boundary. Positive
            values lie outside the label.

    Returns:
        A closed triangle mesh in physical coordinates.

    Raises:
        ValueError: If the distance field never reaches level.
    """
    values = sitk.GetArrayViewFromImage(distance_field)
    if not values.min() <= level <= values.max():
        raise ValueError(f"The distance field has no surface at level {level}")

    # Padding with an outside value closes the surface where the structure
    # touches the edge of the image.
    outside = float(values.max()) + max(distance_field.GetSpacing())
    padded = sitk.ConstantPad(distance_field, (1, 1, 1), (1, 1, 1), outside)

    flying_edges = vtkFlyingEdges3D()
    flying_edges.SetInputData(_to_vtk_image(padded))
    flying_edges.SetValue(0, level)
    flying_edges.ComputeNormalsOff()
    flying_edges.ComputeGradientsOff()
    flying_edges.ComputeScalarsOff()
    flying_edges.Update()

    return flying_edges.GetOutput()


def slice_isocontour(mesh: vtkPolyData, curve: BSpline, spacing: float, points_per_slice: int = 64) -> list[ContourSlice]:
    """Slice a mesh with planes normal to a curve at uniform arc-length spacing.

    Slices are spaced exactly spacing apart along the curve, with any remainder
    split equally between its two ends, and are ordered along the curve from
    u = 0. Each cut is resampled to evenly spaced ordered points with a closed
    Kochanek spline. Slices are removed from either end of the curve until one
    encloses at least 20% of the average area.

    In-plane axes are defined from the curve's best-fit plane, so they do not
    twist along a roughly planar curve.

    Args:
        mesh: Closed surface from extract_isocontour.
        curve: Curve from centerline.
        spacing: Arc-length distance between slices, in physical units.
        points_per_slice: Number of points on each contour.

    Returns:
        The retained ContourSlices.

    Raises:
        ValueError: If spacing is not positive, points_per_slice is below 3,
            or a retained plane does not cut the mesh in a closed contour.
    """
    if spacing <= 0:
        raise ValueError("Slice spacing must be positive")
    if points_per_slice < 3:
        raise ValueError("Slices need at least 3 points")

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
        _slice_at(mesh, center, u_axis, v_axis, normal, points_per_slice)
        for center, u_axis, v_axis, normal in zip(centers, u_axes, v_axes, normals)
    ]
    areas = np.array([0.0 if contour is None else contour.area for contour in slices])
    if not areas.any():
        raise ValueError("The curve's planes do not cut the mesh in any closed contour")

    keep = np.flatnonzero(areas >= _MIN_END_AREA_RATIO * areas.mean())
    retained = slices[keep[0] : keep[-1] + 1]
    if any(contour is None for contour in retained):
        raise ValueError("A plane along the curve does not cut the mesh in a closed contour")
    return retained  # pyright: ignore[reportReturnType]


def _require_3d_label_map(label_map: sitk.Image) -> None:
    if label_map.GetDimension() != 3 or label_map.GetPixelID() != sitk.sitkUInt8:
        raise ValueError(f"LabelMap must be 3D sitkUInt8, got {label_map.GetDimension()}D {label_map.GetPixelIDTypeAsString()}")


def _maurer_distance(mask: sitk.Image) -> sitk.Image:
    return sitk.SignedMaurerDistanceMap(mask, insideIsPositive=False, squaredDistance=False, useImageSpacing=True)


def _to_vtk_image(image: sitk.Image) -> vtkImageData:
    result = vtkImageData()
    result.SetDimensions(image.GetSize())
    result.SetSpacing(image.GetSpacing())
    result.SetOrigin(image.GetOrigin())
    result.SetDirectionMatrix(image.GetDirection())
    # SimpleITK's z, y, x array flattens to VTK's x-fastest point order.
    result.GetPointData().SetScalars(numpy_to_vtk(sitk.GetArrayFromImage(image).ravel(), deep=True))
    return result


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

    return ContourSlice(_resample_closed(loop, point_count), center, u_axis, v_axis, normal, abs(signed_area))


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


def _resample_closed(loop: np.ndarray, count: int) -> np.ndarray:
    """Return count points evenly spaced in length around a closed Kochanek spline."""
    points = vtkPoints()
    points.SetData(numpy_to_vtk(loop, deep=True))

    spline = vtkParametricSpline()
    spline.SetXSpline(vtkKochanekSpline())
    spline.SetYSpline(vtkKochanekSpline())
    spline.SetZSpline(vtkKochanekSpline())
    spline.SetPoints(points)
    spline.ClosedOn()
    spline.ParameterizeByLengthOn()

    result = np.empty((count, 3))
    point = [0.0, 0.0, 0.0]
    derivatives = [0.0] * 9
    for index in range(count):
        spline.Evaluate([index / count, 0.0, 0.0], point, derivatives)
        result[index] = point
    return result


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
    skeleton = sitk.GetArrayFromImage(sitk.BinaryThinning(sitk.GetImageFromArray(projection.astype(np.uint8))))
    pixels = np.argwhere(skeleton)

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
