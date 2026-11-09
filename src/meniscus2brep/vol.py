"""Operations on segmented LabelMaps."""

import numpy as np
import SimpleITK as sitk
from scipy.interpolate import BSpline, make_splprep
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import KDTree

_MIN_LABEL_POINTS = 4
_REFINEMENT_ITERATIONS = 3
_ARC_LENGTH_SAMPLES = 2000
_STATION_NEIGHBORS = 2
_END_EXTENSION_STATIONS = 4
_MIN_SECTION_RATIO = 0.5
_SKELETON_TOLERANCE_VOXELS = 0.5
_CENTROID_TOLERANCE_VOXELS = 0.25
_CENTROID_TOLERANCE_HALF_WIDTHS = 0.05


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
    if label_map.GetDimension() != 3 or label_map.GetPixelID() != sitk.sitkUInt8:
        raise ValueError(f"LabelMap must be 3D sitkUInt8, got {label_map.GetDimension()}D {label_map.GetPixelIDTypeAsString()}")

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
    spline, _ = make_splprep(list(samples.T), k=3, s=round(len(samples) * rms_tolerance**2))
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


def _arc_length_stations(spline: BSpline, step: float) -> np.ndarray:
    u = np.linspace(0.0, 1.0, _ARC_LENGTH_SAMPLES)
    lengths = _cumulative_lengths(spline(u))
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
