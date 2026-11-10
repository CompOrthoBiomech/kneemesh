from itertools import pairwise

import numpy as np
import pytest
import SimpleITK as sitk
from scipy.interpolate import BSpline
from scipy.spatial import KDTree
from vtkmodules.vtkFiltersCore import vtkFeatureEdges, vtkMassProperties

from meniscus2brep.vol import centerline, extract_isocontour, signed_distance_field, slice_isocontour

# Area centroid of the voxelized fixture's cross-section. It differs slightly
# from the ideal triangle (16.67, -1.33) because the fixture's bottom slice lies
# exactly on the triangle's base and counts as a whole voxel layer.
SECTION_CENTER_RADIUS = 16.43
SECTION_CENTER_Z = -1.67


def sample(spline, count=1001):
    return spline(np.linspace(0.0, 1.0, count))


def polar(points):
    angles = np.degrees(np.unwrap(np.arctan2(points[:, 1], points[:, 0])))
    return np.hypot(points[:, 0], points[:, 1]), angles


def test_centerline_is_cubic_spline_of_physical_points(meniscus_centerline):
    assert isinstance(meniscus_centerline, BSpline)
    assert meniscus_centerline.k == 3
    assert sample(meniscus_centerline, 11).shape == (11, 3)


def test_centerline_passes_through_cross_section_centers(meniscus_centerline):
    points = sample(meniscus_centerline)
    radius, _ = polar(points)

    assert radius == pytest.approx(np.full_like(radius, SECTION_CENTER_RADIUS), abs=0.5)
    assert points[:, 2] == pytest.approx(np.full(len(points), SECTION_CENTER_Z), abs=0.25)


def test_centerline_spans_the_full_sweep(meniscus_centerline):
    _, angles = polar(sample(meniscus_centerline))

    assert angles.min() == pytest.approx(-135.0, abs=5.0)
    assert angles.max() == pytest.approx(135.0, abs=5.0)


def test_centerline_is_smooth(meniscus_centerline):
    u = np.linspace(0.0, 1.0, 1001)

    tangents = meniscus_centerline(u, nu=1)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    turn_degrees = np.degrees(np.arccos(np.clip((tangents[1:] * tangents[:-1]).sum(axis=1), -1, 1)))

    assert turn_degrees.max() < 1.0


def test_centerline_is_independent_of_image_axis_order(meniscus_label, meniscus_centerline):
    permuted = sitk.PermuteAxes(meniscus_label, (2, 0, 1))

    expected = sample(meniscus_centerline)
    result = sample(centerline(permuted))

    distances, _ = KDTree(expected).query(result)
    assert distances.max() < 0.1


def test_centerline_selects_requested_label(meniscus_label, meniscus_centerline):
    expected = sample(meniscus_centerline)
    relabeled = meniscus_label * 7
    relabeled[0, 0, 0] = 1

    result = sample(centerline(relabeled, label=7))

    distances, _ = KDTree(expected).query(result)
    assert distances.max() < 0.1


def test_centerline_rejects_absent_label(meniscus_label):
    with pytest.raises(ValueError, match="not present"):
        centerline(meniscus_label, label=2)


@pytest.mark.parametrize(("pixel_id", "label"), [(sitk.sitkUInt16, 1000), (sitk.sitkInt16, -300)])
def test_centerline_accepts_16_bit_label_maps(meniscus_label, meniscus_centerline, pixel_id, label):
    # Labels outside the 8-bit range show the full pixel type is used.
    relabeled = sitk.Cast(meniscus_label, pixel_id) * label

    distances, _ = KDTree(sample(meniscus_centerline)).query(sample(centerline(relabeled, label=label)))

    assert distances.max() < 0.1


@pytest.mark.parametrize("pixel_id", [sitk.sitkInt8, sitk.sitkUInt32, sitk.sitkInt32, sitk.sitkFloat32])
def test_centerline_rejects_other_pixel_types(meniscus_label, pixel_id):
    with pytest.raises(ValueError, match="3D sitkUInt8, sitkUInt16, or sitkInt16"):
        centerline(sitk.Cast(meniscus_label, pixel_id))


def test_centerline_rejects_2d_image():
    with pytest.raises(ValueError, match="3D sitkUInt8"):
        centerline(sitk.Image(8, 8, sitk.sitkUInt8))


def test_centerline_rejects_tiny_label():
    image = sitk.Image(8, 8, 8, sitk.sitkUInt8)
    image[1, 1, 1] = 1

    with pytest.raises(ValueError, match="too small"):
        centerline(image)


def as_array(image):
    return sitk.GetArrayFromImage(image)


def mesh_volume(mesh):
    properties = vtkMassProperties()
    properties.SetInputData(mesh)
    properties.Update()
    return properties.GetVolume()


def open_edge_count(mesh):
    edges = vtkFeatureEdges()
    edges.SetInputData(mesh)
    edges.BoundaryEdgesOn()
    edges.NonManifoldEdgesOn()
    edges.FeatureEdgesOff()
    edges.ManifoldEdgesOff()
    edges.Update()
    return edges.GetOutput().GetNumberOfCells()


def straight_line(start, end):
    return BSpline([0.0, 0.0, 1.0, 1.0], np.array([start, end], dtype=float), 1)


def in_plane(contour_slice):
    offsets = contour_slice.points - contour_slice.points.mean(axis=0)
    return offsets @ contour_slice.u_axis, offsets @ contour_slice.v_axis


def test_unsmoothed_distance_field_is_negative_exactly_inside_label(meniscus_label):
    field = signed_distance_field(meniscus_label, sigma=0.0)

    assert np.array_equal(as_array(field) < 0, as_array(meniscus_label) == 1)


def test_distance_field_zero_lies_on_voxel_edge():
    image = sitk.Image(20, 6, 6, sitk.sitkUInt8)
    image.SetSpacing((0.5, 1.0, 1.0))
    for x in range(5, 15):
        for y in range(6):
            for z in range(6):
                image[x, y, z] = 1

    field = signed_distance_field(image, sigma=0.0)

    assert field[4, 3, 3] == pytest.approx(0.25)
    assert field[5, 3, 3] == pytest.approx(-0.25)


def test_distance_field_default_sigma_spans_three_voxels_per_axis(meniscus_label, meniscus_distance_field):
    unsmoothed = signed_distance_field(meniscus_label, sigma=0.0)
    sigmas = [3.0 / 3.92 * size for size in meniscus_label.GetSpacing()]

    assert sitk.Hash(meniscus_distance_field) == sitk.Hash(sitk.SmoothingRecursiveGaussian(unsmoothed, sigmas))


def test_distance_field_uses_explicit_sigma_on_every_axis(meniscus_label):
    unsmoothed = signed_distance_field(meniscus_label, sigma=0.0)

    result = signed_distance_field(meniscus_label, sigma=1.5)

    assert sitk.Hash(result) == sitk.Hash(sitk.SmoothingRecursiveGaussian(unsmoothed, [1.5, 1.5, 1.5]))


def test_distance_field_rejects_negative_sigma(meniscus_label):
    with pytest.raises(ValueError, match="sigma"):
        signed_distance_field(meniscus_label, sigma=-1.0)


def test_smoothing_removes_the_staircase_from_contours(meniscus_label, meniscus_centerline):
    def roughness(sigma):
        field = signed_distance_field(meniscus_label, sigma=sigma)
        slices = slice_isocontour(extract_isocontour(field), meniscus_centerline, spacing=5.0, points_per_slice=64)
        # A contour's bending is highest at its real corners, so the median measures the steps between them.
        second_differences = [
            np.linalg.norm(np.roll(s.points, -1, axis=0) - 2 * s.points + np.roll(s.points, 1, axis=0), axis=1) for s in slices
        ]
        return np.median(np.concatenate(second_differences))

    assert roughness(None) < 0.5 * roughness(0.0)


def test_distance_field_selects_requested_label(meniscus_label, meniscus_distance_field):
    relabeled = meniscus_label * 7

    assert sitk.Hash(signed_distance_field(relabeled, label=7)) == sitk.Hash(meniscus_distance_field)


def test_distance_field_rejects_absent_label(meniscus_label):
    with pytest.raises(ValueError, match="not present"):
        signed_distance_field(meniscus_label, label=2)


@pytest.mark.parametrize(("pixel_id", "label"), [(sitk.sitkUInt16, 1000), (sitk.sitkInt16, -300)])
def test_distance_field_accepts_16_bit_label_maps(meniscus_label, meniscus_distance_field, pixel_id, label):
    relabeled = sitk.Cast(meniscus_label, pixel_id) * label

    assert sitk.Hash(signed_distance_field(relabeled, label=label)) == sitk.Hash(meniscus_distance_field)


@pytest.mark.parametrize("pixel_id", [sitk.sitkInt8, sitk.sitkUInt32, sitk.sitkFloat32])
def test_distance_field_rejects_other_pixel_types(meniscus_label, pixel_id):
    with pytest.raises(ValueError, match="3D sitkUInt8, sitkUInt16, or sitkInt16"):
        signed_distance_field(sitk.Cast(meniscus_label, pixel_id))


def test_isocontour_is_closed_mesh_with_label_volume(meniscus_label, meniscus_mesh):
    voxel_volume = np.prod(meniscus_label.GetSpacing()) * as_array(meniscus_label).sum()
    unsmoothed = extract_isocontour(signed_distance_field(meniscus_label, sigma=0.0))

    assert open_edge_count(meniscus_mesh) == 0
    assert mesh_volume(unsmoothed) == pytest.approx(voxel_volume, rel=0.03)
    # Smoothing rounds the fixture's sharp rim, removing a few percent of its volume.
    assert mesh_volume(meniscus_mesh) == pytest.approx(voxel_volume, rel=0.12)
    assert mesh_volume(meniscus_mesh) < mesh_volume(unsmoothed)


def test_isocontour_level_offsets_the_surface(meniscus_distance_field, meniscus_mesh):
    outer = extract_isocontour(meniscus_distance_field, level=1.0)
    inner = extract_isocontour(meniscus_distance_field, level=-1.0)

    assert outer.GetBounds()[1] == pytest.approx(meniscus_mesh.GetBounds()[1] + 1.0, abs=0.3)
    assert mesh_volume(inner) < mesh_volume(meniscus_mesh) < mesh_volume(outer)


def test_isocontour_is_in_physical_coordinates_for_oriented_image(meniscus_label, meniscus_mesh):
    permuted = sitk.PermuteAxes(meniscus_label, (2, 0, 1))

    mesh = extract_isocontour(signed_distance_field(permuted))

    assert mesh.GetBounds() == pytest.approx(meniscus_mesh.GetBounds(), abs=1e-3)


def test_isocontour_closes_where_label_touches_image_edge():
    image = sitk.Image(8, 8, 8, sitk.sitkUInt8)
    for x in range(4):
        for y in range(2, 6):
            for z in range(2, 6):
                image[x, y, z] = 1

    assert open_edge_count(extract_isocontour(signed_distance_field(image))) == 0


@pytest.mark.parametrize("level", [1000.0, -1000.0])
def test_isocontour_rejects_level_outside_field(meniscus_distance_field, level):
    with pytest.raises(ValueError, match="no surface"):
        extract_isocontour(meniscus_distance_field, level=level)


def test_slices_default_to_32_points(meniscus_slices):
    assert all(len(contour_slice.points) == 32 for contour_slice in meniscus_slices)


@pytest.mark.parametrize("count", [16, 64])
def test_slice_points_are_evenly_spaced_around_a_closed_contour(meniscus_mesh, meniscus_centerline, count):
    slices = slice_isocontour(meniscus_mesh, meniscus_centerline, spacing=20.0, points_per_slice=count)

    for contour_slice in slices:
        assert contour_slice.points.shape == (count, 3)
        gaps = np.linalg.norm(contour_slice.points - np.roll(contour_slice.points, -1, axis=0), axis=1)
        assert gaps.max() < 1.5 * np.median(gaps)


def test_slice_points_lie_in_their_plane(meniscus_slices):
    for contour_slice in meniscus_slices:
        assert np.abs((contour_slice.points - contour_slice.center) @ contour_slice.normal).max() < 1e-4


def test_slice_points_wind_counterclockwise_from_positive_u(meniscus_slices):
    for contour_slice in meniscus_slices:
        u, v = in_plane(contour_slice)
        assert np.sum(u * np.roll(v, -1) - np.roll(u, -1) * v) > 0
        assert abs(np.degrees(np.arctan2(v[0], u[0]))) < 30.0


def test_slice_points_line_up_with_the_previous_slice(meniscus_slices):
    for previous, contour_slice in pairwise(meniscus_slices):
        target = np.column_stack(in_plane(previous))
        current = np.column_stack(in_plane(contour_slice))
        costs = [np.sum((target - np.roll(current, -shift, axis=0)) ** 2) for shift in range(len(current))]
        # The centroids subtracted here differ slightly from the slice centers
        # used for alignment, so allow the optimum to land on a neighbor.
        assert min(np.argmin(costs), len(costs) - np.argmin(costs)) <= 1


def test_slice_contours_lie_on_the_label_boundary(meniscus_slices, meniscus_distance_field):
    voxel = max(meniscus_distance_field.GetSpacing())
    for contour_slice in meniscus_slices:
        for point in contour_slice.points:
            index = meniscus_distance_field.TransformPhysicalPointToIndex(point.tolist())
            assert abs(meniscus_distance_field[index]) <= voxel


def test_slices_take_the_contour_around_the_curve(meniscus_slices):
    for contour_slice in meniscus_slices:
        assert np.linalg.norm(contour_slice.points - contour_slice.center, axis=1).max() < 9.0


def test_slice_areas_match_cross_section(meniscus_slices):
    for contour_slice in meniscus_slices:
        # About 45 mm^2 when voxelized, less the rim corners that smoothing rounds.
        assert contour_slice.area == pytest.approx(42.0, abs=2.0)
        u, v = in_plane(contour_slice)
        assert 0.5 * np.sum(u * np.roll(v, -1) - np.roll(u, -1) * v) == pytest.approx(contour_slice.area, rel=0.05)


@pytest.mark.parametrize("spacing", [3.0, 7.0])
def test_slices_are_uniformly_spaced_on_the_curve(meniscus_mesh, meniscus_centerline, spacing):
    slices = slice_isocontour(meniscus_mesh, meniscus_centerline, spacing)
    centers = np.array([contour_slice.center for contour_slice in slices])

    distances, _ = KDTree(sample(meniscus_centerline, 5001)).query(centers)
    assert distances.max() < 0.05
    # Chords of a circle of radius 16.4 are under 1% shorter than these arcs.
    assert np.linalg.norm(np.diff(centers, axis=0), axis=1) == pytest.approx(np.full(len(centers) - 1, spacing), rel=0.02)


def test_slice_frames_are_right_handed_and_normal_to_curve(meniscus_slices, meniscus_centerline):
    u = np.linspace(0.0, 1.0, 5001)
    points = meniscus_centerline(u)
    tangents = meniscus_centerline(u, nu=1)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    tree = KDTree(points)

    for contour_slice in meniscus_slices:
        frame = np.column_stack((contour_slice.u_axis, contour_slice.v_axis, contour_slice.normal))
        assert frame.T @ frame == pytest.approx(np.eye(3), abs=1e-9)
        assert np.linalg.det(frame) == pytest.approx(1.0)
        _, nearest = tree.query(contour_slice.center)
        assert contour_slice.normal @ tangents[nearest] == pytest.approx(1.0, abs=1e-3)


def test_slices_do_not_twist_along_the_curve(meniscus_slices):
    v_axes = np.array([contour_slice.v_axis for contour_slice in meniscus_slices])

    assert (v_axes[1:] * v_axes[:-1]).sum(axis=1).min() > 0.99


def make_label(mask):
    return sitk.GetImageFromArray(mask.astype(np.uint8))


def test_slices_clip_ends_below_a_fifth_of_average_area():
    # A cone along z with base radius 15 at z = 5 and its apex at z = 75. Mean
    # area is a third of the base area, so slices stop where r / 15 = sqrt(0.2 / 3).
    z, y, x = np.indices((81, 41, 41))
    radius = 15.0 * (75 - z) / 70
    mask = (z >= 5) & (z <= 75) & (np.hypot(x - 20, y - 20) <= radius)
    mesh = extract_isocontour(signed_distance_field(make_label(mask)))

    slices = slice_isocontour(mesh, straight_line((20, 20, 5.5), (20, 20, 74.5)), spacing=2.0)

    heights = [contour_slice.center[2] for contour_slice in slices]
    expected_tip = 75 - 70 * np.sqrt(0.2 / 3)
    assert min(heights) < 8.0
    assert max(heights) == pytest.approx(expected_tip, abs=4.0)


def test_slices_keep_small_sections_between_the_ends():
    z, y, x = np.indices((61, 41, 41))
    radius = np.where((z >= 25) & (z <= 35), 3.0, 15.0)
    mask = (z >= 5) & (z <= 55) & (np.hypot(x - 20, y - 20) <= radius)
    mesh = extract_isocontour(signed_distance_field(make_label(mask)))

    slices = slice_isocontour(mesh, straight_line((20, 20, 6), (20, 20, 54)), spacing=2.0)

    areas = np.array([contour_slice.area for contour_slice in slices])
    assert len(slices) == 25
    assert areas.min() < 0.2 * areas.mean()


def test_slice_rejects_gap_in_the_mesh_along_the_curve():
    z, y, x = np.indices((71, 41, 41))
    in_blocks = ((z >= 5) & (z <= 25)) | ((z >= 45) & (z <= 65))
    mask = in_blocks & (np.hypot(x - 20, y - 20) <= 10)
    mesh = extract_isocontour(signed_distance_field(make_label(mask)))

    with pytest.raises(ValueError, match="does not cut the mesh"):
        slice_isocontour(mesh, straight_line((20, 20, 6), (20, 20, 64)), spacing=2.0)


def test_slice_rejects_curve_that_misses_the_mesh(meniscus_mesh):
    with pytest.raises(ValueError, match="do not cut the mesh"):
        slice_isocontour(meniscus_mesh, straight_line((100, 100, 100), (100, 100, 120)), spacing=2.0)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"spacing": 0.0}, "must be positive"),
        ({"spacing": -1.0}, "must be positive"),
        ({"spacing": 5.0, "points_per_slice": 2}, "at least 3"),
    ],
)
def test_slice_rejects_invalid_arguments(meniscus_mesh, meniscus_centerline, arguments, message):
    with pytest.raises(ValueError, match=message):
        slice_isocontour(meniscus_mesh, meniscus_centerline, **arguments)
