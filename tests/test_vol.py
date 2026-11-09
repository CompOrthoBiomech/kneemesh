import numpy as np
import pytest
import SimpleITK as sitk
from scipy.interpolate import BSpline
from scipy.spatial import KDTree

from meniscus2brep.vol import centerline

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


def test_centerline_rejects_non_uint8_image(meniscus_label):
    with pytest.raises(ValueError, match="3D sitkUInt8"):
        centerline(sitk.Cast(meniscus_label, sitk.sitkUInt16))


def test_centerline_rejects_2d_image():
    with pytest.raises(ValueError, match="3D sitkUInt8"):
        centerline(sitk.Image(8, 8, sitk.sitkUInt8))


def test_centerline_rejects_tiny_label():
    image = sitk.Image(8, 8, 8, sitk.sitkUInt8)
    image[1, 1, 1] = 1

    with pytest.raises(ValueError, match="too small"):
        centerline(image)
