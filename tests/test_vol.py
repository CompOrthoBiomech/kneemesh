import numpy as np
import pytest
import SimpleITK as sitk
from vtkmodules.vtkFiltersCore import vtkFeatureEdges, vtkMassProperties

from kneemesh.geo.meniscus import slice_isocontour
from kneemesh.vol import extract_isocontour, signed_distance_field, smooth_isocontour


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
        second_differences = [np.linalg.norm(np.roll(s.points, -1, axis=0) - 2 * s.points + np.roll(s.points, 1, axis=0), axis=1) for s in slices]
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


def excess_turning_degrees(slices):
    """Mean turning around each contour beyond one full turn, which ripples add to."""
    totals = []
    for contour_slice in slices:
        steps = np.roll(contour_slice.points, -1, axis=0) - contour_slice.points
        steps /= np.linalg.norm(steps, axis=1, keepdims=True)
        cosines = np.clip(np.sum(steps * np.roll(steps, 1, axis=0), axis=1), -1.0, 1.0)
        totals.append(np.degrees(np.arccos(cosines)).sum())
    return np.mean(totals) - 360.0


def test_smoothed_isocontour_keeps_connectivity_and_stays_closed(meniscus_mesh, meniscus_smoothed_mesh):
    assert meniscus_smoothed_mesh.GetNumberOfPoints() == meniscus_mesh.GetNumberOfPoints()
    assert meniscus_smoothed_mesh.GetNumberOfCells() == meniscus_mesh.GetNumberOfCells()
    assert open_edge_count(meniscus_smoothed_mesh) == 0


def test_smoothed_isocontour_loses_little_volume(meniscus_mesh, meniscus_smoothed_mesh):
    assert mesh_volume(meniscus_smoothed_mesh) == pytest.approx(mesh_volume(meniscus_mesh), rel=0.05)


def test_smoothing_isocontour_straightens_slice_contours(meniscus_mesh, meniscus_smoothed_mesh, meniscus_centerline):
    rough = slice_isocontour(meniscus_mesh, meniscus_centerline, spacing=2.0)
    smooth = slice_isocontour(meniscus_smoothed_mesh, meniscus_centerline, spacing=2.0)

    assert excess_turning_degrees(smooth) < 0.7 * excess_turning_degrees(rough)


def test_smoothing_isocontour_more_shrinks_more(meniscus_mesh, meniscus_smoothed_mesh):
    stronger = smooth_isocontour(meniscus_mesh, iterations=100)

    assert mesh_volume(stronger) < mesh_volume(meniscus_smoothed_mesh)


def test_smoothing_isocontour_with_no_iterations_returns_the_mesh(meniscus_mesh):
    assert smooth_isocontour(meniscus_mesh, iterations=0) is meniscus_mesh


@pytest.mark.parametrize(
    ("arguments", "message"),
    [({"iterations": -1}, "iterations"), ({"pass_band": 0.0}, "pass band"), ({"pass_band": 2.5}, "pass band")],
)
def test_smoothing_isocontour_rejects_invalid_arguments(meniscus_mesh, arguments, message):
    with pytest.raises(ValueError, match=message):
        smooth_isocontour(meniscus_mesh, **arguments)
