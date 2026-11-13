import numpy as np
import pytest
import SimpleITK as sitk

from kneemesh.cartilage import classify_sides, field_gradient, field_value
from kneemesh.vol import extract_isocontour, signed_distance_field

# The radius of the cartilage_segmentation fixture's spherical bone, whose top touches z = 0.
BONE_RADIUS = 40.0
BONE_CENTER = np.array([0.0, 0.0, -BONE_RADIUS])


def radial_heights(points):
    return np.linalg.norm(points - BONE_CENTER, axis=1) - BONE_RADIUS


def test_bone_side_lies_on_the_bone_and_articular_side_above_it(cartilage_sides):
    bone = radial_heights(cartilage_sides.bone)
    articular = radial_heights(cartilage_sides.articular)

    # The bone side follows the voxelized sphere, about 0.35 mm (under a 0.5 mm voxel) outside the analytic one.
    assert np.abs(bone).max() < 1.0
    assert np.median(articular) > 1.0


def test_sides_drop_the_steep_rim(cartilage_sides):
    rims = [np.hypot(points[:, 0] / 15.0, points[:, 1] / 10.0).max() for points in (cartilage_sides.articular, cartilage_sides.bone)]

    # The rim tapers to nothing at the ellipse; neither side reaches all the way out.
    assert max(rims) < 1.0


def test_higher_side_cosine_keeps_fewer_points(cartilage_segmentation):
    surface = extract_isocontour(signed_distance_field(cartilage_segmentation.image, 2))
    bone_field = signed_distance_field(cartilage_segmentation.image, 1)

    loose, strict = classify_sides(surface, bone_field, 0.2), classify_sides(surface, bone_field, 0.8)

    assert len(strict.articular) < len(loose.articular)
    assert len(strict.bone) < len(loose.bone)


@pytest.mark.parametrize("cosine", [0.0, 1.0, -0.5])
def test_classify_sides_rejects_side_cosine_out_of_range(cartilage_segmentation, cosine):
    surface = extract_isocontour(signed_distance_field(cartilage_segmentation.image, 2))

    with pytest.raises(ValueError, match="side_cosine"):
        classify_sides(surface, signed_distance_field(cartilage_segmentation.image, 1), cosine)


def test_classify_sides_needs_a_bone_beside_the_cartilage(cartilage_segmentation):
    surface = extract_isocontour(signed_distance_field(cartilage_segmentation.image, 2))
    # A uniform field has no gradient, so no point faces toward or away from it.
    flat = sitk.Image(cartilage_segmentation.image.GetSize(), sitk.sitkFloat32)
    flat.CopyInformation(cartilage_segmentation.image)

    with pytest.raises(ValueError, match="no points facing"):
        classify_sides(surface, flat)


def test_field_value_and_gradient_follow_the_image_orientation():
    image = sitk.Image(12, 10, 8, sitk.sitkFloat64)
    image.SetSpacing((0.5, 1.0, 2.0))
    image.SetOrigin((3.0, -2.0, 1.0))
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    image.SetDirection(rotation.ravel().tolist())
    slope = np.array([2.0, -3.0, 0.5])
    indices = np.stack(np.meshgrid(*(np.arange(size) for size in image.GetSize()), indexing="ij"), axis=-1).reshape(-1, 3)
    physical = np.array([image.TransformIndexToPhysicalPoint(index.tolist()) for index in indices])
    values = np.zeros(image.GetSize()[::-1])
    values[indices[:, 2], indices[:, 1], indices[:, 0]] = physical @ slope
    field = sitk.GetImageFromArray(values)
    field.CopyInformation(image)

    inside = np.array(
        [field.TransformContinuousIndexToPhysicalPoint([5.3, 4.6, 3.2]), field.TransformContinuousIndexToPhysicalPoint([2.5, 7.0, 4.9])]
    )

    assert field_value(field, inside) == pytest.approx(inside @ slope)
    assert field_gradient(field, inside) == pytest.approx(np.tile(slope, (2, 1)))
