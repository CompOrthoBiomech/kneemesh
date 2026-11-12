import numpy as np
import pytest
import SimpleITK as sitk
from loguru import logger

from kneemesh.segmentation import Segmentation, assemble_label_maps


def voxel_count(image, value=None):
    array = sitk.GetArrayViewFromImage(image)
    return int(np.count_nonzero(array if value is None else array == value))


def physical_centroid(image, value):
    indices = np.argwhere(sitk.GetArrayViewFromImage(image) == value)[:, ::-1].astype(float)
    return np.mean([image.TransformContinuousIndexToPhysicalPoint(index.tolist()) for index in indices], axis=0)


def physical_corners(image):
    size = image.GetSize()
    return np.array(
        [image.TransformContinuousIndexToPhysicalPoint([-0.5 if o == 0 else n - 0.5 for o, n in zip(offsets, size)]) for offsets in np.ndindex(2, 2, 2)]
    )


def box_label(size, spacing, origin, low, high, direction=None):
    """A sitkUInt8 LabelMap that is 1 on index box [low, high) and 0 elsewhere."""
    array = np.zeros(size[::-1], dtype=np.uint8)
    array[low[2] : high[2], low[1] : high[1], low[0] : high[0]] = 1
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    if direction is not None:
        image.SetDirection(direction)
    return image


@pytest.fixture(scope="module")
def two_grids(meniscus_label):
    """The meniscus fixture and a coarser, offset box below it, as single-label LabelMaps on different grids.

    The box's voxel faces fall on the shared grid's, which the finer meniscus sets.
    """
    box = box_label((20, 20, 10), (2.0, 2.0, 2.0), (-19.25, -19.25, -27.5), (2, 2, 1), (18, 18, 8))
    return {"bone": box, "meniscus": meniscus_label}


def test_segmentation_names_its_labels(meniscus_label):
    segmentation = Segmentation(meniscus_label * 3, {"meniscus": 3})

    assert segmentation.value("meniscus") == 3
    assert voxel_count(segmentation.mask("meniscus")) == voxel_count(meniscus_label)
    assert segmentation.mask("meniscus").GetPixelID() == sitk.sitkUInt8


def test_segmentation_copies_its_labels(meniscus_label):
    labels = {"meniscus": 1}
    segmentation = Segmentation(meniscus_label, labels)
    labels["tibia"] = 2

    assert segmentation.labels == {"meniscus": 1}


def test_segmentation_rejects_unknown_name(meniscus_label):
    with pytest.raises(ValueError, match="No label named 'tibia'"):
        Segmentation(meniscus_label, {"meniscus": 1}).value("tibia")


@pytest.mark.parametrize(
    ("labels", "message"),
    [({}, "at least one"), ({"meniscus": 0}, "background"), ({"meniscus": 1, "copy": 1}, "its own label value"), ({"meniscus": 2}, "not present")],
)
def test_segmentation_rejects_invalid_labels(meniscus_label, labels, message):
    with pytest.raises(ValueError, match=message):
        Segmentation(meniscus_label, labels)


def test_segmentation_rejects_non_label_map(meniscus_label):
    with pytest.raises(ValueError, match="LabelMap must be 3D"):
        Segmentation(sitk.Cast(meniscus_label, sitk.sitkFloat32), {"meniscus": 1})


def test_assembly_numbers_labels_in_order(two_grids):
    segmentation = assemble_label_maps(two_grids)

    assert segmentation.labels == {"bone": 1, "meniscus": 2}
    assert segmentation.image.GetPixelID() == sitk.sitkUInt8


def test_assembly_uses_the_smallest_spacing_on_each_axis(two_grids):
    assert assemble_label_maps(two_grids).image.GetSpacing() == pytest.approx((0.5, 0.5, 1.0))


def test_assembly_matches_spacing_to_axes_by_direction(meniscus_label):
    permuted = sitk.DICOMOrient(sitk.BinaryDilate(meniscus_label, [1, 1, 1]), "SLP")
    assert permuted.GetSpacing() == pytest.approx((1.0, 0.5, 0.5))

    image = assemble_label_maps({"grown": permuted, "meniscus": meniscus_label}).image

    assert sorted(image.GetSpacing()) == pytest.approx([0.5, 0.5, 1.0])
    assert image.GetSpacing() == pytest.approx(permuted.GetSpacing())


def test_assembly_aligns_the_grid_with_the_finest_input(two_grids):
    segmentation = assemble_label_maps(two_grids, "nearest_neighbor")
    meniscus = two_grids["meniscus"]

    assert voxel_count(segmentation.image, segmentation.value("meniscus")) == voxel_count(meniscus)
    offset = np.subtract(segmentation.image.GetOrigin(), meniscus.GetOrigin()) / segmentation.image.GetSpacing()
    assert offset == pytest.approx(np.round(offset))


def test_assembly_covers_every_input_with_padding(two_grids):
    image = assemble_label_maps(two_grids, padding=3).image
    low, high = physical_corners(image).min(axis=0), physical_corners(image).max(axis=0)
    inputs = np.vstack([physical_corners(label) for label in two_grids.values()])
    margin = 3 * np.array(image.GetSpacing())

    assert np.all(low <= inputs.min(axis=0) - margin + 1e-6)
    assert np.all(high >= inputs.max(axis=0) + margin - 1e-6)
    assert np.all(high - low < inputs.max(axis=0) - inputs.min(axis=0) + 2 * margin + np.array(image.GetSpacing()))


def test_assembly_pads_with_background(two_grids):
    # A copy, because a view of a temporary image reads freed memory once the image is collected.
    array = sitk.GetArrayFromImage(assemble_label_maps(two_grids, padding=2).image)

    for axis in range(3):
        assert not np.take(array, [0, 1, -2, -1], axis=axis).any()


def test_assembly_without_padding_reproduces_a_single_input_grid(meniscus_label):
    image = assemble_label_maps({"meniscus": meniscus_label}, "nearest_neighbor", padding=0).image

    assert image.GetSize() == meniscus_label.GetSize()
    assert image.GetOrigin() == pytest.approx(meniscus_label.GetOrigin())
    assert np.array_equal(sitk.GetArrayViewFromImage(image), sitk.GetArrayViewFromImage(meniscus_label))


@pytest.mark.parametrize("interpolator", ["label_gaussian", "nearest_neighbor"])
def test_assembly_keeps_each_structure_in_place(two_grids, interpolator):
    segmentation = assemble_label_maps(two_grids, interpolator)

    for name, source in two_grids.items():
        value = segmentation.value(name)
        source_volume = voxel_count(source) * np.prod(source.GetSpacing())
        volume = voxel_count(segmentation.image, value) * np.prod(segmentation.image.GetSpacing())
        assert volume == pytest.approx(source_volume, rel=0.05)
        assert physical_centroid(segmentation.image, value) == pytest.approx(physical_centroid(source, 1), abs=0.3)


def test_label_gaussian_smooths_the_staircase_of_a_coarse_input():
    sphere = sitk.Cast(sitk.GaussianSource(sitk.sitkFloat32, [16] * 3, [3.0] * 3, [7.5] * 3, 1.0, [2.0] * 3) > 0.3, sitk.sitkUInt8)
    sphere.SetSpacing([2.0] * 3)
    fine = box_label((60, 60, 60), (0.5, 0.5, 0.5), (0.0, 0.0, 0.0), (0, 0, 0), (8, 8, 8))

    def boundary_voxels(interpolator):
        image = assemble_label_maps({"fine": fine, "sphere": sphere}, interpolator).mask("sphere")
        return voxel_count(sitk.BinaryContour(image))

    assert boundary_voxels("label_gaussian") < boundary_voxels("nearest_neighbor")


def test_assembly_reconciles_direction_cosines(meniscus_label):
    rim = sitk.Cast(sitk.BinaryDilate(meniscus_label, [1, 1, 1]) - meniscus_label, sitk.sitkUInt8)
    # Each reorders the voxel array and adjusts the direction cosines, keeping the structure in place.
    inputs = {"meniscus": sitk.DICOMOrient(meniscus_label, "RAI"), "rim": sitk.DICOMOrient(rim, "SPL")}
    assert len({image.GetDirection() for image in (*inputs.values(), meniscus_label)}) == 3

    segmentation = assemble_label_maps(inputs, "nearest_neighbor", padding=0)

    assert segmentation.image.GetDirection() == inputs["meniscus"].GetDirection()
    for name, source in (("meniscus", meniscus_label), ("rim", rim)):
        assert voxel_count(segmentation.image, segmentation.value(name)) == voxel_count(source)
        assert physical_centroid(segmentation.image, segmentation.value(name)) == pytest.approx(physical_centroid(source, 1), abs=1e-6)


def test_assembly_lets_later_structures_win_overlaps_and_warns(meniscus_label):
    grown = sitk.BinaryDilate(meniscus_label, [1, 1, 1])
    messages = []
    logger.enable("kneemesh")
    handler = logger.add(messages.append, level="WARNING")
    try:
        segmentation = assemble_label_maps({"grown": grown, "meniscus": meniscus_label}, "nearest_neighbor")
    finally:
        logger.remove(handler)
        logger.disable("kneemesh")

    assert voxel_count(segmentation.image, 2) == voxel_count(meniscus_label)
    assert voxel_count(segmentation.image, 1) == voxel_count(grown) - voxel_count(meniscus_label)
    assert any("overlap" in message for message in messages)


def test_assembly_rejects_a_structure_covered_by_later_ones(meniscus_label):
    with pytest.raises(ValueError, match=r"\['first'\] are entirely covered"):
        assemble_label_maps({"first": meniscus_label, "second": meniscus_label}, "nearest_neighbor")


def test_assembly_treats_every_nonzero_value_as_the_structure(meniscus_label):
    segmentation = assemble_label_maps({"meniscus": sitk.Cast(meniscus_label * 1000, sitk.sitkUInt16)}, "nearest_neighbor")

    assert voxel_count(segmentation.image, 1) == voxel_count(meniscus_label)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"label_maps": {}}, "at least one"),
        ({"interpolator": "linear"}, "interpolator"),
        ({"padding": -1}, "padding"),
    ],
)
def test_assembly_rejects_invalid_arguments(meniscus_label, arguments, message):
    with pytest.raises(ValueError, match=message):
        assemble_label_maps(**{"label_maps": {"meniscus": meniscus_label}, **arguments})


def test_assembly_rejects_empty_and_2d_inputs(meniscus_label):
    with pytest.raises(ValueError, match="no foreground"):
        assemble_label_maps({"meniscus": meniscus_label, "empty": meniscus_label * 0})
    with pytest.raises(ValueError, match="LabelMap must be 3D"):
        assemble_label_maps({"flat": sitk.Image(4, 4, sitk.sitkUInt8)})
