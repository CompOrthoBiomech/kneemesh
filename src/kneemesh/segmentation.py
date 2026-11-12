"""Multilabel segmentations: a LabelMap with a name for each label, and assembling one from single-label LabelMaps."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
import SimpleITK as sitk
from loguru import logger

from kneemesh.vol import require_3d_label_map

type Interpolator = Literal["label_gaussian", "nearest_neighbor"]

_INTERPOLATORS: dict[str, int] = {"label_gaussian": sitk.sitkLabelGaussian, "nearest_neighbor": sitk.sitkNearestNeighbor}
# The distance field's default Gaussian has a sigma of about 0.77 voxels, so
# 4 voxels of background keep its support inside the image around every label.
DEFAULT_PADDING_VOXELS = 4


@dataclass(frozen=True, eq=False)
class Segmentation:
    """A LabelMap holding several structures, and the name of each structure's label.

    Attributes:
        image: 3D LabelMap of sitkUInt8, sitkUInt16, or sitkInt16 pixels.
        labels: Label value of each structure, by name. Other values in
            image are allowed and ignored.
    """

    image: sitk.Image
    labels: dict[str, int]

    def __post_init__(self) -> None:
        require_3d_label_map(self.image)
        if not self.labels:
            raise ValueError("A segmentation needs at least one named label")
        if 0 in self.labels.values():
            raise ValueError("Label value 0 is the background and cannot name a structure")
        if len(set(self.labels.values())) < len(self.labels):
            raise ValueError(f"Each name needs its own label value, got {self.labels}")
        present = set(np.unique(sitk.GetArrayViewFromImage(self.image)).tolist())
        absent = [name for name, value in self.labels.items() if value not in present]
        if absent:
            raise ValueError(f"Labels {absent} are not present in the image")
        object.__setattr__(self, "labels", dict(self.labels))

    def value(self, name: str) -> int:
        """Return the label value of the named structure.

        Raises:
            ValueError: If no label has that name.
        """
        if name not in self.labels:
            raise ValueError(f"No label named {name!r}; the segmentation has {sorted(self.labels)}")
        return self.labels[name]

    def mask(self, name: str) -> sitk.Image:
        """Return a sitkUInt8 image that is 1 on the named structure and 0 elsewhere."""
        return sitk.Equal(self.image, self.value(name))


def assemble_label_maps(
    label_maps: Mapping[str, sitk.Image],
    interpolator: Interpolator = "label_gaussian",
    padding: int = DEFAULT_PADDING_VOXELS,
) -> Segmentation:
    """Resample single-label LabelMaps onto one grid and combine them into a multilabel Segmentation.

    Every nonzero voxel of an input belongs to its structure. The shared grid
    takes the first input's direction cosines, the smallest voxel spacing of
    any input along each axis, and the bounding box of all inputs in physical
    space, padded with background on every side. Input axes count toward the
    shared axis nearest them in direction, and the grid's voxel faces line up
    with the input that has the smallest spacing along each axis. Structures get label values
    1, 2, ... in the order given. Where resampled structures overlap, the one
    given later wins.

    Args:
        label_maps: Single-label 3D LabelMaps by structure name.
        interpolator: "label_gaussian" resamples with sitkLabelGaussian, which
            smooths the voxel staircase on finer grids. "nearest_neighbor"
            uses sitkNearestNeighbor.
        padding: Background voxels added beyond the bounding box on each side.

    Returns:
        The assembled segmentation, sitkUInt8 for up to 255 structures and
        sitkUInt16 above that.

    Raises:
        ValueError: If label_maps is empty, an input is not a 3D LabelMap of
            an accepted pixel type or has no foreground, interpolator is
            unknown, padding is negative, or a structure vanishes in resampling
            or is entirely covered by later ones.
    """
    if not label_maps:
        raise ValueError("Assembling needs at least one LabelMap")
    if interpolator not in _INTERPOLATORS:
        raise ValueError(f"interpolator must be one of {sorted(_INTERPOLATORS)}, got {interpolator!r}")
    if padding < 0:
        raise ValueError(f"padding must not be negative, got {padding}")
    for name, image in label_maps.items():
        require_3d_label_map(image)
        if not sitk.GetArrayViewFromImage(image).any():
            raise ValueError(f"LabelMap {name!r} has no foreground")

    reference = _shared_grid(list(label_maps.values()), padding)
    pixel_id = sitk.sitkUInt8 if len(label_maps) <= 255 else sitk.sitkUInt16
    combined = np.zeros(sitk.GetArrayViewFromImage(reference).shape, dtype=np.uint16)
    labels = {}
    for value, (name, image) in enumerate(label_maps.items(), start=1):
        foreground = sitk.NotEqual(image, 0)
        resampled = sitk.Resample(foreground, reference, sitk.Transform(), _INTERPOLATORS[interpolator], 0, sitk.sitkUInt8)
        mask = sitk.GetArrayViewFromImage(resampled) != 0
        if not mask.any():
            raise ValueError(f"LabelMap {name!r} vanished when resampled onto the shared grid")
        overlap = int(np.count_nonzero(mask & (combined != 0)))
        if overlap:
            logger.warning("{} assembled voxels of {!r} overlap earlier structures and take its label", overlap, name)
        combined[mask] = value
        labels[name] = value
    covered = [name for name, value in labels.items() if not np.any(combined == value)]
    if covered:
        raise ValueError(f"LabelMaps {covered} are entirely covered by structures given after them")

    image = sitk.Cast(sitk.GetImageFromArray(combined), pixel_id)
    image.CopyInformation(reference)
    logger.debug("Assembled {} on a {} grid with spacing {}", labels, image.GetSize(), image.GetSpacing())
    return Segmentation(image, labels)


def _shared_grid(images: list[sitk.Image], padding: int) -> sitk.Image:
    """Return an empty image on the grid that covers every image, in the first image's orientation.

    Each image axis counts toward the shared axis closest to it in direction,
    so images with permuted or flipped axes contribute their spacing to the
    right shared axis. Along each shared axis the grid takes the smallest
    spacing and lines its voxel faces up with the image that has it, so that
    image resamples exactly when it shares the orientation.
    """
    direction = np.reshape(images[0].GetDirection(), (3, 3))
    spacing = np.full(3, np.inf)
    anchors = np.zeros(3)
    corners = []
    for image in images:
        image_direction = np.reshape(image.GetDirection(), (3, 3))
        nearest_axes = np.argmax(np.abs(direction.T @ image_direction), axis=0)
        first_corner = direction.T @ image.TransformContinuousIndexToPhysicalPoint([-0.5, -0.5, -0.5])
        for image_axis, axis in enumerate(nearest_axes):
            if image.GetSpacing()[image_axis] < spacing[axis]:
                spacing[axis] = image.GetSpacing()[image_axis]
                anchors[axis] = first_corner[axis]
        size = image.GetSize()
        for offsets in np.ndindex(2, 2, 2):
            index = [-0.5 if offset == 0 else extent - 0.5 for offset, extent in zip(offsets, size)]
            corners.append(direction.T @ image.TransformContinuousIndexToPhysicalPoint(index))
    low, high = np.min(corners, axis=0), np.max(corners, axis=0)

    # The tolerance keeps floating-point excess from adding a voxel where extents fit exactly.
    low = anchors - np.ceil((anchors - low) / spacing - 1e-6) * spacing
    size = [math.ceil(extent / step - 1e-6) + 2 * padding for extent, step in zip(high - low, spacing)]
    origin = direction @ (low + spacing * (0.5 - padding))

    reference = sitk.Image(size, sitk.sitkUInt8)
    reference.SetSpacing(spacing.tolist())
    reference.SetOrigin(origin.tolist())
    reference.SetDirection(direction.ravel().tolist())
    return reference
