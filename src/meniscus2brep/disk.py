"""Reading and writing segmented LabelMaps as .nii, .nrrd, or .mhd + .raw."""

from os import PathLike
from pathlib import Path

import SimpleITK as sitk

SUPPORTED_SUFFIXES = frozenset({".nii", ".nrrd", ".mhd"})


def read_label_map(path: str | PathLike[str]) -> sitk.Image:
    """Read a segmented LabelMap from disk.

    For a .mhd file, the .raw file it references must also exist.

    Raises:
        ValueError: If the file format is unsupported or the image is not
            sitkUInt8.
    """
    path = _validated_path(path)
    image = sitk.ReadImage(str(path))
    _require_label_map(image)
    return image


def write_label_map(image: sitk.Image, path: str | PathLike[str]) -> None:
    """Write a segmented LabelMap to disk, overwriting any existing file.

    A .mhd path also writes a .raw file alongside it.

    Raises:
        ValueError: If the file format is unsupported or the image is not
            sitkUInt8.
    """
    path = _validated_path(path)
    _require_label_map(image)
    sitk.WriteImage(image, str(path), useCompression=False)


def _validated_path(path: str | PathLike[str]) -> Path:
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ValueError(f"Unsupported file format {path.name!r}; expected one of: {supported}")
    return path


def _require_label_map(image: sitk.Image) -> None:
    if image.GetPixelID() != sitk.sitkUInt8:
        raise ValueError(f"LabelMap must be sitkUInt8, got {image.GetPixelIDTypeAsString()}")
