"""Crop the OpenKnee(s) LabelMaps to their labels and store them as sitkUInt8, as tests/dat is.

Each <subject>/<structure>.nii under --data is cropped to its nonzero voxels' bounding box, with no
margin, and cast to sitkUInt8 with every nonzero voxel 1. A file is replaced only after the crop is
checked: the same voxels, at the same physical points, survive. Files already cropped and in sitkUInt8
are left alone, so running it again changes nothing.

Run from the repository root:

    uv run python examples/validation/crop_oks.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from kneemesh.disk import read_label_map, write_label_map


def crop_label(image: sitk.Image) -> sitk.Image:
    """The image cropped to its nonzero voxels' bounding box, with every nonzero voxel 1 in sitkUInt8."""
    binary = sitk.Cast(sitk.NotEqual(image, 0), sitk.sitkUInt8)
    statistics = sitk.LabelShapeStatisticsImageFilter()
    statistics.Execute(binary)
    if not statistics.HasLabel(1):
        raise ValueError("the image has no nonzero voxels")
    box = statistics.GetBoundingBox(1)
    dimension = image.GetDimension()
    return sitk.RegionOfInterest(binary, size=box[dimension:], index=box[:dimension])


def _check(original: sitk.Image, cropped: sitk.Image) -> None:
    """Raise unless the cropped image holds exactly the original's nonzero voxels at the same physical points."""
    source = sitk.GetArrayViewFromImage(original) != 0
    target = sitk.GetArrayViewFromImage(cropped)
    if int(np.count_nonzero(source)) != int(np.count_nonzero(target)):
        raise ValueError("cropping changed the voxel count")
    index = original.TransformPhysicalPointToIndex(cropped.GetOrigin())
    # Arrays are indexed (z, y, x), image indices (x, y, z).
    window = tuple(slice(start, start + size) for start, size in zip(reversed(index), reversed(cropped.GetSize()), strict=True))
    if not np.array_equal(source[window], target != 0):
        raise ValueError("cropping moved voxels")
    if cropped.GetSpacing() != original.GetSpacing() or cropped.GetDirection() != original.GetDirection():
        raise ValueError("cropping changed the grid")


def crop_file(path: Path) -> bool:
    """Crop and cast one LabelMap in place; False if it was already cropped and in sitkUInt8."""
    original = read_label_map(path)
    cropped = crop_label(original)
    if original.GetPixelID() == sitk.sitkUInt8 and cropped.GetSize() == original.GetSize():
        return False
    _check(original, cropped)
    temporary = path.with_name(f"{path.stem}.cropping{path.suffix}")
    write_label_map(cropped, temporary)
    _check(original, read_label_map(temporary))
    _ = temporary.replace(path)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("dat/oks"), help="directory of oks*/ subject directories")
    arguments = parser.parse_args()
    paths = sorted(arguments.data.glob("oks*/*.nii"))
    if not paths:
        parser.error(f"no LabelMaps under {arguments.data}")
    for path in paths:
        before = path.stat().st_size
        changed = crop_file(path)
        print(f"{path}: " + (f"{before / 1e6:.1f} MB -> {path.stat().st_size / 1e6:.1f} MB" if changed else "already cropped"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
