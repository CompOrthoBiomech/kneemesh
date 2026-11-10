"""Reading and writing LabelMaps as .nii, .nrrd, or .mhd + .raw, poly data as .vtp, and B-reps as .brep."""

from os import PathLike
from pathlib import Path

import gmsh
import SimpleITK as sitk
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader, vtkXMLPolyDataWriter

from meniscus2brep.geo import BRep, gmsh_session

SUPPORTED_SUFFIXES = frozenset({".nii", ".nrrd", ".mhd"})
POLY_DATA_SUFFIXES = frozenset({".vtp"})
BREP_SUFFIXES = frozenset({".brep"})


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


def read_poly_data(path: str | PathLike[str]) -> vtkPolyData:
    """Read poly data from a VTK XML .vtp file.

    Raises:
        ValueError: If the suffix is not .vtp or the file is not VTK XML poly data.
        FileNotFoundError: If the file does not exist.
    """
    path = _validated_path(path, POLY_DATA_SUFFIXES)
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {path}")

    # The reader logs failures and returns empty output without any error state.
    reader = vtkXMLPolyDataReader()
    if not reader.CanReadFile(str(path)):
        raise ValueError(f"Not a VTK XML poly data file: {path}")
    reader.SetFileName(str(path))
    reader.Update()
    return reader.GetOutput()


def write_poly_data(poly_data: vtkPolyData, path: str | PathLike[str]) -> None:
    """Write poly data to a VTK XML .vtp file, overwriting any existing file.

    Raises:
        ValueError: If the suffix is not .vtp.
        OSError: If the file cannot be written.
    """
    path = _validated_path(path, POLY_DATA_SUFFIXES)

    writer = vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(poly_data)
    if not writer.Write():
        raise OSError(f"Could not write poly data to {path}")


def read_brep(path: str | PathLike[str]) -> BRep:
    """Read a B-rep from an Open Cascade .brep file.

    Raises:
        ValueError: If the suffix is not .brep or the file holds no valid shape.
        FileNotFoundError: If the file does not exist.
    """
    path = _validated_path(path, BREP_SUFFIXES)
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {path}")

    with gmsh_session():
        try:
            shapes = gmsh.model.occ.importShapes(str(path))
        except Exception as error:  # gmsh raises only the bare Exception type.
            raise ValueError(f"Not a valid B-rep file: {path}") from error
    if not shapes:
        raise ValueError(f"Not a valid B-rep file: {path}")
    return BRep(path.read_text(encoding="utf-8"))


def write_brep(brep: BRep, path: str | PathLike[str]) -> None:
    """Write a B-rep to an Open Cascade .brep file, overwriting any existing file.

    Raises:
        ValueError: If the suffix is not .brep.
        OSError: If the file cannot be written.
    """
    path = _validated_path(path, BREP_SUFFIXES)
    path.write_text(brep.text, encoding="utf-8")


def _validated_path(path: str | PathLike[str], suffixes: frozenset[str] = SUPPORTED_SUFFIXES) -> Path:
    path = Path(path)
    if path.suffix.lower() not in suffixes:
        supported = ", ".join(sorted(suffixes))
        raise ValueError(f"Unsupported file format {path.name!r}; expected one of: {supported}")
    return path


def _require_label_map(image: sitk.Image) -> None:
    if image.GetPixelID() != sitk.sitkUInt8:
        raise ValueError(f"LabelMap must be sitkUInt8, got {image.GetPixelIDTypeAsString()}")
