import numpy as np
import pytest
import SimpleITK as sitk
from vtkmodules.util.numpy_support import vtk_to_numpy

from meniscus2brep.disk import read_brep, read_label_map, read_poly_data, write_brep, write_label_map, write_poly_data
from meniscus2brep.geo import BRep


@pytest.mark.parametrize("filename", ["seg.nii", "seg.nrrd", "seg.mhd"])
def test_round_trip_preserves_labels_and_geometry(tmp_path, meniscus_label, filename):
    path = tmp_path / filename

    write_label_map(meniscus_label, path)
    result = read_label_map(path)

    assert result.GetPixelID() == sitk.sitkUInt8
    assert result.GetSize() == meniscus_label.GetSize()
    assert sitk.Hash(result) == sitk.Hash(meniscus_label)
    assert result.GetSpacing() == pytest.approx(meniscus_label.GetSpacing())
    assert result.GetOrigin() == pytest.approx(meniscus_label.GetOrigin())
    assert result.GetDirection() == pytest.approx(meniscus_label.GetDirection())


def test_mhd_writes_separate_raw_file(tmp_path, meniscus_label):
    write_label_map(meniscus_label, tmp_path / "seg.mhd")

    assert (tmp_path / "seg.raw").is_file()


@pytest.mark.parametrize("filename", ["seg.png", "seg.nii.gz", "seg"])
def test_write_rejects_unsupported_format(tmp_path, meniscus_label, filename):
    with pytest.raises(ValueError, match="Unsupported file format"):
        write_label_map(meniscus_label, tmp_path / filename)


def test_read_rejects_unsupported_format(tmp_path):
    with pytest.raises(ValueError, match="Unsupported file format"):
        read_label_map(tmp_path / "seg.png")


@pytest.mark.parametrize("pixel_id", [sitk.sitkInt8, sitk.sitkUInt16, sitk.sitkFloat32])
def test_write_rejects_non_uint8_image(tmp_path, meniscus_label, pixel_id):
    image = sitk.Cast(meniscus_label, pixel_id)

    with pytest.raises(ValueError, match="must be sitkUInt8"):
        write_label_map(image, tmp_path / "seg.nrrd")


@pytest.mark.parametrize("pixel_id", [sitk.sitkInt8, sitk.sitkUInt16, sitk.sitkFloat32])
def test_read_rejects_non_uint8_image(tmp_path, meniscus_label, pixel_id):
    path = tmp_path / "seg.nrrd"
    sitk.WriteImage(sitk.Cast(meniscus_label, pixel_id), str(path))

    with pytest.raises(ValueError, match="must be sitkUInt8"):
        read_label_map(path)


def test_poly_data_round_trip_preserves_points_and_cells(tmp_path, meniscus_contours):
    path = tmp_path / "contours.vtp"

    write_poly_data(meniscus_contours, path)
    result = read_poly_data(path)

    assert np.array_equal(vtk_to_numpy(result.GetPoints().GetData()), vtk_to_numpy(meniscus_contours.GetPoints().GetData()))
    assert result.GetNumberOfLines() == meniscus_contours.GetNumberOfLines()
    assert np.array_equal(
        vtk_to_numpy(result.GetLines().GetConnectivityArray()),
        vtk_to_numpy(meniscus_contours.GetLines().GetConnectivityArray()),
    )
    assert np.array_equal(
        vtk_to_numpy(result.GetLines().GetOffsetsArray()),
        vtk_to_numpy(meniscus_contours.GetLines().GetOffsetsArray()),
    )


def test_write_poly_data_overwrites_existing_file(tmp_path, meniscus_contours):
    path = tmp_path / "contours.vtp"
    path.write_text("old contents")

    write_poly_data(meniscus_contours, path)

    assert read_poly_data(path).GetNumberOfLines() == meniscus_contours.GetNumberOfLines()


@pytest.mark.parametrize("filename", ["contours.vtk", "contours.stl", "contours"])
def test_poly_data_rejects_unsupported_format(tmp_path, meniscus_contours, filename):
    with pytest.raises(ValueError, match="Unsupported file format"):
        write_poly_data(meniscus_contours, tmp_path / filename)
    with pytest.raises(ValueError, match="Unsupported file format"):
        read_poly_data(tmp_path / filename)


def test_read_poly_data_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="No such file"):
        read_poly_data(tmp_path / "missing.vtp")


@pytest.mark.parametrize(
    "contents",
    ["not xml", '<?xml version="1.0"?><VTKFile type="ImageData" version="1.0"><ImageData></ImageData></VTKFile>'],
)
def test_read_poly_data_rejects_other_file_contents(tmp_path, contents):
    path = tmp_path / "contours.vtp"
    path.write_text(contents)

    with pytest.raises(ValueError, match="Not a VTK XML poly data"):
        read_poly_data(path)


def test_write_poly_data_raises_when_it_cannot_write(tmp_path, meniscus_contours):
    with pytest.raises(OSError, match="Could not write"):
        write_poly_data(meniscus_contours, tmp_path / "missing_directory" / "contours.vtp")


def test_brep_round_trip_preserves_text(tmp_path, meniscus_surface):
    path = tmp_path / "surface.brep"

    write_brep(meniscus_surface, path)

    assert read_brep(path) == meniscus_surface


@pytest.mark.parametrize("filename", ["surface.step", "surface.brp", "surface"])
def test_brep_rejects_unsupported_format(tmp_path, meniscus_surface, filename):
    with pytest.raises(ValueError, match="Unsupported file format"):
        write_brep(meniscus_surface, tmp_path / filename)
    with pytest.raises(ValueError, match="Unsupported file format"):
        read_brep(tmp_path / filename)


def test_read_brep_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="No such file"):
        read_brep(tmp_path / "missing.brep")


def test_read_brep_rejects_invalid_contents(tmp_path):
    path = tmp_path / "surface.brep"
    path.write_text("not a b-rep")

    with pytest.raises(ValueError, match="Not a valid B-rep"):
        read_brep(path)


def test_write_brep_raises_when_it_cannot_write(tmp_path):
    with pytest.raises(OSError):
        write_brep(BRep("text"), tmp_path / "missing_directory" / "surface.brep")
