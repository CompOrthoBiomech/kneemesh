import pytest
import SimpleITK as sitk

from meniscus2brep.disk import read_label_map, write_label_map


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
