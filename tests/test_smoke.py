import SimpleITK as sitk

import kneemesh


def test_package_has_version():
    assert kneemesh.__version__


def test_simpleitk_creates_image():
    image = sitk.Image(4, 3, 2, sitk.sitkUInt8)

    assert image.GetSize() == (4, 3, 2)
