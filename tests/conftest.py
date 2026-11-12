import math
from pathlib import Path

import pytest
import SimpleITK as sitk
from scipy.interpolate import BSpline
from vtkmodules.vtkCommonDataModel import vtkPolyData

from kneemesh.brep import BRep, contours_to_polydata
from kneemesh.disk import read_label_map
from kneemesh.geo.meniscus import centerline, contours_to_brep, contours_to_hex_blocks, slice_isocontour
from kneemesh.mesh.meniscus import hexahedral_mesh
from kneemesh.tet_meshing import Mesh, tetrahedral_mesh
from kneemesh.vol import ContourSlice, extract_isocontour, signed_distance_field, smooth_isocontour

INNER_RADIUS = 10.0
OUTER_RADIUS = 20.0
HEIGHT = 8.0
SWEEP_DEGREES = 270.0


@pytest.fixture(scope="session")
def meniscus_label() -> sitk.Image:
    """Approximate knee meniscus as a sitkUInt8 LabelMap with label 1.

    A triangular (r, z) cross section, with vertices (INNER_RADIUS, -HEIGHT / 2),
    (OUTER_RADIUS, -HEIGHT / 2), and (OUTER_RADIUS, HEIGHT / 2), swept
    SWEEP_DEGREES about the z axis through the image center. The sweep is
    centered on +x, leaving the gap facing -x. Physical units are mm.
    """
    source = sitk.PhysicalPointImageSource()
    source.SetOutputPixelType(sitk.sitkVectorFloat64)
    source.SetSize((101, 101, 17))
    source.SetSpacing((0.5, 0.5, 1.0))
    source.SetOrigin((-25.0, -25.0, -8.0))
    points = source.Execute()

    x = sitk.VectorIndexSelectionCast(points, 0)
    y = sitk.VectorIndexSelectionCast(points, 1)
    z = sitk.VectorIndexSelectionCast(points, 2)

    radius = sitk.Sqrt(x * x + y * y)
    angle = sitk.Abs(sitk.Atan2(y, x))
    height_above_base = z + HEIGHT / 2
    rim_fraction = (radius - INNER_RADIUS) / (OUTER_RADIUS - INNER_RADIUS)

    cross_section = (radius <= OUTER_RADIUS) & (height_above_base >= 0.0) & (height_above_base <= HEIGHT * rim_fraction)
    return cross_section & (angle <= math.radians(SWEEP_DEGREES / 2))


@pytest.fixture(scope="session")
def meniscus_centerline(meniscus_label: sitk.Image) -> BSpline:
    """Centerline of meniscus_label."""
    return centerline(meniscus_label)


@pytest.fixture(scope="session")
def meniscus_distance_field(meniscus_label: sitk.Image) -> sitk.Image:
    """Signed distance field of meniscus_label."""
    return signed_distance_field(meniscus_label)


@pytest.fixture(scope="session")
def meniscus_mesh(meniscus_distance_field: sitk.Image) -> vtkPolyData:
    """Surface mesh on the boundary of meniscus_label."""
    return extract_isocontour(meniscus_distance_field)


@pytest.fixture(scope="session")
def meniscus_smoothed_mesh(meniscus_mesh: vtkPolyData) -> vtkPolyData:
    """meniscus_mesh after default windowed sinc smoothing."""
    return smooth_isocontour(meniscus_mesh)


@pytest.fixture(scope="session")
def meniscus_slices(meniscus_mesh: vtkPolyData, meniscus_centerline: BSpline) -> list[ContourSlice]:
    """Slices of meniscus_mesh every 5 mm along meniscus_centerline."""
    return slice_isocontour(meniscus_mesh, meniscus_centerline, spacing=5.0)


@pytest.fixture(scope="session")
def meniscus_contours(meniscus_slices: list[ContourSlice]) -> vtkPolyData:
    """Slices of meniscus_mesh every 5 mm along meniscus_centerline."""
    return contours_to_polydata(meniscus_slices)


@pytest.fixture(scope="session")
def meniscus_surface(meniscus_slices: list[ContourSlice]) -> BRep:
    """Open cubic surface lofted through meniscus_slices."""
    return contours_to_brep(meniscus_slices)


@pytest.fixture(scope="session")
def meniscus_solid(meniscus_slices: list[ContourSlice]) -> BRep:
    """Cubic solid lofted through meniscus_slices with planar caps."""
    return contours_to_brep(meniscus_slices, cap="planar")


@pytest.fixture(scope="session")
def meniscus_domed_solid(meniscus_slices: list[ContourSlice]) -> BRep:
    """Cubic solid lofted through meniscus_slices with smooth caps."""
    return contours_to_brep(meniscus_slices, cap="smooth")


@pytest.fixture(scope="session")
def meniscus_hex_slices(meniscus_mesh: vtkPolyData, meniscus_centerline: BSpline) -> list[ContourSlice]:
    """10-point slices of meniscus_mesh every 5 mm, for building hexahedral blocks."""
    return slice_isocontour(meniscus_mesh, meniscus_centerline, spacing=5.0, points_per_slice=10)


@pytest.fixture(scope="session")
def meniscus_hex_blocks(meniscus_hex_slices: list[ContourSlice]) -> BRep:
    """Three-block solid for hexahedral meshing, built from meniscus_hex_slices."""
    return contours_to_hex_blocks(meniscus_hex_slices)


@pytest.fixture(scope="session")
def real_label() -> sitk.Image:
    """A real meniscus LabelMap (label 1) whose arc touches the edge of the image."""
    return read_label_map(Path(__file__).parent / "dat" / "mns_l.nii")


@pytest.fixture(scope="session")
def meniscus_tetrahedra(meniscus_solid: BRep) -> Mesh:
    """Linear tetrahedral mesh of meniscus_solid with a 2 mm maximum size."""
    return tetrahedral_mesh(meniscus_solid, max_size=2.0)


@pytest.fixture(scope="session")
def meniscus_hexahedra(meniscus_hex_blocks: BRep) -> Mesh:
    """Linear hexahedral mesh of meniscus_hex_blocks, 4 elements per cap edge and 30 along."""
    return hexahedral_mesh(meniscus_hex_blocks, elements_per_edge=4, elements_along=30)
