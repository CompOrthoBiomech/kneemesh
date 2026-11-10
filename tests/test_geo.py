import re

import gmsh
import numpy as np
import pytest
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkCommonDataModel import VTK_POLY_LINE

from meniscus2brep.disk import write_brep
from meniscus2brep.geo import contours_to_brep, contours_to_polydata, gmsh_session
from meniscus2brep.vol import ContourSlice


def cell_ids(poly_data, cell):
    ids = poly_data.GetCell(cell).GetPointIds()
    return [ids.GetId(index) for index in range(ids.GetNumberOfIds())]


def make_slice(points):
    axes = np.eye(3)
    return ContourSlice(np.array(points, dtype=float), np.zeros(3), axes[0], axes[1], axes[2], 1.0)


def test_poly_data_has_one_polyline_per_slice(meniscus_slices, meniscus_contours):
    assert meniscus_contours.GetNumberOfCells() == len(meniscus_slices)
    assert meniscus_contours.GetNumberOfLines() == len(meniscus_slices)
    for cell in range(meniscus_contours.GetNumberOfCells()):
        assert meniscus_contours.GetCellType(cell) == VTK_POLY_LINE


def test_poly_data_keeps_slice_points_in_order(meniscus_slices, meniscus_contours):
    expected = np.vstack([contour.points for contour in meniscus_slices])

    assert np.array_equal(vtk_to_numpy(meniscus_contours.GetPoints().GetData()), expected)


def test_polylines_are_closed_without_duplicate_points():
    slices = [make_slice([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), make_slice([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])]

    poly_data = contours_to_polydata(slices)

    assert poly_data.GetNumberOfPoints() == 7
    assert cell_ids(poly_data, 0) == [0, 1, 2, 0]
    assert cell_ids(poly_data, 1) == [3, 4, 5, 6, 3]


def test_poly_data_rejects_empty_slices():
    with pytest.raises(ValueError, match="No contour slices"):
        contours_to_polydata([])


def import_brep(brep, tmp_path):
    path = tmp_path / "shape.brep"
    write_brep(brep, path)
    gmsh.model.occ.importShapes(str(path))
    gmsh.model.occ.synchronize()


def entity_counts():
    return [len(gmsh.model.getEntities(dim)) for dim in range(4)]


def spline_surface_degrees(brep):
    """Return (around-contour, along-curve) degrees of each B-spline surface in the file."""
    records = re.findall(r"^9 \d \d \d \d (\d+) (\d+) ", brep.text, flags=re.MULTILINE)
    return [(int(around), int(along)) for around, along in records]


def test_surface_is_one_open_face(meniscus_surface, tmp_path):
    with gmsh_session():
        import_brep(meniscus_surface, tmp_path)

        assert entity_counts()[2:] == [1, 0]


def test_solid_has_caps_and_the_label_volume(meniscus_solid, meniscus_slices, tmp_path):
    with gmsh_session():
        import_brep(meniscus_solid, tmp_path)

        assert entity_counts()[2:] == [3, 1]
        mean_area = np.mean([contour.area for contour in meniscus_slices])
        length = 5.0 * (len(meniscus_slices) - 1)
        assert gmsh.model.occ.getMass(3, gmsh.model.getEntities(3)[0][1]) == pytest.approx(mean_area * length, rel=0.03)


def test_surface_passes_through_every_contour_point(meniscus_surface, meniscus_slices, tmp_path):
    points = np.vstack([contour.points for contour in meniscus_slices])

    with gmsh_session():
        import_brep(meniscus_surface, tmp_path)
        face = gmsh.model.getEntities(2)[0][1]
        closest = np.array(gmsh.model.getClosestPoint(2, face, points.ravel().tolist())[0]).reshape(-1, 3)

    assert np.linalg.norm(closest - points, axis=1).max() < 1e-3


def test_surface_stays_near_the_label(meniscus_surface, meniscus_distance_field, tmp_path):
    with gmsh_session():
        import_brep(meniscus_surface, tmp_path)
        face = gmsh.model.getEntities(2)[0][1]
        parameters = np.array([[u, v] for u in np.linspace(0, 1, 200) for v in np.linspace(0, 1, 150)])
        samples = np.array(gmsh.model.getValue(2, face, parameters.ravel().tolist())).reshape(-1, 3)

    field = meniscus_distance_field
    distances = [abs(field[field.TransformPhysicalPointToIndex(point.tolist())]) for point in samples]
    # Distances are sampled from the nearest voxel of a field with 1 mm z spacing.
    assert max(distances) <= 1.25


@pytest.mark.parametrize("order", [2, 3, 4, 5])
def test_spline_order_sets_surface_degree(meniscus_slices, order):
    degrees = spline_surface_degrees(contours_to_brep(meniscus_slices, spline_order=order))

    assert len(degrees) == 1
    assert degrees[0][0] == order
    assert 1 <= degrees[0][1] <= order


def test_spline_order_one_is_ruled_between_slices(meniscus_slices):
    degrees = spline_surface_degrees(contours_to_brep(meniscus_slices, spline_order=1))

    assert degrees == [(1, 1)] * (len(meniscus_slices) - 1)


def test_solid_can_be_meshed_by_gmsh(meniscus_solid, tmp_path):
    with gmsh_session():
        import_brep(meniscus_solid, tmp_path)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 2.0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 4.0)
        gmsh.model.mesh.generate(3)

        tetrahedra = gmsh.model.mesh.getElementsByType(4)[0]

    assert len(tetrahedra) > 0


def test_gmsh_session_finalizes_gmsh_it_started():
    with gmsh_session():
        assert gmsh.isInitialized()

    assert not gmsh.isInitialized()


def test_gmsh_session_leaves_existing_gmsh_running():
    gmsh.initialize(readConfigFiles=False, interruptible=False)
    try:
        gmsh.model.add("existing")
        with gmsh_session():
            assert gmsh.model.getCurrent() != "existing"
        assert gmsh.isInitialized()
        assert gmsh.model.getCurrent() == "existing"
    finally:
        gmsh.finalize()


def test_brep_rejects_too_few_slices(meniscus_slices):
    with pytest.raises(ValueError, match="at least 2"):
        contours_to_brep(meniscus_slices[:1])


@pytest.mark.parametrize("order", [0, 26])
def test_brep_rejects_invalid_spline_order(meniscus_slices, order):
    with pytest.raises(ValueError, match="spline_order"):
        contours_to_brep(meniscus_slices, spline_order=order)


def test_brep_rejects_contours_too_short_for_order():
    slices = [make_slice([[0, 0, z], [1, 0, z], [0, 1, z]]) for z in (0.0, 1.0)]

    with pytest.raises(ValueError, match="more than 3 points"):
        contours_to_brep(slices, spline_order=3)
