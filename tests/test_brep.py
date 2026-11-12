
import gmsh
import numpy as np
import pytest
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkCommonDataModel import VTK_POLY_LINE

from kneemesh.brep import contours_to_polydata, gmsh_session
from kneemesh.vol import ContourSlice


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
