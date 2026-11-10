"""Generation of B-rep geometry."""

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData, vtkPolyLine

from meniscus2brep.vol import ContourSlice


def contours_to_polydata(slices: list[ContourSlice]) -> vtkPolyData:
    """Assemble contour slices into a single poly data of closed polylines.

    Args:
        slices: Contours from slice_isocontour.

    Returns:
        A vtkPolyData with one vtkPolyLine per slice, in the order given. Each
        polyline ends on its own first point id to close the contour, so no
        point is duplicated.

    Raises:
        ValueError: If slices is empty.
    """
    if not slices:
        raise ValueError("No contour slices to assemble")

    points = vtkPoints()
    points.SetData(numpy_to_vtk(np.vstack([contour.points for contour in slices]), deep=True))

    lines = vtkCellArray()
    first_id = 0
    for contour in slices:
        count = len(contour.points)
        polyline = vtkPolyLine()
        ids = polyline.GetPointIds()
        ids.SetNumberOfIds(count + 1)
        for index in range(count):
            ids.SetId(index, first_id + index)
        ids.SetId(count, first_id)
        lines.InsertNextCell(polyline)
        first_id += count

    poly_data = vtkPolyData()
    poly_data.SetPoints(points)
    poly_data.SetLines(lines)
    return poly_data
