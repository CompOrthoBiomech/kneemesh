"""Operations shared by every cartilage: telling its articular side from the side against its bone."""

from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
from loguru import logger
from scipy.ndimage import map_coordinates
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkFiltersCore import vtkPolyDataConnectivityFilter, vtkPolyDataNormals

DEFAULT_SIDE_COSINE = 0.3
"""Default cosine between a surface normal and the bone's distance gradient beyond which a point is on one side."""


@dataclass(frozen=True, eq=False)
class CartilageSides:
    """Points of a cartilage surface on its articular side and on the side against its bone.

    Points whose normals run roughly along the bone's surface, around the
    cartilage's perimeter, belong to neither.

    Attributes:
        articular: (N, 3) physical points whose outward normals point away from the bone.
        bone: (M, 3) physical points whose outward normals point toward the bone.
    """

    articular: np.ndarray
    bone: np.ndarray


def classify_sides(surface: vtkPolyData, bone_distance_field: sitk.Image, side_cosine: float = DEFAULT_SIDE_COSINE) -> CartilageSides:
    """Split a closed cartilage surface into its articular and bone sides.

    Only the largest connected piece of the surface is used, so small islands
    of the segmentation are dropped. Each point's outward normal is compared
    with the gradient of the bone's distance field there, which points away
    from the bone wherever the bone is. This works for any cartilage, unlike
    comparing normals with one axis, which cannot separate the sides of the
    femoral cartilage wrapping the condyles.

    Args:
        surface: Closed cartilage surface from extract_isocontour or smooth_isocontour.
        bone_distance_field: Signed distance field of the bone from signed_distance_field.
        side_cosine: A point is on the articular side when the cosine between
            its normal and the gradient is at least this, and on the bone side
            when it is at most the negative of this. Higher values drop more
            of the steep perimeter between the sides.

    Returns:
        The points on each side.

    Raises:
        ValueError: If side_cosine is not in (0, 1), or a side has no points.
    """
    if not 0 < side_cosine < 1:
        raise ValueError(f"side_cosine must be in (0, 1), got {side_cosine}")

    connectivity = vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(surface)
    connectivity.SetExtractionModeToLargestRegion()
    connectivity.Update()
    normals = vtkPolyDataNormals()
    normals.SetInputConnection(connectivity.GetOutputPort())
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOff()
    normals.SplittingOff()
    normals.ConsistencyOn()
    # A closed surface's normals can be oriented outward reliably.
    normals.AutoOrientNormalsOn()
    normals.Update()
    output = normals.GetOutput()
    points = vtk_to_numpy(output.GetPoints().GetData()).astype(float)
    point_normals = vtk_to_numpy(output.GetPointData().GetNormals()).astype(float)

    gradients = field_gradient(bone_distance_field, points)
    lengths = np.linalg.norm(gradients, axis=1)
    cosines = np.einsum("ij,ij->i", point_normals, gradients) / np.where(lengths > 0, lengths, 1.0)
    articular, bone = points[cosines >= side_cosine], points[cosines <= -side_cosine]
    logger.debug(
        "Classified {} of {} surface points as articular and {} as against the bone, at cosine {}",
        len(articular),
        len(points),
        len(bone),
        side_cosine,
    )
    if len(articular) == 0 or len(bone) == 0:
        raise ValueError("The cartilage surface has no points facing away from or toward its bone; is the bone label right?")
    return CartilageSides(articular, bone)


def field_value(image: sitk.Image, points: np.ndarray) -> np.ndarray:
    """Return a scalar image's trilinearly interpolated values at (N, 3) physical points, clamped at its edges."""
    return map_coordinates(sitk.GetArrayViewFromImage(image), _array_coordinates(image, points), order=1, mode="nearest")


def field_gradient(image: sitk.Image, points: np.ndarray) -> np.ndarray:
    """Return a scalar image's physical gradient at (N, 3) physical points, interpolated trilinearly."""
    array = sitk.GetArrayViewFromImage(image)
    coordinates = _array_coordinates(image, points)
    # np.gradient differentiates along the z, y, x array axes; reverse them to the image's x, y, z index axes.
    index_gradients = [
        map_coordinates(component, coordinates, order=1, mode="nearest") for component in np.gradient(array, *image.GetSpacing()[::-1])
    ][::-1]
    direction = np.reshape(image.GetDirection(), (3, 3))
    return np.column_stack(index_gradients) @ direction.T


def _array_coordinates(image: sitk.Image, points: np.ndarray) -> np.ndarray:
    """Return (3, N) continuous z, y, x array coordinates of (N, 3) physical points."""
    direction = np.reshape(image.GetDirection(), (3, 3))
    indices = (points - np.asarray(image.GetOrigin())) @ direction / np.asarray(image.GetSpacing())
    return indices[:, ::-1].T
