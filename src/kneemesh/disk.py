"""Reading and writing LabelMaps as .nii, .nrrd, or .mhd + .raw, segmentations, poly data as .vtp, B-reps as .brep, and bone surfaces as .stl, and writing meshes."""

import json
from collections.abc import Mapping
from os import PathLike
from pathlib import Path

import gmsh
import numpy as np
import SimpleITK as sitk
from loguru import logger
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonCore import VTK_ID_TYPE, VTK_UNSIGNED_CHAR, vtkPoints, vtkStringArray
from vtkmodules.vtkCommonDataModel import (
    VTK_VERTEX,
    vtkCellArray,
    vtkCompositeDataSet,
    vtkDataObject,
    vtkFieldData,
    vtkMultiBlockDataSet,
    vtkPolyData,
    vtkUnstructuredGrid,
)
from vtkmodules.vtkIOGeometry import vtkSTLReader, vtkSTLWriter
from vtkmodules.vtkIOHDF import vtkHDFWriter
from vtkmodules.vtkIOXML import (
    vtkXMLPolyDataReader,
    vtkXMLPolyDataWriter,
    vtkXMLUnstructuredGridWriter,
)

from kneemesh.brep import BRep, gmsh_session
from kneemesh.segmentation import Segmentation
from kneemesh.tet_meshing import Mesh
from kneemesh.vol import LABEL_MAP_PIXEL_IDS

SUPPORTED_SUFFIXES = frozenset({".nii", ".nrrd", ".mhd"})
POLY_DATA_SUFFIXES = frozenset({".vtp"})
BREP_SUFFIXES = frozenset({".brep"})
SURFACE_SUFFIXES = frozenset({".stl"})
MESH_SET_SUFFIXES = frozenset({".vtu", ".vtkhdf"})
"""Mesh formats written with their node and element sets, and the formats of an assembled mesh."""

_HDF_CHUNK_SIZE = 4096
_HDF_COMPRESSION_LEVEL = 4
# The VTK cell type of each gmsh element, and the gmsh node at each of the VTK cell's nodes where their orders differ.
_VTK_CELLS: dict[str, tuple[int, list[int] | None]] = {
    "Triangle 3": (5, None),
    "Triangle 6": (22, None),
    "Quadrilateral 4": (9, None),
    "Quadrilateral 8": (23, None),
    "Quadrilateral 9": (28, None),
    "Tetrahedron 4": (10, None),
    "Tetrahedron 10": (24, [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]),
    "Hexahedron 8": (12, None),
    "Hexahedron 20": (25, [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 14, 15]),
    "Hexahedron 27": (29, [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 14, 15, 22, 23, 21, 24, 20, 25, 26]),
}


def read_label_map(path: str | PathLike[str]) -> sitk.Image:
    """Read a segmented LabelMap from disk.

    For a .mhd file, the .raw file it references must also exist.

    Raises:
        ValueError: If the file format is unsupported or the image is not
            sitkUInt8, sitkUInt16, or sitkInt16.
    """
    path = _validated_path(path)
    image = sitk.ReadImage(str(path))
    _require_label_map(image)
    logger.debug("Read {} LabelMap of size {} and spacing {} from {}", image.GetPixelIDTypeAsString(), image.GetSize(), image.GetSpacing(), path)
    return image


def write_label_map(image: sitk.Image, path: str | PathLike[str]) -> None:
    """Write a segmented LabelMap to disk, overwriting any existing file.

    A .mhd path also writes a .raw file alongside it.

    Raises:
        ValueError: If the file format is unsupported or the image is not
            sitkUInt8, sitkUInt16, or sitkInt16.
    """
    path = _validated_path(path)
    _require_label_map(image)
    sitk.WriteImage(image, str(path), useCompression=False)


def segmentation_labels_path(path: str | PathLike[str]) -> Path:
    """Return where write_segmentation puts the labels of a segmentation written to path: <stem>_labels.json beside it."""
    path = Path(path)
    return path.with_name(f"{path.stem}_labels.json")


def write_segmentation(segmentation: Segmentation, path: str | PathLike[str]) -> None:
    """Write a segmentation's LabelMap to path and its {name: value} labels as JSON beside it.

    The labels go to segmentation_labels_path(path), so the image can be read
    back with read_segmentation or given to a pipeline as label_map with those labels.

    Raises:
        ValueError: If the file format is unsupported.
        OSError: If a file cannot be written.
    """
    write_label_map(segmentation.image, path)
    labels_path = segmentation_labels_path(path)
    labels_path.write_text(json.dumps(segmentation.labels, indent=2) + "\n", encoding="utf-8")
    logger.debug("Wrote labels {} to {}", segmentation.labels, labels_path)


def read_segmentation(path: str | PathLike[str]) -> Segmentation:
    """Read a segmentation written by write_segmentation.

    Raises:
        ValueError: If the file format is unsupported, the image is not a
            LabelMap, or the labels are invalid for it.
        OSError: If the image or its labels file cannot be read.
    """
    labels = json.loads(segmentation_labels_path(path).read_text(encoding="utf-8"))
    return Segmentation(read_label_map(path), labels)


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
    logger.debug("Wrote B-rep to {}", path)


def read_surface(path: str | PathLike[str]) -> vtkPolyData:
    """Read a triangulated surface, such as a bone's, from a .stl file, merging coincident points.

    Raises:
        ValueError: If the suffix is not .stl or the file holds no triangles.
        FileNotFoundError: If the file does not exist.
    """
    path = _validated_path(path, SURFACE_SUFFIXES)
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {path}")
    reader = vtkSTLReader()
    reader.SetFileName(str(path))
    reader.MergingOn()
    reader.Update()
    surface = reader.GetOutput()
    if not surface.GetNumberOfPolys():
        raise ValueError(f"Not a triangulated surface: {path}")
    return surface


def write_surface(surface: vtkPolyData, path: str | PathLike[str]) -> None:
    """Write a triangulated surface to a binary .stl file, overwriting any existing file.

    Raises:
        ValueError: If the suffix is not .stl.
        OSError: If the file cannot be written.
    """
    path = _validated_path(path, SURFACE_SUFFIXES)
    writer = vtkSTLWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(surface)
    writer.SetFileTypeToBinary()
    if not writer.Write():
        raise OSError(f"Could not write the surface to {path}")
    logger.debug("Wrote surface to {}", path)


def write_mesh(mesh: Mesh, path: str | PathLike[str]) -> None:
    """Write a mesh, overwriting any existing file: with its sets as .vtu or .vtkhdf, or in another format gmsh infers from the extension.

    .vtu and .vtkhdf are written as write_meshes writes one structure, with
    set names as they are. gmsh's formats include .msh, .vtk, .inp, .unv,
    .bdf, .med, .mesh, and .su2, and .stl for meshes of triangles; they hold
    the elements only, without faces or sets.

    Raises:
        ValueError: If gmsh does not recognize the extension.
        OSError: If the file cannot be written.
    """
    path = Path(path)
    if path.suffix.lower() in MESH_SET_SUFFIXES:
        _write_vtk({"": mesh}, path)
        return
    with gmsh_session():
        dimension = max((gmsh.model.mesh.getElementProperties(element_type)[1] for element_type in mesh.element_types.values()), default=3)
        entity = gmsh.model.addDiscreteEntity(dimension)
        gmsh.model.mesh.addNodes(dimension, entity, np.arange(1, len(mesh.nodes) + 1).tolist(), mesh.nodes.ravel().tolist())
        first_tag = 1
        for name, connectivity in mesh.elements.items():
            element_tags = np.arange(first_tag, first_tag + len(connectivity))
            gmsh.model.mesh.addElementsByType(entity, mesh.element_types[name], element_tags.tolist(), (connectivity + 1).ravel().tolist())
            first_tag += len(connectivity)
        try:
            gmsh.write(str(path))
        except Exception as error:  # gmsh raises only the bare Exception type.
            if "Unknown output file format" in str(error):
                raise ValueError(f"gmsh cannot write meshes in the format of {path.name!r}") from error
            raise OSError(f"Could not write mesh to {path}") from error
    logger.debug("Wrote mesh to {}", path)


def write_meshes[Name: str](meshes: Mapping[Name, Mesh], path: str | PathLike[str]) -> None:
    """Write several structures' meshes as one, overwriting any existing file, as .vtu or .vtkhdf.

    Nodes and elements are numbered from 1 through the structures in order,
    each structure's elements before its faces, and each structure's sets
    are named "<structure>_<set>", such as "femur_surface". Every node and
    element carries its number, as "node_id" and "element_id".

    - .vtu: one unstructured grid of every element and face, with a
      "structure" cell array indexing the field array "structures", and a
      point array per node set and a cell array per element set, 1 for members.
    - .vtkhdf: a multiblock with a block per structure, by name, as one
      structure's .vtkhdf: an "elements" grid of its elements, and
      "surfaces" and "node_sets" blocks holding a grid per set, named
      "<structure>_<set>", of the set's faces or of vertices at its nodes.
      Every grid's "node_id" numbers the same node at the same coordinates.

    Raises:
        ValueError: If the extension is not .vtu or .vtkhdf, or no meshes are given.
        OSError: If the file cannot be written.
    """
    path = _validated_path(path, MESH_SET_SUFFIXES)
    if not meshes:
        raise ValueError("Give at least one mesh to write")
    _write_vtk(meshes, path)


def _write_vtk[Name: str](meshes: Mapping[Name, Mesh], path: Path) -> None:
    assembled = list(meshes) != [""]
    node_offset, element_offset = 0, 0
    parts = []
    for name, mesh in meshes.items():
        parts.append((name, mesh, node_offset, element_offset))
        node_offset += len(mesh.nodes)
        element_offset += mesh.element_count
    if path.suffix.lower() == ".vtu":
        output = _single_grid(parts, assembled)
        writer = vtkXMLUnstructuredGridWriter()
        writer.SetDataModeToAppended()
    else:
        if assembled:
            output = vtkMultiBlockDataSet()
            for index, (name, mesh, nodes, elements) in enumerate(parts):
                _set_block(output, index, name, _part_blocks(mesh, nodes, elements, f"{name}_"))
        else:
            output = _part_blocks(parts[0][1], 0, 0, "")
        writer = vtkHDFWriter()
        # The default chunks of 25,000 values pad each set's small arrays: the real knee's tetrahedra took 81 MB, and 10 MB so.
        writer.SetChunkSize(_HDF_CHUNK_SIZE)
        writer.SetCompressionLevel(_HDF_COMPRESSION_LEVEL)
    writer.SetInputData(output)
    writer.SetFileName(str(path))
    if not writer.Write():
        raise OSError(f"Could not write mesh to {path}")
    logger.debug("Wrote {} meshes to {}", len(parts), path)


def _numbered(mesh: Mesh) -> list[tuple[str, np.ndarray]]:
    """A mesh's elements then faces, by gmsh name, in the order of their numbers."""
    return [*mesh.elements.items(), *mesh.faces.items()]


def _grid(points: np.ndarray, cells: list[tuple[str, np.ndarray]]) -> vtkUnstructuredGrid:
    """An unstructured grid of points and cells given by gmsh name and connectivity into points."""
    grid = vtkUnstructuredGrid()
    vtk_points = vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float64), deep=True))
    grid.SetPoints(vtk_points)
    types, sizes, connectivity = [], [], []
    for name, rows in cells:
        if name not in _VTK_CELLS:
            raise ValueError(f"No VTK cell for {name} elements")
        cell_type, order = _VTK_CELLS[name]
        rows = rows if order is None else rows[:, order]
        types.append(np.full(len(rows), cell_type, dtype=np.uint8))
        sizes.append(np.full(len(rows), rows.shape[1], dtype=np.int64))
        connectivity.append(rows.ravel())
    offsets = np.concatenate([[0], np.cumsum(np.concatenate([np.zeros(0, np.int64), *sizes]))]).astype(np.int64)
    cell_array = vtkCellArray()
    cell_array.SetData(
        numpy_to_vtk(offsets, deep=True, array_type=VTK_ID_TYPE),
        numpy_to_vtk(np.concatenate([np.zeros(0, np.int64), *connectivity]).astype(np.int64), deep=True, array_type=VTK_ID_TYPE),
    )
    grid.SetCells(numpy_to_vtk(np.concatenate([np.zeros(0, np.uint8), *types]), deep=True, array_type=VTK_UNSIGNED_CHAR), cell_array)
    return grid


def _add_array(data: vtkFieldData, name: str, values: np.ndarray) -> None:
    array = numpy_to_vtk(np.ascontiguousarray(values), deep=True)
    array.SetName(name)
    data.AddArray(array)


def _single_grid(parts: list[tuple[str, Mesh, int, int]], assembled: bool) -> vtkUnstructuredGrid:
    points = np.concatenate([mesh.nodes for _, mesh, _, _ in parts])
    cells = [(name, rows + nodes) for _, mesh, nodes, _ in parts for name, rows in _numbered(mesh)]
    grid = _grid(points, cells)
    total_elements = sum(mesh.element_count for _, mesh, _, _ in parts)
    _add_array(grid.GetPointData(), "node_id", np.arange(1, len(points) + 1, dtype=np.int64))
    _add_array(grid.GetCellData(), "element_id", np.arange(1, total_elements + 1, dtype=np.int64))
    if assembled:
        _add_array(grid.GetCellData(), "structure", np.repeat(np.arange(len(parts), dtype=np.int32), [mesh.element_count for _, mesh, _, _ in parts]))
        names = vtkStringArray()
        names.SetName("structures")
        for name, *_ in parts:
            names.InsertNextValue(name)
        grid.GetFieldData().AddArray(names)
    for name, mesh, nodes, elements in parts:
        prefix = f"{name}_" if assembled else ""
        for set_name, members in mesh.node_sets.items():
            mask = np.zeros(len(points), dtype=np.uint8)
            mask[nodes + members] = 1
            _add_array(grid.GetPointData(), prefix + set_name, mask)
        for set_name, members in mesh.element_sets.items():
            mask = np.zeros(total_elements, dtype=np.uint8)
            mask[elements + members] = 1
            _add_array(grid.GetCellData(), prefix + set_name, mask)
    return grid


def _subgrid(mesh: Mesh, cells: list[tuple[str, np.ndarray]], node_offset: int, element_ids: np.ndarray) -> vtkUnstructuredGrid:
    """A grid of some of a mesh's cells and only the nodes they use, carrying their numbers."""
    used = np.unique(np.concatenate([np.zeros(0, np.int64), *(rows.ravel() for _, rows in cells)]))
    renumber = np.zeros(len(mesh.nodes), dtype=np.int64)
    renumber[used] = np.arange(len(used))
    grid = _grid(mesh.nodes[used], [(name, renumber[rows]) for name, rows in cells])
    _add_array(grid.GetPointData(), "node_id", node_offset + used + 1)
    _add_array(grid.GetCellData(), "element_id", element_ids)
    return grid


def _part_blocks(mesh: Mesh, node_offset: int, element_offset: int, prefix: str) -> vtkMultiBlockDataSet:
    """One structure's blocks: its volume elements, its face sets as "surfaces", and its node sets, set names prefixed."""
    numbered = _numbered(mesh)
    starts = np.cumsum([0, *(len(rows) for _, rows in numbered)])
    part = vtkMultiBlockDataSet()
    volume = sum(len(rows) for rows in mesh.elements.values())
    _set_block(part, 0, "elements", _subgrid(mesh, list(mesh.elements.items()), node_offset, element_offset + np.arange(1, volume + 1)))
    surfaces = vtkMultiBlockDataSet()
    for index, (name, members) in enumerate(mesh.element_sets.items()):
        members = np.sort(members)
        cells = [
            (kind, rows[members[(members >= start) & (members < end)] - start])
            for (kind, rows), start, end in zip(numbered, starts[:-1], starts[1:], strict=True)
        ]
        cells = [(kind, rows) for kind, rows in cells if len(rows)]
        _set_block(surfaces, index, prefix + name, _subgrid(mesh, cells, node_offset, element_offset + members + 1))
    _set_block(part, 1, "surfaces", surfaces)
    node_sets = vtkMultiBlockDataSet()
    for index, (name, members) in enumerate(mesh.node_sets.items()):
        grid = _grid(mesh.nodes[members], [])
        vertices = vtkCellArray()
        for point in range(len(members)):
            vertices.InsertNextCell(1, [point])
        grid.SetCells(VTK_VERTEX, vertices)
        _add_array(grid.GetPointData(), "node_id", node_offset + members + 1)
        _set_block(node_sets, index, prefix + name, grid)
    _set_block(part, 2, "node_sets", node_sets)
    return part


def _set_block(blocks: vtkMultiBlockDataSet, index: int, name: str, block: vtkDataObject) -> None:
    blocks.SetBlock(index, block)
    blocks.GetMetaData(index).Set(vtkCompositeDataSet.NAME(), name)


def _validated_path(path: str | PathLike[str], suffixes: frozenset[str] = SUPPORTED_SUFFIXES) -> Path:
    path = Path(path)
    if path.suffix.lower() not in suffixes:
        supported = ", ".join(sorted(suffixes))
        raise ValueError(f"Unsupported file format {path.name!r}; expected one of: {supported}")
    return path


def _require_label_map(image: sitk.Image) -> None:
    if image.GetPixelID() not in LABEL_MAP_PIXEL_IDS:
        raise ValueError(f"LabelMap must be sitkUInt8, sitkUInt16, or sitkInt16, got {image.GetPixelIDTypeAsString()}")
