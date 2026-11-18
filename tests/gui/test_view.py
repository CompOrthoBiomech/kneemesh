from dataclasses import replace

import numpy as np
import pytest
from vtkmodules.vtkFiltersGeometry import vtkUnstructuredGridGeometryFilter

from kneemesh.disk import write_mesh
from kneemesh.gui.labels import LabelEntry
from kneemesh.gui.view import Scene, edge_lines, read_mesh_preview
from kneemesh.mesh_optimization import element_quality
from kneemesh.tet_meshing import Mesh


def _cube_mesh(path, lift=0.0, columns=2):
    """Hexahedra in a row along x, with their faces and quality, written as a preview; lift raises the last top corner."""
    xs = np.arange(columns + 1, dtype=float)
    nodes = np.array([[x, y, z] for z in (0.0, 1.0) for y in (0.0, 1.0) for x in xs])
    nodes[-1, 2] += lift
    width = columns + 1

    def node(i, j, k):
        return k * 2 * width + j * width + i

    corners = [
        [node(i, 0, 0), node(i + 1, 0, 0), node(i + 1, 1, 0), node(i, 1, 0), node(i, 0, 1), node(i + 1, 0, 1), node(i + 1, 1, 1), node(i, 1, 1)]
        for i in range(columns)
    ]
    mesh = Mesh(nodes, {"Hexahedron 8": np.array(corners)}, {"Hexahedron 8": 5}).with_face_sets()
    write_mesh(replace(mesh, quality=element_quality(mesh)), path)
    return read_mesh_preview(path)


def _visible(actors):
    return [actor.GetVisibility() for actor in actors]


def test_meshes_are_colored_by_a_metric_with_a_scalar_bar(tmp_path):
    scene = Scene()
    scene.set_mesh("lateral_meniscus", _cube_mesh(tmp_path / "a.vtu"))
    scene.set_mesh("medial_meniscus", _cube_mesh(tmp_path / "b.vtu", lift=0.5))
    assert scene.metrics() == ["scaled_jacobian", "frobenius_aspect", "warp"]

    scene.refresh([], "latest", [], metric="scaled_jacobian")
    assert scene.scalar_bar.GetVisibility()
    low, high = scene.lookup_table.GetTableRange()
    assert low < 1.0
    assert np.isclose(high, 1.0)
    mapper = scene._stages[("mesh", "medial_meniscus")].actors[0].GetMapper()
    assert mapper.GetScalarVisibility()
    # The element metrics are drawn on the elements' outer faces, the warp on the boundary faces.
    mapper.Update()
    assert mapper.GetInput().GetNumberOfCells() == 10
    assert mapper.GetInput().GetPointData().GetNormals() is not None

    scene.refresh([], "latest", [], metric="warp")
    assert scene.lookup_table.GetTableRange()[0] == 0.0
    assert scene.lookup_table.GetTableRange()[1] > 10

    # Only the shown meshes set the range.
    scene.refresh([], "latest", [], solo="lateral_meniscus", metric="scaled_jacobian")
    assert np.allclose(scene.lookup_table.GetTableRange()[0], 1.0)

    scene.refresh([], "latest", [])
    assert not scene.scalar_bar.GetVisibility()
    assert not mapper.GetScalarVisibility()


def test_solo_hides_every_other_structure_and_its_labels(tmp_path):
    scene = Scene()
    scene.set_mesh("lateral_meniscus", _cube_mesh(tmp_path / "a.vtu"))
    scene.set_mesh("medial_meniscus", _cube_mesh(tmp_path / "b.vtu"))
    mapper = scene._stages[("mesh", "lateral_meniscus")].actors[0].GetMapper()
    mapper.Update()
    surface = mapper.GetInput()
    entries = [
        LabelEntry(id=1, path="a.nii", name="a.nii", file_value=1, structure="tibia", value=1, include=True),
        LabelEntry(id=2, path="a.nii", name="a.nii", file_value=2, structure=None, value=2, include=True),
    ]
    for entry in entries:
        scene.set_label(entry["id"], surface)

    scene.refresh(entries, "latest", [], solo="medial_meniscus")
    others = scene._stages[("mesh", "lateral_meniscus")].actors + scene._labels[1].actors + scene._labels[2].actors
    assert _visible(others) == [False, False, False]
    assert _visible(scene._stages[("mesh", "medial_meniscus")].actors) == [True]
    scene.refresh(entries, "latest", ["medial_meniscus"])
    assert _visible(others) == [True, True, True]


def _cells(actor):
    mapper = actor.GetMapper()
    mapper.Update()
    return mapper.GetInput().GetNumberOfCells()


def test_crinkle_clip_keeps_whole_cells_beyond_the_plane(tmp_path):
    scene = Scene()
    scene.set_mesh("lateral_meniscus", _cube_mesh(tmp_path / "a.vtu", columns=6))
    actor = scene._stages[("mesh", "lateral_meniscus")].actors[0]
    assert _cells(actor) == 6 * 4 + 2
    camera = scene.renderer.GetActiveCamera()
    camera.SetFocalPoint(3.0, 0.5, 0.5)
    camera.SetPosition(10.0, 0.5, 0.5)

    # Facing the viewer at +x through x = 3, the clip keeps the three columns behind it, and the column the plane touches, whole.
    scene.set_clipping(True)
    assert _cells(actor) == 4 * 4 + 2
    scene.set_clipping(True, offset=0.5)
    assert _cells(actor) < 4 * 4 + 2

    # Fixed, the plane stays where it was when the camera turns; following the view, it turns too.
    scene.set_clipping(True, "fixed")
    fixed = _cells(actor)
    camera.SetPosition(3.0, 10.0, 0.5)
    scene.render_window.Render()
    assert _cells(actor) == fixed
    scene.set_clipping(True, "view")
    scene.render_window.Render()
    assert _cells(actor) == 6 * 4 + 2
    scene.set_clipping(False)
    assert _cells(actor) == 6 * 4 + 2


def test_reset_camera_fits_what_is_shown_unclipped(tmp_path):
    scene = Scene()
    scene.set_mesh("lateral_meniscus", _cube_mesh(tmp_path / "a.vtu", columns=6))
    camera = scene.renderer.GetActiveCamera()
    camera.SetFocalPoint(3.0, 0.5, 0.5)
    camera.SetPosition(10.0, 0.5, 0.5)
    scene.reset_camera()
    unclipped = (camera.GetFocalPoint(), camera.GetDistance())

    # Fitting the clipped cells would close in on the half left; the plane follows the camera, so each render would clip closer.
    scene.set_clipping(True, offset=0.5)
    scene.render_window.Render()
    scene.reset_camera()
    scene.render_window.Render()
    assert np.allclose(camera.GetFocalPoint(), unclipped[0])
    assert camera.GetDistance() == pytest.approx(unclipped[1])


def test_nonlinear_meshes_draw_curved_edges_without_the_subdivision(tmp_path):
    corners = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    pairs = [(0, 1), (1, 2), (2, 0), (3, 0), (3, 2), (3, 1)]
    nodes = np.vstack([corners, [(corners[a] + corners[b]) / 2 + [0.0, 0.0, 0.1] for a, b in pairs]])
    mesh = Mesh(nodes, {"Tetrahedron 10": np.arange(10)[None]}, {"Tetrahedron 10": 11}).with_face_sets()
    write_mesh(replace(mesh, quality=element_quality(mesh)), tmp_path / "t.vtu")
    scene = Scene()
    scene.set_mesh("lateral_meniscus", read_mesh_preview(tmp_path / "t.vtu"))
    surface, edges = scene._stages[("mesh", "lateral_meniscus")].actors
    # Four faces subdivided twice, 16 triangles each, and the six edges as curves of five points.
    assert _cells(surface) == 4 * 16
    assert not surface.GetProperty().GetEdgeVisibility()
    assert _cells(edges) == 6
    lines = edges.GetMapper().GetInput()
    assert lines.GetNumberOfPoints() == 6 * 5
    middle = np.array(lines.GetPoint(2))
    assert np.isclose(middle[2], 0.1)


def test_edge_lines_of_linear_cells_are_their_corner_pairs(tmp_path):
    grid = _cube_mesh(tmp_path / "a.vtu", columns=1)
    boundary = vtkUnstructuredGridGeometryFilter()
    boundary.SetInputData(grid)
    boundary.Update()
    lines = edge_lines(boundary.GetOutput())
    assert lines.GetNumberOfLines() == 12
    assert lines.GetNumberOfPoints() == 24
