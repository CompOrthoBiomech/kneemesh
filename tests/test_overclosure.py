from dataclasses import replace

import numpy as np
import pytest
from vtkmodules.vtkFiltersCore import vtkFeatureEdges
from vtkmodules.vtkFiltersSources import vtkSphereSource

from kneemesh.geo.bone import surface_arrays
from kneemesh.geo.meniscus import contours_to_brep
from kneemesh.geo.sheet import PlaneFrame, sheet_to_brep
from kneemesh.overclosure import (
    Adjustable,
    Obstacle,
    brep_boundary,
    push_meniscus_out,
    push_surface_out,
    resolve_overclosure,
    surface_obstacle,
)


def sphere(center, radius, resolution=48):
    source = vtkSphereSource()
    source.SetCenter(*center)
    source.SetRadius(radius)
    source.SetThetaResolution(resolution)
    source.SetPhiResolution(resolution)
    source.Update()
    return source.GetOutput()


def overlap(first, second):
    """How deep two obstacles reach into each other, through either's vertices."""
    return max(first.depth(second.points), second.depth(first.points))


def test_obstacle_measures_signed_distances_whatever_the_triangles_orientation():
    obstacle = Obstacle(sphere((0, 0, 0), 5.0))
    signed, nearest = obstacle.distances(np.array([[0.0, 0.0, 4.0], [0.0, 0.0, 6.0], [50.0, 0.0, 0.0]]), reach=2.0)

    assert signed[0] == pytest.approx(-1.0, abs=0.02)
    assert signed[1] == pytest.approx(1.0, abs=0.02)
    # Points far outside the bounding box are not measured.
    assert np.isinf(signed[2])
    np.testing.assert_allclose(nearest[:2, 2], [5.0, 5.0], atol=0.02)


def test_obstacle_pushes_points_to_clearance_outside():
    obstacle = Obstacle(sphere((0, 0, 0), 5.0))
    points = np.array([[0.0, 0.0, 4.0], [0.0, 0.0, 5.05], [0.0, 0.0, 7.0]])

    moved = points + obstacle.pushes(points, clearance=0.1)

    np.testing.assert_allclose(moved[:2, 2], [5.1, 5.1], atol=0.02)
    np.testing.assert_array_equal(moved[2], points[2])


def test_brep_boundary_leaves_out_faces_between_blocks(meniscus_hex_blocks):
    edges = vtkFeatureEdges()
    edges.SetInputData(brep_boundary(meniscus_hex_blocks))
    edges.BoundaryEdgesOn()
    edges.NonManifoldEdgesOn()
    edges.FeatureEdgesOff()
    edges.ManifoldEdgesOff()
    edges.Update()

    # Faces between the butterfly's blocks would meet the outer faces at non-manifold edges.
    assert edges.GetOutput().GetNumberOfCells() == 0


def test_surface_is_pushed_out_of_an_obstacle_between_its_vertices():
    obstacle = Obstacle(sphere((0, 0, 0), 10.0, resolution=96))
    # A coarse sphere whose bottom reaches 0.5 into the obstacle: its facets also cut into it between vertices.
    surface = sphere((0, 0, 19.5), 10.0, resolution=16)

    moved = push_surface_out(surface, [obstacle], clearance=0.0)

    before, after = surface_arrays(surface)[0], surface_arrays(moved)[0]
    assert overlap(obstacle, Obstacle(surface)) > 0.4
    assert overlap(obstacle, Obstacle(moved)) < 0.005
    # Only the vertices around the contact move.
    assert np.count_nonzero(np.linalg.norm(after - before, axis=1)) < len(before) / 4


def bump_under(boundary, depth=0.5, radius=6.0):
    """A sphere reaching depth up into the bottom of a meniscus's boundary where it crosses the +x axis."""
    points = boundary.points
    near_axis = points[(np.abs(points[:, 1]) < 1.0) & (points[:, 0] > 0)]
    bottom = near_axis[np.argmin(near_axis[:, 2])]
    middle = near_axis[:, 0].mean()
    return Obstacle(sphere((middle, 0.0, bottom[2] + depth - radius), radius, resolution=96))


def test_meniscus_is_lofted_again_out_of_an_obstacle_keeping_its_caps(meniscus_slices, meniscus_solid):
    boundary = Obstacle(brep_boundary(meniscus_solid))
    # A sphere whose top reaches 0.5 up into the meniscus's flat base, under its body on the +x side.
    obstacle = bump_under(boundary)
    assert overlap(obstacle, boundary) > 0.4

    slices, brep = push_meniscus_out(meniscus_slices, lambda moved: contours_to_brep(moved, cap="planar"), [obstacle])

    assert overlap(obstacle, Obstacle(brep_boundary(brep))) < 0.01
    assert len(slices) == len(meniscus_slices)
    for before, after in zip(meniscus_slices, slices, strict=True):
        # Points move within their slice's plane.
        np.testing.assert_allclose((after.points - before.points) @ before.normal, 0.0, atol=1e-9)
    assert max(np.abs(after.points - before.points).max() for before, after in zip(meniscus_slices, slices, strict=True)) < 1.0


def test_clipped_meniscus_cut_stays_straight(meniscus_clipped_slices, meniscus_clipped_solid):
    boundary = Obstacle(brep_boundary(meniscus_clipped_solid))
    obstacle = bump_under(boundary)

    slices, _ = push_meniscus_out(meniscus_clipped_slices, lambda moved: contours_to_brep(moved, cap="planar"), [obstacle])

    for contour in slices:
        assert contour.straight_from is not None
        cut = contour.points[[*range(contour.straight_from, len(contour.points)), 0]]
        direction = (cut[-1] - cut[0]) / np.linalg.norm(cut[-1] - cut[0])
        offsets = (cut - cut[0]) - np.outer((cut - cut[0]) @ direction, direction)
        assert np.abs(offsets).max() < 1e-9


def facing_sheet(sheet, overlap_depth):
    """The sheet turned over about its s axis, its articular side facing down onto the sheet's, reaching overlap_depth into it at the middle."""
    frame = sheet.frame
    middle = sheet.perimeter.mean(axis=0)
    height = float(sheet.articular.ev(middle[:1], middle[1:])[0])
    top = frame.physical(middle[np.newaxis], np.array([height]))[0]
    s_axis, t_axis, normal = frame.axes
    target = top - overlap_depth * normal
    # The turned frame maps (s, t, h) to origin + s * s_axis - t * t_axis - h * normal.
    origin = target - middle[0] * s_axis + middle[1] * t_axis + height * normal
    return replace(sheet, frame=PlaneFrame(origin, np.array([s_axis, -t_axis, -normal])))


def test_overlapping_cartilages_share_the_overlap_by_thickness(cartilage_sheet, cartilage_hex_solid):
    upper = facing_sheet(cartilage_sheet, 0.3)
    structures = {
        "lower": Adjustable(cartilage_hex_solid, sheet=cartilage_sheet),
        "upper": Adjustable(sheet_to_brep(upper), sheet=upper),
    }
    assert overlap(surface_obstacle(structures["lower"].geometry), surface_obstacle(structures["upper"].geometry)) > 0.25

    resolved = resolve_overclosure(structures, ["lower", "upper"])

    lower, top = (surface_obstacle(resolved[name].geometry) for name in ("lower", "upper"))
    assert overlap(lower, top) < 0.01
    middle = cartilage_sheet.perimeter.mean(axis=0)[np.newaxis]
    lowered = []
    for name, original in (("lower", cartilage_sheet), ("upper", upper)):
        sheet = resolved[name].sheet
        assert sheet is not None
        lowered.append(float((sheet.articular.ev(middle[:, 0], middle[:, 1]) - original.articular.ev(middle[:, 0], middle[:, 1]))[0]))
    # Equal thicknesses share the overlap equally, wherever the two rank.
    assert lowered == pytest.approx([-0.15, -0.15], abs=0.03)


def test_cartilage_must_rank_above_other_structures(cartilage_sheet, cartilage_hex_solid):
    structures = {"cartilage": Adjustable(cartilage_hex_solid, sheet=cartilage_sheet), "bone": Adjustable(sphere((0, 0, 0), 1.0))}

    with pytest.raises(ValueError, match="must rank above"):
        resolve_overclosure(structures, ["bone", "cartilage"])


def test_structures_rank_by_priority_and_unmoved_ones_stay():
    first, second = sphere((0, 0, 0), 5.0), sphere((0, 0, 9.5), 5.0)
    structures = {"first": Adjustable(first), "second": Adjustable(second)}

    resolved = resolve_overclosure(structures, ["second", "first"], clearance=0.2)

    assert resolved["second"].geometry is second
    moved = surface_obstacle(resolved["first"].geometry)
    signed, _ = Obstacle(second).distances(moved.points, reach=1.0)
    assert signed.min() > 0.2 - 0.01
