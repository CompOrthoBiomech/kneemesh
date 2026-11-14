import math

import gmsh
import numpy as np
import pytest

from kneemesh.brep import gmsh_session
from kneemesh.geo.sheet import _outward_normals, fit_sheet, sheet_to_brep
from kneemesh.tet_meshing import import_solid, tetrahedral_mesh

# The cartilage_segmentation fixture's ellipse semi-axes and the centers of its thin dip and defect, in mm.
SEMI_AXES = (15.0, 10.0)
THIN_CENTER = np.array([-6.0, 0.0, 0.0])
DEFECT_CENTER = np.array([6.0, 0.0, 0.0])


def plane_point(sheet, physical):
    return (physical - sheet.frame.origin) @ sheet.frame.axes[:2].T


def thickness(sheet, plane_points):
    return sheet.articular.ev(plane_points[..., 0], plane_points[..., 1]) - sheet.bone.ev(plane_points[..., 0], plane_points[..., 1])


def solid_summary(brep):
    with gmsh_session():
        import_solid(brep)
        volumes = [tag for _, tag in gmsh.model.getEntities(3)]
        faces = {tag: len(gmsh.model.getAdjacencies(2, tag)[0]) for _, tag in gmsh.model.getEntities(2)}
        sides = [[abs(tag) for _, tag in gmsh.model.getBoundary([(3, volume)], oriented=False)] for volume in volumes]
        return sum(gmsh.model.occ.getMass(3, volume) for volume in volumes), sides, faces


def radii_of_curvature(loop):
    incoming, outgoing = loop - np.roll(loop, 1, axis=0), np.roll(loop, -1, axis=0) - loop
    turns = np.arctan2(incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0], np.einsum("ij,ij->i", incoming, outgoing))
    curvature = turns / (0.5 * (np.linalg.norm(incoming, axis=1) + np.linalg.norm(outgoing, axis=1)))
    return 1 / curvature.max(), -1 / curvature.min()


def inside_points(sheet):
    """Domain points on loops shrunk toward the perimeter's mean, which the convex synthetic perimeter contains."""
    center = sheet.perimeter.mean(axis=0)
    return np.concatenate([center + fraction * (sheet.perimeter - center) for fraction in np.linspace(0.0, 0.95, 10)])


def sides_and_walls(brep):
    """The solid's two sides, the larger first, and its other faces, the walls."""
    with gmsh_session():
        import_solid(brep)
        faces = sorted((tag for _, tag in gmsh.model.getEntities(2)), key=lambda face: -gmsh.model.occ.getMass(2, face))
        return faces[:2], faces[2:]


def test_sheet_normal_points_from_the_bone_to_the_articular_side(cartilage_sheet):
    axes = cartilage_sheet.frame.axes

    assert axes @ axes.T == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(axes) == pytest.approx(1.0)
    # The cap's thinnest direction is the bone's radius at its center, +z.
    assert axes[2] @ np.array([0.0, 0.0, 1.0]) > 0.99


def test_sheet_perimeter_lies_inside_the_cartilage_counterclockwise(cartilage_sheet):
    perimeter = cartilage_sheet.perimeter
    physical = cartilage_sheet.frame.physical(perimeter, np.zeros(len(perimeter)))
    ellipse_area = math.pi * SEMI_AXES[0] * SEMI_AXES[1]

    assert np.hypot(physical[:, 0] / SEMI_AXES[0], physical[:, 1] / SEMI_AXES[1]).max() < 1.0
    # Cutting off the tapered rim costs a band around the ellipse.
    assert 0.6 * ellipse_area < cartilage_sheet.area < ellipse_area


def test_sheet_mean_thickness_matches_the_label(cartilage_sheet):
    # Over the ellipse the taper averages two thirds of the 3 mm peak; the dip and the cut-off rim offset each other roughly.
    assert cartilage_sheet.mean_thickness == pytest.approx(2.0, rel=0.15)


@pytest.mark.parametrize("radius", [1.0, 3.0])
def test_perimeter_curves_no_more_tightly_than_its_radius(real_cartilage, radius):
    sheet = fit_sheet(real_cartilage["lateral_tibial_cartilage"], perimeter_radius=radius, min_thickness=0.1)

    convex, concave = radii_of_curvature(sheet.perimeter)

    # The Gaussian after rounding loosens the bound slightly.
    assert min(convex, concave) > 0.85 * radius


def test_thin_regions_are_thickened_on_the_bone_side_only(cartilage_sides):
    thin = plane_point(fit_sheet(cartilage_sides, min_thickness=0.01), THIN_CENTER)
    loose = fit_sheet(cartilage_sides, min_thickness=0.01)
    thick = fit_sheet(cartilage_sides, min_thickness=0.5)

    assert thickness(loose, thin) < 0.5 * loose.mean_thickness
    assert thick.min_thickness == pytest.approx(0.5 * thick.mean_thickness)
    assert thickness(thick, thin) >= thick.min_thickness
    points = inside_points(thick)
    assert thickness(thick, points).min() >= thick.min_thickness
    assert thick.articular.ev(points[:, 0], points[:, 1]) == pytest.approx(loose.articular.ev(points[:, 0], points[:, 1]))


def test_defects_become_holes_only_when_asked(cartilage_defect_sides):
    without = fit_sheet(cartilage_defect_sides, perimeter_radius=1.0, min_thickness=0.1)
    with_holes = fit_sheet(cartilage_defect_sides, perimeter_radius=1.0, min_thickness=0.1, hole_thickness=0.05)

    assert without.holes == []
    assert len(with_holes.holes) == 1
    hole = with_holes.holes[0]
    assert hole.mean(axis=0) == pytest.approx(plane_point(with_holes, DEFECT_CENTER), abs=0.5)
    area = 0.5 * abs(np.sum(hole[:, 0] * np.roll(hole[:, 1], -1) - np.roll(hole[:, 0], -1) * hole[:, 1]))
    # The defect has a 2.5 mm radius; rounding and smoothing change it a little.
    assert math.sqrt(area / math.pi) == pytest.approx(2.5, rel=0.15)


def test_thin_cartilage_becomes_a_hole_below_hole_thickness(cartilage_sides):
    # The fitted dip is 15% of the mean thickness at its center and 30% about 1.3 mm out, wide enough for a 1 mm radius.
    sheet = fit_sheet(cartilage_sides, perimeter_radius=1.0, min_thickness=0.4, hole_thickness=0.3)

    assert len(sheet.holes) == 1
    assert sheet.holes[0].mean(axis=0) == pytest.approx(plane_point(sheet, THIN_CENTER), abs=0.5)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"perimeter_radius": 0.0}, "perimeter_radius"),
        ({"perimeter_inset": -1.0}, "perimeter_inset"),
        ({"min_thickness": 0.0}, "min_thickness"),
        ({"min_thickness": 1.0}, "min_thickness"),
        ({"min_thickness": 0.1, "hole_thickness": 0.1}, "hole_thickness"),
        ({"perimeter_inset": 50.0}, "do not overlap enough"),
    ],
)
def test_fit_sheet_rejects_invalid_arguments(cartilage_sides, arguments, message):
    with pytest.raises(ValueError, match=message):
        fit_sheet(cartilage_sides, **arguments)


def test_perimeter_inset_shrinks_the_sheet(cartilage_sides, cartilage_sheet):
    inset = fit_sheet(cartilage_sides, perimeter_inset=1.0)

    assert inset.area < 0.95 * cartilage_sheet.area


def test_solid_is_two_sides_and_a_wall(cartilage_sheet, cartilage_hex_solid):
    _, sides, _ = solid_summary(cartilage_hex_solid)
    # occ.getMass is unreliable with ruled walls, so the volume comes from tetrahedra.
    mesh = tetrahedral_mesh(cartilage_hex_solid)
    corners = mesh.nodes[mesh.elements["Tetrahedron 4"]]
    edges = corners[:, 1:] - corners[:, :1]
    volume = abs(np.linalg.det(edges)).sum() / 6
    # Each shrunk loop stands for a ring of area proportional to its fraction.
    weights = np.repeat(np.linspace(0.0, 0.95, 10), len(cartilage_sheet.perimeter))

    assert [len(faces) for faces in sides] == [3]
    assert volume == pytest.approx(
        cartilage_sheet.area * np.average(thickness(cartilage_sheet, inside_points(cartilage_sheet)), weights=weights), rel=0.05
    )


def test_solid_sides_are_parameterized_by_the_domain(cartilage_sheet, cartilage_hex_solid):
    points = inside_points(cartilage_sheet)
    (top, bottom), _ = sides_and_walls(cartilage_hex_solid)

    with gmsh_session():
        import_solid(cartilage_hex_solid)
        values = {face: np.reshape(gmsh.model.getValue(2, face, points.ravel().tolist()), (-1, 3)) for face in (top, bottom)}
        bounds = [np.concatenate(gmsh.model.getParametrizationBounds(2, face)) for face in (top, bottom)]

    assert bounds[0] == pytest.approx(bounds[1])
    # The surfaces interpolate the sides' points on a 0.5 mm grid.
    expected = {side: cartilage_sheet.points(points, side) for side in ("articular", "bone")}
    deviations = sorted(np.linalg.norm(values[face] - expected[side], axis=1).max() for face in (top, bottom) for side in expected)
    assert deviations[1] < 0.02


def test_columns_between_the_sides_lie_on_the_wall(cartilage_sheet, cartilage_hex_solid):
    (top, bottom), walls = sides_and_walls(cartilage_hex_solid)

    with gmsh_session():
        import_solid(cartilage_hex_solid)
        ends = [np.reshape(gmsh.model.getValue(2, face, cartilage_sheet.perimeter.ravel().tolist()), (-1, 3)) for face in (top, bottom)]
        quarters = [ends[0] + fraction * (ends[1] - ends[0]) for fraction in (0.25, 0.5, 0.75)]
        distances = [
            np.linalg.norm(np.reshape(gmsh.model.getClosestPoint(2, walls[0], points.ravel().tolist())[0], (-1, 3)) - points, axis=1)
            for points in quarters
        ]

    assert np.max(distances) < 1e-3


def test_solid_has_a_hole_through_it(cartilage_defect_sides):
    sheet = fit_sheet(cartilage_defect_sides, perimeter_radius=1.0, min_thickness=0.1, hole_thickness=0.05)
    filled = fit_sheet(cartilage_defect_sides, perimeter_radius=1.0, min_thickness=0.1)

    holed_volume, sides, faces = solid_summary(sheet_to_brep(sheet))
    filled_volume, _, filled_faces = solid_summary(sheet_to_brep(filled))

    assert len(sides) == 1
    assert len(faces) > len(filled_faces)
    # Filling dips into the funnel the smoothed surface makes at the defect, so it adds less than a full thickness.
    assert 0 < filled_volume - holed_volume < math.pi * 2.5**2 * filled.mean_thickness


def test_holes_narrower_than_the_perimeter_radius_vanish(cartilage_defect_sides):
    # Opening by a disk removes any hole it cannot roll around inside.
    assert fit_sheet(cartilage_defect_sides, perimeter_radius=3.0, min_thickness=0.2, hole_thickness=0.05).holes == []


@pytest.mark.parametrize("structure", ["lateral_tibial_cartilage", "medial_tibial_cartilage", "patellar_cartilage"])
def test_real_cartilage_hex_solids_are_two_sides_and_a_wall(real_cartilage_hex_solids, structure):
    _, sides, _ = solid_summary(real_cartilage_hex_solids[structure])

    assert [len(faces) for faces in sides] == [3]


def test_real_lateral_tibial_tetrahedral_solid_builds_around_its_lobe(real_cartilage):
    # The 1 mm perimeter follows a lobe whose tip's inward normals cross a ring row, which folded the blocks.
    sheet = fit_sheet(real_cartilage["lateral_tibial_cartilage"], perimeter_radius=1.0, min_thickness=0.1, hole_thickness=0.05)

    volume, sides, _ = solid_summary(sheet_to_brep(sheet))

    assert len(sides) == 1
    assert volume == pytest.approx(sheet.area * sheet.mean_thickness, rel=0.15)


@pytest.mark.parametrize("structure", ["lateral_tibial_cartilage", "medial_tibial_cartilage", "patellar_cartilage"])
def test_walls_tilt_within_their_limits(real_cartilage, structure):
    sheet = fit_sheet(real_cartilage[structure])
    perimeter = sheet.perimeter

    rim = sheet.articular.ev(perimeter[:, 0], perimeter[:, 1]) - sheet.bone.ev(perimeter[:, 0], perimeter[:, 1])
    assert np.all(np.abs(sheet.tilt) <= 0.25 * rim + 1e-12)
    assert np.all(np.abs(sheet.tilt) <= 0.25 * 3.0)
    # The articular sides curve down at the rim, so their walls mostly lean outward at the top.
    assert np.median(sheet.tilt) > 0


def test_tilt_moves_the_sides_apart_along_the_normal_at_the_perimeter_only(real_cartilage):
    sheet = fit_sheet(real_cartilage["patellar_cartilage"])
    perimeter = sheet.perimeter

    apart = sheet.tilted(perimeter, "articular") - sheet.tilted(perimeter, "bone")
    assert apart == pytest.approx(sheet.tilt[:, np.newaxis] * _outward_normals(perimeter), abs=1e-9)
    center = np.mean(perimeter, axis=0, keepdims=True)
    assert np.array_equal(sheet.tilted(center, "articular"), center)


def test_tilt_continues_smoothly_outside_the_perimeter(real_cartilage):
    sheet = fit_sheet(real_cartilage["patellar_cartilage"])
    normals = _outward_normals(sheet.perimeter)
    step = 1e-4 * sheet.tilt_reach

    moves = [sheet.tilted(sheet.perimeter + offset * normals, "articular") - (sheet.perimeter + offset * normals) for offset in (-step, 0.0, step)]

    # The move grows outward through the perimeter without a kink: its two one-sided slopes agree, up to the
    # polyline's piecewise normals. Fading by unsigned distance gave them opposite signs.
    inward, outward = (moves[1] - moves[0]) / step, (moves[2] - moves[1]) / step
    assert np.linalg.norm(outward - inward) < 0.1 * np.linalg.norm(inward)


def test_plane_frame_round_trips(cartilage_sheet):
    frame = cartilage_sheet.frame
    points = np.random.default_rng(0).uniform(-10, 10, (20, 3))

    assert frame.physical(*frame.local(points)) == pytest.approx(points, abs=1e-12)
