import math

import gmsh
import numpy as np
import pytest

from kneemesh.brep import gmsh_session
from kneemesh.geo.sheet import _butterfly_blocks, _signed_turning_angles, coons_block, fit_sheet, sheet_to_brep, sheet_to_hex_blocks
from kneemesh.tet_meshing import import_solid

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
    lengths = np.linalg.norm(np.roll(loop, -1, axis=0) - loop, axis=1)
    curvature = _signed_turning_angles(loop) / (0.5 * (lengths + np.roll(lengths, 1)))
    return 1 / curvature.max(), -1 / curvature.min()


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
    blocks = np.concatenate([grid.reshape(-1, 2) for grid in _butterfly_blocks(thick)])
    assert thickness(thick, blocks).min() >= thick.min_thickness
    assert thick.articular.ev(blocks[:, 0], blocks[:, 1]) == pytest.approx(loose.articular.ev(blocks[:, 0], blocks[:, 1]))


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


def test_hex_blocks_are_five_six_sided_blocks_around_an_inner_one(cartilage_hex_blocks):
    _, sides, faces = solid_summary(cartilage_hex_blocks)

    assert [len(block) for block in sides] == [6] * 5
    shared = sorted(sum(faces[face] == 2 for face in block) for block in sides)
    # The inner block shares all four walls; each outer block shares two spoke walls and its inner wall.
    assert shared == [3, 3, 3, 3, 4]


def test_hex_blocks_and_solid_have_the_same_volume(cartilage_sheet, cartilage_hex_blocks):
    blocks_volume, _, _ = solid_summary(cartilage_hex_blocks)
    solid_volume, sides, _ = solid_summary(sheet_to_brep(cartilage_sheet))

    assert len(sides) == 1
    assert solid_volume == pytest.approx(blocks_volume, rel=1e-3)
    # The volume over the grid of thicknesses inside the perimeter.
    blocks = _butterfly_blocks(cartilage_sheet)
    assert blocks_volume == pytest.approx(cartilage_sheet.area * thickness(cartilage_sheet, np.concatenate(blocks)).mean(), rel=0.1)


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


def test_hex_blocks_ignore_holes(cartilage_defect_sides):
    sheet = fit_sheet(cartilage_defect_sides, perimeter_radius=1.0, min_thickness=0.2, hole_thickness=0.05)

    _, sides, _ = solid_summary(sheet_to_hex_blocks(sheet))

    assert len(sheet.holes) == 1
    assert len(sides) == 5


@pytest.mark.parametrize("structure", ["lateral_tibial_cartilage", "medial_tibial_cartilage", "patellar_cartilage"])
def test_real_cartilage_hex_blocks_do_not_fold(real_cartilage_hex_blocks, structure):
    _, sides, _ = solid_summary(real_cartilage_hex_blocks[structure])

    # Fragmenting would split overlapping or folded blocks into more than five volumes.
    assert [len(block) for block in sides] == [6] * 5


def test_butterfly_blocks_bend_around_the_real_lateral_notch(real_cartilage):
    blocks = _butterfly_blocks(fit_sheet(real_cartilage["lateral_tibial_cartilage"]))

    for grid in blocks:
        along, across = grid[:-1, 1:] - grid[:-1, :-1], grid[1:, :-1] - grid[:-1, :-1]
        assert (along[..., 0] * across[..., 1] - along[..., 1] * across[..., 0]).min() > 0


def test_butterfly_columns_leave_the_perimeter_at_right_angles(real_cartilage):
    blocks = _butterfly_blocks(fit_sheet(real_cartilage["medial_tibial_cartilage"]))

    for grid in blocks[1:]:
        # Row 0 lies on the perimeter; interior points of the row have central tangents.
        tangents = grid[0, 2:] - grid[0, :-2]
        columns = grid[1, 1:-1] - grid[0, 1:-1]
        cosines = np.einsum("ij,ij->i", tangents, columns) / (np.linalg.norm(tangents, axis=1) * np.linalg.norm(columns, axis=1))
        # The B-spline arc's samples and the polygon's normals differ slightly.
        assert np.abs(cosines).max() < 0.05


def test_plane_frame_round_trips(cartilage_sheet):
    frame = cartilage_sheet.frame
    points = np.random.default_rng(0).uniform(-10, 10, (20, 3))

    assert frame.physical(*frame.local(points)) == pytest.approx(points, abs=1e-12)


def test_coons_block_keeps_its_sides_and_fills_a_square_evenly():
    samples = np.linspace(0.0, 1.0, 5)[:, np.newaxis]
    bottom, top = samples * [1.0, 0.0], samples * [1.0, 0.0] + [0.0, 1.0]
    left, right = samples * [0.0, 1.0], samples * [0.0, 1.0] + [1.0, 0.0]
    curved = bottom + np.column_stack((np.zeros(5), 0.1 * np.sin(np.pi * samples[:, 0])))

    square = coons_block([bottom, top, left, right])
    bent = coons_block([curved, top, left, right])

    assert square[2, 3] == pytest.approx([0.75, 0.5])
    assert bent[0] == pytest.approx(curved)
    assert bent[:, -1] == pytest.approx(right)
