import math

import gmsh
import numpy as np
import pytest
import SimpleITK as sitk

from kneemesh.brep import gmsh_session
from kneemesh.cartilage import CartilageSides
from kneemesh.geo.femur import femoral_blocks, femoral_sheet_to_brep
from kneemesh.geo.sheet import CylinderFrame, fit_sheet
from kneemesh.tet_meshing import import_solid

# The femoral_segmentation fixture's femur radius, cartilage thickness, angular spans, and half-widths of the notch and the cartilage, in mm and degrees.
FEMUR_RADIUS = 20.0
FEMORAL_THICKNESS = 2.0
TROCHLEA_DEGREES = 68.0
CONDYLE_DEGREES = 136.0
NOTCH_HALF_WIDTH = 6.0
FEMORAL_HALF_WIDTH = 18.0


def solid_summary(brep):
    with gmsh_session():
        import_solid(brep)
        volumes = [tag for _, tag in gmsh.model.getEntities(3)]
        sides = [len(gmsh.model.getBoundary([(3, volume)], oriented=False)) for volume in volumes]
        return sum(gmsh.model.occ.getMass(3, volume) for volume in volumes), sides


def cell_areas(grid):
    along, across = grid[:-1, 1:] - grid[:-1, :-1], grid[1:, :-1] - grid[:-1, :-1]
    return along[..., 0] * across[..., 1] - along[..., 1] * across[..., 0]


def test_cylinder_frame_round_trips_and_is_right_handed(femoral_sheet):
    frame = femoral_sheet.frame
    rng = np.random.default_rng(0)
    points = rng.uniform(-20, 20, (50, 3))

    assert frame.physical(*frame.local(points)) == pytest.approx(points, abs=1e-9)
    # Turning from s toward t is counterclockwise seen from outside, where heights grow.
    domain, heights = frame.local(points[:1])
    step = 1e-4
    ds = frame.physical(domain + [step, 0], heights) - frame.physical(domain, heights)
    dt = frame.physical(domain + [0, step], heights) - frame.physical(domain, heights)
    dh = frame.physical(domain, heights + step) - frame.physical(domain, heights)
    assert np.cross(ds[0], dt[0]) @ dh[0] > 0


def test_cylinder_frame_fits_the_femur(femoral_sheet):
    frame = femoral_sheet.frame

    assert abs(frame.axes[0] @ [1.0, 0.0, 0.0]) > 0.999
    assert np.hypot(frame.center[1], frame.center[2]) < 0.5
    # The reference radius is the sides' mean distance from the axis, midway through the shell.
    assert frame.radius == pytest.approx(FEMUR_RADIUS + FEMORAL_THICKNESS / 2, abs=0.3)


def test_cylinder_frame_rejects_a_closed_ring():
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    bone = np.vstack([np.column_stack((np.full_like(angles, x), 20 * np.cos(angles), 20 * np.sin(angles))) for x in range(-5, 6)])

    with pytest.raises(ValueError, match="too far to unroll"):
        CylinderFrame.fit(CartilageSides(bone * 1.1, bone))


def test_femoral_sheet_unrolls_the_shell(femoral_sheet):
    arc = math.radians(1) * (FEMUR_RADIUS + FEMORAL_THICKNESS / 2)
    shell = 2 * FEMORAL_HALF_WIDTH * TROCHLEA_DEGREES * arc + 2 * (FEMORAL_HALF_WIDTH - NOTCH_HALF_WIDTH) * CONDYLE_DEGREES * arc

    assert femoral_sheet.mean_thickness == pytest.approx(FEMORAL_THICKNESS, rel=0.1)
    # Cutting off the steep rim costs a band around the U.
    assert 0.75 * shell < femoral_sheet.area < shell


def test_femoral_blocks_are_a_core_of_five_inside_a_ring_of_twelve(femoral_sheet):
    blocks = femoral_blocks(femoral_sheet)

    assert len(blocks) == 17
    assert all(cell_areas(grid).min() > 0 for grid in blocks)
    # The ring's first rows run around the perimeter, so its blocks' areas add up to the sheet's with the core's.
    total = sum(float(cell_areas(grid).sum()) for grid in blocks)
    assert total == pytest.approx(femoral_sheet.area, rel=0.02)


def test_real_femoral_blocks_do_not_fold(real_femoral_sheet):
    blocks = femoral_blocks(real_femoral_sheet)

    assert all(cell_areas(grid).min() > 0 for grid in blocks)


def test_femoral_hex_blocks_are_seventeen_six_sided_blocks(femoral_hex_blocks):
    _, sides = solid_summary(femoral_hex_blocks)

    assert sides == [6] * 17


def test_real_femoral_hex_blocks_are_seventeen_six_sided_blocks(real_femoral_hex_blocks, real_femoral_sheet):
    volume, sides = solid_summary(real_femoral_hex_blocks)

    assert sides == [6] * 17
    assert volume == pytest.approx(real_femoral_sheet.area * real_femoral_sheet.mean_thickness, rel=0.1)


def test_femoral_solid_and_blocks_have_the_same_volume(femoral_sheet, femoral_hex_blocks):
    blocks_volume, _ = solid_summary(femoral_hex_blocks)
    solid_volume, sides = solid_summary(femoral_sheet_to_brep(femoral_sheet))

    assert len(sides) == 1
    assert solid_volume == pytest.approx(blocks_volume, rel=1e-3)


def test_femoral_tetrahedral_solid_keeps_most_of_the_shell(femoral_solid, femoral_segmentation):
    volume, sides = solid_summary(femoral_solid)
    voxels = np.count_nonzero(sitk.GetArrayViewFromImage(femoral_segmentation.image) == 2) * 0.5**3

    assert len(sides) == 1
    assert 0.85 * voxels < volume < voxels


def test_femoral_blocks_need_a_u_shaped_domain(cartilage_sheet):
    with pytest.raises(ValueError, match="U-shaped"):
        femoral_blocks(cartilage_sheet)


@pytest.mark.parametrize("ring_width", [0.0, 1.0])
def test_femoral_blocks_reject_ring_widths_out_of_range(femoral_sheet, ring_width):
    with pytest.raises(ValueError, match="ring_width"):
        femoral_blocks(femoral_sheet, ring_width)


def test_a_plane_cannot_hold_the_wrapped_real_femoral_cartilage(real_femoral_sides, real_femoral_sheet):
    # Projected on a plane, the condyles fold back over the trochlea, so the sides' overlap misses much of the cartilage.
    plane = fit_sheet(real_femoral_sides)

    assert real_femoral_sheet.area > 1.5 * plane.area
