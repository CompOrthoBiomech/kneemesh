import re
from dataclasses import replace
from itertools import pairwise

import gmsh
import numpy as np
import pytest
import SimpleITK as sitk
from scipy.interpolate import BSpline
from scipy.spatial import KDTree

from kneemesh.brep import gmsh_session
from kneemesh.disk import write_brep
from kneemesh.geo.meniscus import centerline, contours_to_brep, contours_to_hex_blocks, slice_isocontour
from kneemesh.segmentation import assemble_label_maps
from kneemesh.vol import DEFAULT_SMOOTHING_ITERATIONS, ContourSlice, extract_isocontour, signed_distance_field, smooth_isocontour

# Area centroid of the voxelized fixture's cross-section. It differs slightly
# from the ideal triangle (16.67, -1.33) because the fixture's bottom slice lies
# exactly on the triangle's base and counts as a whole voxel layer.
SECTION_CENTER_RADIUS = 16.43
SECTION_CENTER_Z = -1.67


def sample(spline, count=1001):
    return spline(np.linspace(0.0, 1.0, count))


def polar(points):
    angles = np.degrees(np.unwrap(np.arctan2(points[:, 1], points[:, 0])))
    return np.hypot(points[:, 0], points[:, 1]), angles


def test_centerline_is_cubic_spline_of_physical_points(meniscus_centerline):
    assert isinstance(meniscus_centerline, BSpline)
    assert meniscus_centerline.k == 3
    assert sample(meniscus_centerline, 11).shape == (11, 3)


def test_centerline_passes_through_cross_section_centers(meniscus_centerline):
    points = sample(meniscus_centerline)
    radius, _ = polar(points)

    assert radius == pytest.approx(np.full_like(radius, SECTION_CENTER_RADIUS), abs=0.5)
    assert points[:, 2] == pytest.approx(np.full(len(points), SECTION_CENTER_Z), abs=0.25)


def test_centerline_spans_the_full_sweep(meniscus_centerline):
    _, angles = polar(sample(meniscus_centerline))

    assert angles.min() == pytest.approx(-135.0, abs=5.0)
    assert angles.max() == pytest.approx(135.0, abs=5.0)


def test_centerline_is_smooth(meniscus_centerline):
    u = np.linspace(0.0, 1.0, 1001)

    tangents = meniscus_centerline(u, nu=1)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    turn_degrees = np.degrees(np.arccos(np.clip((tangents[1:] * tangents[:-1]).sum(axis=1), -1, 1)))

    assert turn_degrees.max() < 1.0


def test_centerline_is_independent_of_image_axis_order(meniscus_label, meniscus_centerline):
    permuted = sitk.PermuteAxes(meniscus_label, (2, 0, 1))

    expected = sample(meniscus_centerline)
    result = sample(centerline(permuted))

    distances, _ = KDTree(expected).query(result)
    assert distances.max() < 0.1


def test_centerline_selects_requested_label(meniscus_label, meniscus_centerline):
    expected = sample(meniscus_centerline)
    relabeled = meniscus_label * 7
    relabeled[0, 0, 0] = 1

    result = sample(centerline(relabeled, label=7))

    distances, _ = KDTree(expected).query(result)
    assert distances.max() < 0.1


def test_centerline_rejects_absent_label(meniscus_label):
    with pytest.raises(ValueError, match="not present"):
        centerline(meniscus_label, label=2)


@pytest.mark.parametrize(("pixel_id", "label"), [(sitk.sitkUInt16, 1000), (sitk.sitkInt16, -300)])
def test_centerline_accepts_16_bit_label_maps(meniscus_label, meniscus_centerline, pixel_id, label):
    # Labels outside the 8-bit range show the full pixel type is used.
    relabeled = sitk.Cast(meniscus_label, pixel_id) * label

    distances, _ = KDTree(sample(meniscus_centerline)).query(sample(centerline(relabeled, label=label)))

    assert distances.max() < 0.1


@pytest.mark.parametrize("pixel_id", [sitk.sitkInt8, sitk.sitkUInt32, sitk.sitkInt32, sitk.sitkFloat32])
def test_centerline_rejects_other_pixel_types(meniscus_label, pixel_id):
    with pytest.raises(ValueError, match="3D sitkUInt8, sitkUInt16, or sitkInt16"):
        centerline(sitk.Cast(meniscus_label, pixel_id))


def test_centerline_rejects_2d_image():
    with pytest.raises(ValueError, match="3D sitkUInt8"):
        centerline(sitk.Image(8, 8, sitk.sitkUInt8))


def test_centerline_rejects_tiny_label():
    image = sitk.Image(8, 8, 8, sitk.sitkUInt8)
    image[1, 1, 1] = 1

    with pytest.raises(ValueError, match="too small"):
        centerline(image)


def straight_line(start, end):
    return BSpline([0.0, 0.0, 1.0, 1.0], np.array([start, end], dtype=float), 1)


def in_plane(contour_slice):
    offsets = contour_slice.points - contour_slice.points.mean(axis=0)
    return offsets @ contour_slice.u_axis, offsets @ contour_slice.v_axis


def test_slices_default_to_32_points(meniscus_slices):
    assert all(len(contour_slice.points) == 32 for contour_slice in meniscus_slices)


@pytest.mark.parametrize("count", [16, 64])
def test_slice_points_are_evenly_spaced_around_a_closed_contour(meniscus_mesh, meniscus_centerline, count):
    slices = slice_isocontour(meniscus_mesh, meniscus_centerline, spacing=20.0, points_per_slice=count)

    for contour_slice in slices:
        assert contour_slice.points.shape == (count, 3)
        gaps = np.linalg.norm(contour_slice.points - np.roll(contour_slice.points, -1, axis=0), axis=1)
        assert gaps.max() < 1.5 * np.median(gaps)


def test_slice_points_lie_in_their_plane(meniscus_slices):
    for contour_slice in meniscus_slices:
        assert np.abs((contour_slice.points - contour_slice.center) @ contour_slice.normal).max() < 1e-4


def test_slice_points_wind_counterclockwise_from_positive_u(meniscus_slices):
    for contour_slice in meniscus_slices:
        u, v = in_plane(contour_slice)
        assert np.sum(u * np.roll(v, -1) - np.roll(u, -1) * v) > 0
        assert abs(np.degrees(np.arctan2(v[0], u[0]))) < 30.0


def test_slice_points_line_up_with_the_previous_slice(meniscus_slices):
    for previous, contour_slice in pairwise(meniscus_slices):
        target = np.column_stack(in_plane(previous))
        current = np.column_stack(in_plane(contour_slice))
        costs = [np.sum((target - np.roll(current, -shift, axis=0)) ** 2) for shift in range(len(current))]
        # The centroids subtracted here differ slightly from the slice centers
        # used for alignment, so allow the optimum to land on a neighbor.
        assert min(np.argmin(costs), len(costs) - np.argmin(costs)) <= 1


def test_slice_contours_lie_on_the_label_boundary(meniscus_slices, meniscus_distance_field):
    voxel = max(meniscus_distance_field.GetSpacing())
    for contour_slice in meniscus_slices:
        for point in contour_slice.points:
            index = meniscus_distance_field.TransformPhysicalPointToIndex(point.tolist())
            assert abs(meniscus_distance_field[index]) <= voxel


def test_slices_take_the_contour_around_the_curve(meniscus_slices):
    for contour_slice in meniscus_slices:
        assert np.linalg.norm(contour_slice.points - contour_slice.center, axis=1).max() < 9.0


def test_slice_areas_match_cross_section(meniscus_slices):
    for contour_slice in meniscus_slices:
        # About 45 mm^2 when voxelized, less the rim corners that smoothing rounds.
        assert contour_slice.area == pytest.approx(42.0, abs=2.0)
        u, v = in_plane(contour_slice)
        assert 0.5 * np.sum(u * np.roll(v, -1) - np.roll(u, -1) * v) == pytest.approx(contour_slice.area, rel=0.05)


@pytest.mark.parametrize("spacing", [3.0, 7.0])
def test_slices_are_uniformly_spaced_on_the_curve(meniscus_mesh, meniscus_centerline, spacing):
    slices = slice_isocontour(meniscus_mesh, meniscus_centerline, spacing)
    centers = np.array([contour_slice.center for contour_slice in slices])

    distances, _ = KDTree(sample(meniscus_centerline, 5001)).query(centers)
    assert distances.max() < 0.05
    # Chords of a circle of radius 16.4 are under 1% shorter than these arcs.
    assert np.linalg.norm(np.diff(centers, axis=0), axis=1) == pytest.approx(np.full(len(centers) - 1, spacing), rel=0.02)


def test_slice_frames_are_right_handed_and_normal_to_curve(meniscus_slices, meniscus_centerline):
    u = np.linspace(0.0, 1.0, 5001)
    points = meniscus_centerline(u)
    tangents = meniscus_centerline(u, nu=1)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    tree = KDTree(points)

    for contour_slice in meniscus_slices:
        frame = np.column_stack((contour_slice.u_axis, contour_slice.v_axis, contour_slice.normal))
        assert frame.T @ frame == pytest.approx(np.eye(3), abs=1e-9)
        assert np.linalg.det(frame) == pytest.approx(1.0)
        _, nearest = tree.query(contour_slice.center)
        assert contour_slice.normal @ tangents[nearest] == pytest.approx(1.0, abs=1e-3)


def test_slices_do_not_twist_along_the_curve(meniscus_slices):
    v_axes = np.array([contour_slice.v_axis for contour_slice in meniscus_slices])

    assert (v_axes[1:] * v_axes[:-1]).sum(axis=1).min() > 0.99


def make_label(mask):
    return sitk.GetImageFromArray(mask.astype(np.uint8))


def test_slices_clip_ends_below_a_fifth_of_average_area():
    # A cone along z with base radius 15 at z = 5 and its apex at z = 75. Mean
    # area is a third of the base area, so slices stop where r / 15 = sqrt(0.2 / 3).
    z, y, x = np.indices((81, 41, 41))
    radius = 15.0 * (75 - z) / 70
    mask = (z >= 5) & (z <= 75) & (np.hypot(x - 20, y - 20) <= radius)
    mesh = extract_isocontour(signed_distance_field(make_label(mask)))

    slices = slice_isocontour(mesh, straight_line((20, 20, 5.5), (20, 20, 74.5)), spacing=2.0)

    heights = [contour_slice.center[2] for contour_slice in slices]
    expected_tip = 75 - 70 * np.sqrt(0.2 / 3)
    assert min(heights) < 8.0
    assert max(heights) == pytest.approx(expected_tip, abs=4.0)


def test_slices_keep_small_sections_between_the_ends():
    z, y, x = np.indices((61, 41, 41))
    radius = np.where((z >= 25) & (z <= 35), 3.0, 15.0)
    mask = (z >= 5) & (z <= 55) & (np.hypot(x - 20, y - 20) <= radius)
    mesh = extract_isocontour(signed_distance_field(make_label(mask)))

    slices = slice_isocontour(mesh, straight_line((20, 20, 6), (20, 20, 54)), spacing=2.0)

    areas = np.array([contour_slice.area for contour_slice in slices])
    assert len(slices) == 25
    assert areas.min() < 0.2 * areas.mean()


def test_slice_rejects_gap_in_the_mesh_along_the_curve():
    z, y, x = np.indices((71, 41, 41))
    in_blocks = ((z >= 5) & (z <= 25)) | ((z >= 45) & (z <= 65))
    mask = in_blocks & (np.hypot(x - 20, y - 20) <= 10)
    mesh = extract_isocontour(signed_distance_field(make_label(mask)))

    with pytest.raises(ValueError, match="does not cut the mesh"):
        slice_isocontour(mesh, straight_line((20, 20, 6), (20, 20, 64)), spacing=2.0)


def test_slice_rejects_curve_that_misses_the_mesh(meniscus_mesh):
    with pytest.raises(ValueError, match="do not cut the mesh"):
        slice_isocontour(meniscus_mesh, straight_line((100, 100, 100), (100, 100, 120)), spacing=2.0)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"spacing": 0.0}, "must be positive"),
        ({"spacing": -1.0}, "must be positive"),
        ({"spacing": 5.0, "points_per_slice": 2}, "at least 3"),
    ],
)
def test_slice_rejects_invalid_arguments(meniscus_mesh, meniscus_centerline, arguments, message):
    with pytest.raises(ValueError, match=message):
        slice_isocontour(meniscus_mesh, meniscus_centerline, **arguments)


def test_centerline_follows_a_label_cut_off_by_the_image_edge(meniscus_label, meniscus_centerline):
    # Cropping at x = 17.5 mm cuts through the middle of the arc.
    cropped = meniscus_label[:86, :, :]

    lengths = np.linalg.norm(np.diff(sample(centerline(cropped)), axis=0), axis=1).sum()

    full_length = np.linalg.norm(np.diff(sample(meniscus_centerline), axis=0), axis=1).sum()
    assert lengths == pytest.approx(full_length, rel=0.1)


def test_centerline_of_real_label_follows_the_whole_arc(real_label):
    points = sample(centerline(real_label))
    label_points = np.argwhere(sitk.GetArrayViewFromImage(real_label) == 1)[:, ::-1]
    physical = np.array([real_label.TransformIndexToPhysicalPoint(index.tolist()) for index in label_points])

    distances, _ = KDTree(physical).query(points)
    assert np.linalg.norm(np.diff(points, axis=0), axis=1).sum() > 70.0
    assert distances.max() < 1.0


def make_slice(points):
    axes = np.eye(3)
    return ContourSlice(np.array(points, dtype=float), np.zeros(3), axes[0], axes[1], axes[2], 1.0)


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


def test_planar_caps_make_a_solid_with_flat_ends(meniscus_solid, meniscus_slices, tmp_path):
    with gmsh_session():
        import_brep(meniscus_solid, tmp_path)

        assert entity_counts()[2:] == [3, 1]
        assert sorted(gmsh.model.getType(2, tag) for _, tag in gmsh.model.getEntities(2)) == ["BSpline surface", "Plane", "Plane"]
        mean_area = np.mean([contour.area for contour in meniscus_slices])
        length = 5.0 * (len(meniscus_slices) - 1)
        assert gmsh.model.occ.getMass(3, gmsh.model.getEntities(3)[0][1]) == pytest.approx(mean_area * length, rel=0.03)


def test_smooth_caps_make_a_solid_larger_than_planar_caps(meniscus_domed_solid, meniscus_solid, tmp_path):
    def volume(brep):
        with gmsh_session():
            import_brep(brep, tmp_path)
            assert entity_counts()[2:] == [3, 1]
            return gmsh.model.occ.getMass(3, gmsh.model.getEntities(3)[0][1])

    assert volume(meniscus_domed_solid) > volume(meniscus_solid)


def cap_apexes(contours, offset=None):
    apexes = []
    for contour, sign in ((contours[0], -1.0), (contours[-1], 1.0)):
        height = 0.25 * np.sqrt(contour.area / np.pi) if offset is None else offset
        apexes.append((contour, sign * contour.normal, height, contour.center + sign * height * contour.normal))
    return apexes


@pytest.mark.parametrize("offset", [None, 2.0])
def test_smooth_caps_converge_to_a_point_offset_from_the_center(meniscus_slices, tmp_path, offset):
    brep = contours_to_brep(meniscus_slices, cap="smooth", cap_offset=offset)

    with gmsh_session():
        import_brep(brep, tmp_path)
        vertices = np.array([gmsh.model.getValue(0, tag, []) for _, tag in gmsh.model.getEntities(0)])

    for _, _, _, apex in cap_apexes(meniscus_slices, offset):
        assert np.linalg.norm(vertices - apex, axis=1).min() < 1e-6


def test_smooth_caps_are_convex_domes(meniscus_domed_solid, meniscus_slices, tmp_path):
    with gmsh_session():
        import_brep(meniscus_domed_solid, tmp_path)
        faces = [tag for _, tag in gmsh.model.getEntities(2)]
        lateral = max(faces, key=lambda tag: gmsh.model.occ.getMass(2, tag))
        caps = [tag for tag in faces if tag != lateral]
        meridians = {}
        for face in caps:
            (u_low, v_low), (u_high, v_high) = gmsh.model.getParametrizationBounds(2, face)
            v = np.linspace(v_low, v_high, 41)
            parameters = np.array([[u, value] for u in np.linspace(u_low, u_high, 16, endpoint=False) for value in v])
            meridians[face] = np.array(gmsh.model.getValue(2, face, parameters.ravel().tolist())).reshape(16, 41, 3)

    for contour, outward, height, apex in cap_apexes(meniscus_slices):
        cap = min(meridians.values(), key=lambda samples: np.linalg.norm(samples - apex, axis=2).min())
        heights = (cap - contour.center) @ outward
        radii = np.linalg.norm(cap - contour.center - heights[..., np.newaxis] * outward, axis=2)
        assert heights.max() == pytest.approx(height, abs=1e-6)
        for meridian_heights, meridian_radii in zip(heights, radii):
            # Run from the contour to the apex, whichever way the face is parametrized.
            if meridian_radii[0] < meridian_radii[-1]:
                meridian_heights, meridian_radii = meridian_heights[::-1], meridian_radii[::-1]
            assert np.all(np.diff(meridian_heights) >= -1e-9)
            assert np.all(np.diff(meridian_radii) <= 1e-9)
            # Rising from the rim and curving in to the apex is a steady left turn in (radius, height).
            dr, dh = np.diff(meridian_radii), np.diff(meridian_heights)
            assert np.all(dr[:-1] * dh[1:] - dh[:-1] * dr[1:] >= -1e-9)


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
    assert spline_surface_degrees(contours_to_brep(meniscus_slices, spline_order=1)) == [(1, 1)]


def test_loft_rejects_contours_with_unequal_point_counts(meniscus_slices):
    shorter = replace(meniscus_slices[1], points=meniscus_slices[1].points[::2])

    with pytest.raises(ValueError, match="same number of points"):
        contours_to_brep([meniscus_slices[0], shorter, *meniscus_slices[2:]])


@pytest.mark.parametrize("fixture", ["meniscus_solid", "meniscus_domed_solid"])
def test_solid_can_be_meshed_by_gmsh(request, fixture, tmp_path):
    with gmsh_session():
        import_brep(request.getfixturevalue(fixture), tmp_path)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 2.0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 4.0)
        gmsh.model.mesh.generate(3)

        tetrahedra = gmsh.model.mesh.getElementsByType(4)[0]

    assert len(tetrahedra) > 0


@pytest.mark.parametrize("order", [1, 2, 5])
def test_smooth_caps_work_at_other_spline_orders(meniscus_slices, tmp_path, order):
    with gmsh_session():
        import_brep(contours_to_brep(meniscus_slices, spline_order=order, cap="smooth"), tmp_path)

        assert len(gmsh.model.getEntities(3)) == 1


def test_brep_rejects_unknown_cap(meniscus_slices):
    with pytest.raises(ValueError, match="cap must be"):
        contours_to_brep(meniscus_slices, cap="round")  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("offset", [0.0, -1.0])
def test_brep_rejects_non_positive_cap_offset(meniscus_slices, offset):
    with pytest.raises(ValueError, match="cap_offset"):
        contours_to_brep(meniscus_slices, cap="smooth", cap_offset=offset)


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


def test_hex_blocks_are_five_six_sided_blocks_sharing_faces(meniscus_hex_blocks, tmp_path):
    with gmsh_session():
        import_brep(meniscus_hex_blocks, tmp_path)

        # Two caps of 8 vertices and 12 edges each, joined by 8 edges along the
        # curve; 4 outer, 4 shared spoke, and 4 shared inner faces between the 10 cap faces.
        assert entity_counts() == [16, 32, 22, 5]
        for _, volume in gmsh.model.getEntities(3):
            assert len(gmsh.model.getBoundary([(3, volume)], oriented=False)) == 6
        assert sum(gmsh.model.getType(2, face) == "Plane" for _, face in gmsh.model.getEntities(2)) == 10


def test_hex_blocks_surround_one_inner_block(meniscus_hex_blocks, tmp_path):
    with gmsh_session():
        import_brep(meniscus_hex_blocks, tmp_path)
        sides = [
            {tag for _, tag in gmsh.model.getBoundary([(3, volume)], oriented=False) if gmsh.model.getType(2, tag) != "Plane"}
            for _, volume in gmsh.model.getEntities(3)
        ]

    # The inner block shares all four sides; each outer block shares three and keeps one on the boundary.
    assert sorted(sum(1 for face in own if any(face in other for other in sides if other is not own)) for own in sides) == [3, 3, 3, 3, 4]


def test_hex_blocks_split_contours_away_from_sharp_corners(meniscus_slices):
    from kneemesh.geo.meniscus import _butterfly_split_indices, _turning_angles

    turns = np.mean([_turning_angles(contour.points) for contour in meniscus_slices], axis=0)
    splits = _butterfly_split_indices(meniscus_slices)

    nearest = np.rint(splits).astype(int) % len(turns)
    assert np.all(turns[nearest] < 0.25 * turns.max())
    assert np.diff(splits) == pytest.approx(np.full(3, len(turns) / 4), abs=1.0)


def test_hex_blocks_fill_the_planar_capped_solid(meniscus_hex_blocks, meniscus_hex_slices, tmp_path):
    def total_volume(brep):
        with gmsh_session():
            import_brep(brep, tmp_path)
            return sum(gmsh.model.occ.getMass(3, volume) for _, volume in gmsh.model.getEntities(3))

    assert total_volume(meniscus_hex_blocks) == pytest.approx(total_volume(contours_to_brep(meniscus_hex_slices, cap="planar")), rel=0.02)


def test_hex_blocks_pass_through_contour_points(meniscus_hex_blocks, meniscus_hex_slices, tmp_path):
    points = np.vstack([contour.points for contour in meniscus_hex_slices])

    with gmsh_session():
        import_brep(meniscus_hex_blocks, tmp_path)
        outer = [tag for _, tag in gmsh.model.getEntities(2) if gmsh.model.getType(2, tag) != "Plane"]
        distances = np.full(len(points), np.inf)
        for face in outer:
            closest = np.array(gmsh.model.getClosestPoint(2, face, points.ravel().tolist())[0]).reshape(-1, 3)
            distances = np.minimum(distances, np.linalg.norm(closest - points, axis=1))

    assert distances.max() < 1e-3


@pytest.mark.parametrize("order", [1, 2, 5])
def test_hex_blocks_build_at_other_spline_orders(meniscus_hex_slices, tmp_path, order):
    with gmsh_session():
        import_brep(contours_to_hex_blocks(meniscus_hex_slices, spline_order=order), tmp_path)

        assert entity_counts() == [16, 32, 22, 5]


def test_hex_blocks_reject_unequal_point_counts(meniscus_hex_slices, meniscus_slices):
    with pytest.raises(ValueError, match="same number of points"):
        contours_to_hex_blocks([meniscus_hex_slices[0], meniscus_slices[1]])


def test_hex_blocks_reject_too_few_points():
    slices = [make_slice([[0, 0, z], [1, 0, z], [1, 1, z], [0, 1, z], [-1, 1, z], [-1, 0, z], [-1, -1, z]]) for z in (0.0, 1.0)]

    with pytest.raises(ValueError, match="at least 8 points"):
        contours_to_hex_blocks(slices, spline_order=2)


def test_hex_blocks_reject_too_few_slices(meniscus_hex_slices):
    with pytest.raises(ValueError, match="at least 2"):
        contours_to_hex_blocks(meniscus_hex_slices[:1])


def surface_folds(brep, tmp_path):
    """Count sign flips of the surface normal between neighboring samples of each non-planar face."""
    folds = 0
    with gmsh_session():
        import_brep(brep, tmp_path)
        for _, face in gmsh.model.getEntities(2):
            if gmsh.model.getType(2, face) == "Plane":
                continue
            (u_low, v_low), (u_high, v_high) = gmsh.model.getParametrizationBounds(2, face)
            u, v = np.linspace(u_low, u_high, 161), np.linspace(v_low, v_high, 151)
            parameters = np.array([[a, b] for b in v for a in u])
            derivatives = np.array(gmsh.model.getDerivative(2, face, parameters.ravel().tolist())).reshape(len(v), len(u), 2, 3)
            normals = np.cross(derivatives[..., 0, :], derivatives[..., 1, :])
            normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-30)
            folds += int(((normals[1:] * normals[:-1]).sum(-1) < 0).sum() + ((normals[:, 1:] * normals[:, :-1]).sum(-1) < 0).sum())
    return folds


@pytest.mark.parametrize(("assembled", "smoothing_iterations"), [(False, 0), (True, DEFAULT_SMOOTHING_ITERATIONS)])
@pytest.mark.parametrize("order", [2, 3])
def test_loft_of_real_meniscus_does_not_fold(real_label, tmp_path, order, assembled, smoothing_iterations):
    # The pipeline's label Gaussian assembly and smoothing gave sections that
    # made addThruSections fold the surface, where the raw label did not.
    label = assemble_label_maps({"lateral_meniscus": real_label}).image if assembled else real_label
    surface = smooth_isocontour(extract_isocontour(signed_distance_field(label)), smoothing_iterations)
    slices = slice_isocontour(surface, centerline(label), spacing=5.0)

    # The real sections taper quickly toward the horns. Open Cascade's
    # addThruSections overshoots there and folds the surface, which stalls
    # gmsh's surface meshing indefinitely.
    assert surface_folds(contours_to_brep(slices, spline_order=order, cap="planar"), tmp_path) == 0
