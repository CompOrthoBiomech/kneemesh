"""Removing overclosure: where built structures' geometry overlaps, moving it apart after every structure is built.

Every structure is built from its own label, and smoothing, thickening, and
fitting let neighbors overlap by a few tenths of a millimeter: on the example
knee the medial meniscus reaches 0.45 mm into its tibial cartilage, the
cartilages 0.15 mm into their bones, and the femoral and tibial cartilages
0.1 mm into each other. resolve_overclosure moves them apart by a priority:

- Cartilage is thin, so it never yields to a meniscus or bone. Where two
  cartilages overlap, both articular sides are lowered toward their bones,
  each by its share of the two local thicknesses, t_a / (t_a + t_b), and the
  sheets rebuilt (geo.sheet.lower_articular).
- Every other structure, in priority order, is pushed out of the union of
  every structure ranked above it: a meniscus by moving its contour points
  out within their slice planes and lofting them again, which keeps its
  planar caps and both hexahedral designs, and a bone by moving its surface's
  vertices.

Overlaps are measured against each B-rep's outer boundary, triangulated
finely (brep_boundary), and each bone's surface.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

import gmsh
import numpy as np
from loguru import logger
from scipy.spatial import KDTree
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkCleanPolyData, vtkImplicitPolyDataDistance, vtkPolyDataNormals

from kneemesh.brep import BRep, gmsh_session
from kneemesh.geo.bone import surface_arrays
from kneemesh.geo.sheet import CartilageSheet, lower_articular, sheet_to_brep, surface_grid
from kneemesh.tet_meshing import gmsh_options, import_solid
from kneemesh.vol import ContourSlice

# Edge length, in physical units, of the triangles measuring a B-rep's boundary. Its chordal error on the
# cartilage's and menisci's curvature is about a thousandth of a millimeter.
_TESSELLATION_SIZE = 0.5
# Overlap, in physical units, left unresolved.
_TOLERANCE = 1e-3
# Rounds of measuring and moving.
_ITERATIONS = 10
_CARTILAGE_ITERATIONS = 6
_LOFT_ROUNDS = 4
# A pushed point of a meniscus's lofted boundary also pushes the contour points within this fraction of the diagonal
# of the gaps between them: the corners of the gap it lies in.
_NEIGHBOR_REACH = 0.75
# A contour point's push is projected into its slice's plane; projections shorter than this fraction of the push
# are lengthened only as far as it, so a push nearly along the curve cannot fling a point across the section.
_MIN_IN_PLANE = 0.3
# Overlap left after moving, in physical units, above which a warning is logged: below it lies within the
# tessellations' accuracy, a few thousandths of a millimeter on the example knee.
_RESIDUAL_WARNING = 0.01
# A point of a cartilage's boundary counts as on its articular side within this height of it, in physical units.
_ARTICULAR_MATCH = 0.05
# A lowered cartilage warns when it ends thinner than this fraction of its minimum thickness.
_THIN_WARNING = 0.5


@dataclass(frozen=True, eq=False)
class Adjustable:
    """A structure's geometry and what it was built from, so it can be moved and built again.

    Give sheet for a cartilage, slices and loft for a meniscus, and neither
    for a bone, whose geometry is its surface. A B-rep with neither is never
    moved.

    Attributes:
        geometry: The B-rep, or a bone's closed triangulated surface.
        sheet: The cartilage sheet the B-rep was built from (geo.sheet.sheet_to_brep).
        slices: The contours the meniscus's B-rep was lofted through.
        loft: Builds the meniscus's B-rep from contours again, as it was built.
    """

    geometry: BRep | vtkPolyData
    sheet: CartilageSheet | None = None
    slices: list[ContourSlice] | None = None
    loft: Callable[[list[ContourSlice]], BRep] | None = None


class Obstacle:
    """A closed triangulated surface whose inside other geometry must leave, with signed distances to it."""

    def __init__(self, surface: vtkPolyData) -> None:
        """Orient a closed surface's normals outward and prepare distance queries."""
        clean = vtkCleanPolyData()
        clean.SetInputData(surface)
        normals = vtkPolyDataNormals()
        normals.SetInputConnection(clean.GetOutputPort())
        normals.AutoOrientNormalsOn()
        normals.ConsistencyOn()
        normals.SplittingOff()
        normals.ComputeCellNormalsOn()
        normals.Update()
        self.surface: vtkPolyData = normals.GetOutput()
        self._distance: vtkImplicitPolyDataDistance = vtkImplicitPolyDataDistance()
        self._distance.SetInput(self.surface)
        bounds = np.array(self.surface.GetBounds())
        self._low: np.ndarray = bounds[0::2]
        self._high: np.ndarray = bounds[1::2]

    def distances(self, points: np.ndarray, reach: float) -> tuple[np.ndarray, np.ndarray]:
        """Return each of (N, 3) points' signed distance to the surface, negative inside, and its nearest point on it.

        Points farther than reach outside the surface's bounding box are not
        measured: their distance is infinite and their nearest point their own.
        """
        signed = np.full(len(points), np.inf)
        nearest = points.copy()
        near = np.all((points > self._low - reach) & (points < self._high + reach), axis=1)
        closest = [0.0, 0.0, 0.0]
        for index in np.flatnonzero(near):
            signed[index] = self._distance.EvaluateFunctionAndGetClosestPoint(points[index].tolist(), closest)
            nearest[index] = closest
        return signed, nearest

    @property
    def points(self) -> np.ndarray:
        """The surface's (N, 3) vertices."""
        return vtk_to_numpy(self.surface.GetPoints().GetData()).astype(np.float64)

    def pushes(self, points: np.ndarray, clearance: float) -> np.ndarray:
        """Return the (N, 3) moves that take each point inside, or within clearance outside, to clearance outside; zero for the rest."""
        short, nearest, outward = self._short(points, clearance)
        moves = np.zeros_like(points)
        moves[short] = nearest + clearance * outward - points[short]
        return moves

    def recessions(self, points: np.ndarray, clearance: float) -> tuple[np.ndarray, np.ndarray]:
        """Return where this surface must move so that (N, 3) points lie clearance outside it, and how.

        Returns:
            The (M, 3) nearest surface points of the points inside or within
            clearance outside, and the (M, 3) moves that take the surface
            there clearance past them, inward.
        """
        short, nearest, outward = self._short(points, clearance)
        return nearest, points[short] - nearest - clearance * outward

    def _short(self, points: np.ndarray, clearance: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return which points lie inside or within clearance outside, and their nearest surface points and outward normals there."""
        signed, nearest = self.distances(points, clearance + _TOLERANCE)
        short = signed < clearance - _TOLERANCE
        nearest = nearest[short]
        offset = points[short] - nearest
        length = np.linalg.norm(offset, axis=1, keepdims=True)
        # Outward is from the nearest point to an outside point, and the reverse for one inside; a point on the surface takes its normal.
        outward = np.where(length > 1e-12, offset / np.maximum(length, 1e-12) * np.sign(signed[short])[:, np.newaxis], 0.0)
        on_surface = length[:, 0] <= 1e-12
        if on_surface.any():
            outward[on_surface] = self._normals_near(nearest[on_surface])
        return short, nearest, outward

    def depth(self, points: np.ndarray) -> float:
        """Return how far the deepest of (N, 3) points lies inside the surface, 0 if none does."""
        signed, _ = self.distances(points, _TOLERANCE)
        return float(max(0.0, -signed.min())) if len(signed) else 0.0

    def _normals_near(self, points: np.ndarray) -> np.ndarray:
        normals = np.empty_like(points)
        for index, point in enumerate(points):
            gradient = [0.0, 0.0, 0.0]
            self._distance.EvaluateGradient(point.tolist(), gradient)
            normals[index] = np.asarray(gradient) / max(float(np.linalg.norm(gradient)), 1e-12)
        return normals


def brep_boundary(brep: BRep, size: float = _TESSELLATION_SIZE) -> vtkPolyData:
    """Triangulate a solid B-rep's outer boundary, leaving out faces between its volumes, such as a meniscus's butterfly blocks'."""
    with gmsh_session():
        import_solid(brep)
        boundary = {abs(tag) for _, tag in gmsh.model.getBoundary(gmsh.model.getEntities(3), combined=True, oriented=False)}
        options = {"Mesh.MeshSizeMax": size, "Mesh.MeshSizeMin": size / 4, "Mesh.Algorithm": 6, "Mesh.MeshSizeFromCurvature": 0}
        with gmsh_options(options):
            gmsh.model.mesh.generate(2)
        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        index = np.zeros(int(np.max(node_tags)) + 1, dtype=np.int64)
        index[np.asarray(node_tags, dtype=np.int64)] = np.arange(len(node_tags))
        triangles = []
        for face in sorted(boundary):
            _, element_nodes = gmsh.model.mesh.getElementsByType(2, face)
            triangles.append(index[np.asarray(element_nodes, dtype=np.int64)].reshape(-1, 3))
    return _polydata(np.asarray(coordinates, dtype=np.float64).reshape(-1, 3), np.vstack(triangles))


def surface_obstacle(geometry: BRep | vtkPolyData) -> Obstacle:
    """Return a structure's geometry as an Obstacle: a B-rep's outer boundary, or a bone's surface."""
    return Obstacle(brep_boundary(geometry) if isinstance(geometry, BRep) else geometry)


def resolve_overclosure(structures: Mapping[str, Adjustable], priority: Sequence[str], clearance: float = 0.0) -> dict[str, Adjustable]:
    """Move overlapping structures apart, leaving clearance between them.

    Cartilages, those with a sheet, share each overlap between them by
    thickness whatever their order in priority. Every other structure, in
    priority order, highest first, is pushed out of every structure ranked
    above it, as moved. A structure missing from priority, or a B-rep with
    neither sheet nor slices, is never moved, but still pushes others ranked
    below it.

    Returns:
        Every structure, moved where it overlapped.

    Raises:
        ValueError: If a cartilage ranks below a structure that can be moved, a meniscus or bone.
    """
    ranked = [name for name in priority if name in structures] + [name for name in structures if name not in priority]
    cartilages = [name for name in ranked if structures[name].sheet is not None]
    if cartilages and (movers := [name for name in ranked[: ranked.index(cartilages[-1])] if _movable(structures[name])]):
        raise ValueError(f"Cartilage is never moved for another structure, so it must rank above {', '.join(movers)}; got {ranked}")
    resolved = dict(structures)
    resolved.update(_share_cartilage_overlaps({name: structures[name] for name in cartilages}, clearance))
    obstacles: dict[str, Obstacle] = {}
    for rank, name in enumerate(ranked):
        if name in cartilages:
            continue
        above = ranked[:rank]
        if above:
            for other in above:
                if other not in obstacles:
                    obstacles[other] = surface_obstacle(resolved[other].geometry)
            resolved[name] = _push_out(name, resolved[name], {other: obstacles[other] for other in above}, clearance)
    return resolved


def _movable(structure: Adjustable) -> bool:
    """Whether a structure other than a cartilage can be moved: a bone's surface, or a meniscus with its contours."""
    return isinstance(structure.geometry, vtkPolyData) or (structure.slices is not None and structure.loft is not None)


def push_surface_out(surface: vtkPolyData, obstacles: Sequence[Obstacle], clearance: float = 0.0) -> vtkPolyData:
    """Move a closed triangulated surface's vertices so it lies clearance outside the obstacles, keeping its triangles.

    Each round moves the vertices inside an obstacle out of it, and where an
    obstacle's own vertices lie inside the surface, between its vertices,
    the surface's vertices around them that face them inward (_spread).
    """
    points, triangles = surface_arrays(surface)
    points = points.copy()
    reach = _NEIGHBOR_REACH * 2 * float(np.median(np.linalg.norm(points[triangles[:, 0]] - points[triangles[:, 1]], axis=1)))
    for _ in range(_ITERATIONS):
        current = _polydata(points, triangles)
        pushes = _combined_pushes(points, obstacles, clearance)
        locations, demands = _recessions(Obstacle(current), obstacles, clearance)
        if not pushes.any() and not len(demands):
            break
        points += _spread(points, _vertex_normals(current), pushes, locations, demands, reach)
    return _polydata(points, triangles)


def push_meniscus_out(
    slices: list[ContourSlice], loft: Callable[[list[ContourSlice]], BRep], obstacles: Sequence[Obstacle], clearance: float = 0.0
) -> tuple[list[ContourSlice], BRep]:
    """Move a meniscus's contours so its loft lies clearance outside the obstacles, lofting them again in rounds.

    Each contour point moves within its slice's plane (_move_contours). The
    loft interpolates the contours but can still reach into an obstacle
    between them, so each round also measures the lofted boundary, both its
    points inside obstacles and obstacles' points inside it, and those
    moves also move the contour points around them that face them (_spread).

    Returns:
        The moved contours and their loft.
    """
    reach = _NEIGHBOR_REACH * _contour_spacing(slices)
    sizes = [len(contour.points) for contour in slices]
    brep = loft(slices)
    for _ in range(_LOFT_ROUNDS):
        points = np.vstack([contour.points for contour in slices])
        boundary = Obstacle(brep_boundary(brep))
        pushes = _combined_pushes(points, obstacles, clearance)
        boundary_points = boundary.points
        boundary_pushes = _combined_pushes(boundary_points, obstacles, clearance)
        pushed = np.linalg.norm(boundary_pushes, axis=1) > 0
        locations, demands = _recessions(boundary, obstacles, clearance)
        locations, demands = np.vstack((boundary_points[pushed], locations)), np.vstack((boundary_pushes[pushed], demands))
        if not pushes.any() and not len(demands):
            break
        normals = np.vstack([_outward_normals(contour) for contour in slices])
        moves = _spread(points, normals, pushes, locations, demands, reach)
        slices = _move_contours(slices, np.split(moves, np.cumsum(sizes)[:-1]))
        brep = loft(slices)
        logger.debug("Moved {} contour points by up to {:.3g} and lofted them again", *_move_summary(points, points + moves))
    return slices, brep


def _recessions(surface: Obstacle, obstacles: Sequence[Obstacle], clearance: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the points of a surface the obstacles' vertices lie inside, or within clearance of, and the inward moves each needs."""
    found = [surface.recessions(obstacle.points, clearance) for obstacle in obstacles]
    if not found:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.vstack([locations for locations, _ in found]), np.vstack([moves for _, moves in found])


def _spread(points: np.ndarray, normals: np.ndarray, pushes: np.ndarray, locations: np.ndarray, demands: np.ndarray, reach: float) -> np.ndarray:
    """Return each point's move: its own push, or the longest of the demands within reach of it that push it inward.

    A demand at a location on the surface between the points moves the
    points around it. Only points facing the demand take it, those whose
    outward normals it opposes: across a thin edge, such as a meniscus's
    inner edge, a point on the other side would be pushed into the obstacle
    there.
    """
    moves = pushes.copy()
    if not len(demands):
        return moves
    lengths = np.linalg.norm(moves, axis=1)
    tree = KDTree(points)
    for demand, neighbors in zip(demands, tree.query_ball_point(locations, reach), strict=True):
        length = float(np.linalg.norm(demand))
        for neighbor in neighbors:
            if length > lengths[neighbor] and demand @ normals[neighbor] < 0:
                moves[neighbor], lengths[neighbor] = demand, length
    return moves


def _combined_pushes(points: np.ndarray, obstacles: Sequence[Obstacle], clearance: float) -> np.ndarray:
    """Return one round's moves out of every obstacle in turn, each measured from where the previous left the points."""
    total = np.zeros_like(points)
    for obstacle in obstacles:
        total += obstacle.pushes(points + total, clearance)
    return total


def _move_contours(slices: Sequence[ContourSlice], moves: Sequence[np.ndarray]) -> list[ContourSlice]:
    """Move each contour's points by (N, 3) moves projected into the contour's plane.

    Each move is projected into its slice's plane and lengthened so its
    component along the move is kept, as far as _MIN_IN_PLANE allows. A
    clipped contour's cut (ContourSlice.straight_from) moves as a whole, by
    its longest move, so it stays straight.
    """
    moved = []
    for contour, move in zip(slices, moves, strict=True):
        lengths = np.linalg.norm(move, axis=1)
        if not lengths.any():
            moved.append(contour)
            continue
        in_plane = move - np.outer(move @ contour.normal, contour.normal)
        planar = np.linalg.norm(in_plane, axis=1)
        denominator = np.maximum(planar, _MIN_IN_PLANE * lengths)
        scale = np.divide(lengths, denominator, out=np.zeros_like(lengths), where=denominator > 0)
        in_plane *= scale[:, np.newaxis]
        cut = _cut_indices(contour)
        if len(cut):
            in_plane[cut] = in_plane[cut[np.argmax(lengths[cut])]]
        points = contour.points + in_plane
        moved.append(replace(contour, points=points, area=_contour_area(contour, points)))
    return moved


def _push_out(name: str, structure: Adjustable, obstacles: Mapping[str, Obstacle], clearance: float) -> Adjustable:
    """Move a meniscus or bone out of the obstacles, logging how far it reached into each before and after."""
    geometry = structure.geometry
    if isinstance(geometry, vtkPolyData):
        own = Obstacle(geometry)
    else:
        own = Obstacle(brep_boundary(geometry))
    if not (depths := _overlaps(own, obstacles)):
        return structure
    _log_depths(name, depths)
    if isinstance(geometry, vtkPolyData):
        points = surface_arrays(geometry)[0]
        moved = push_surface_out(geometry, list(obstacles.values()), clearance)
        count, largest = _move_summary(points, surface_arrays(moved)[0])
        logger.info("Moved {} of the {} vertices of {} by up to {:.3g}", count, len(points), name, largest)
        result = replace(structure, geometry=moved)
    elif structure.slices is not None and structure.loft is not None:
        points = np.vstack([contour.points for contour in structure.slices])
        slices, brep = push_meniscus_out(structure.slices, structure.loft, list(obstacles.values()), clearance)
        count, largest = _move_summary(points, np.vstack([contour.points for contour in slices]))
        logger.info("Moved {} of the {} contour points of {} by up to {:.3g} and lofted them again", count, len(points), name, largest)
        result = replace(structure, geometry=brep, slices=slices)
    else:
        logger.warning("{} cannot be moved, so it still overlaps", name)
        return structure
    if residual := _overlaps(surface_obstacle(result.geometry), obstacles):
        _log_residual(name, residual)
    return result


def _overlaps(surface: Obstacle, obstacles: Mapping[str, Obstacle]) -> dict[str, float]:
    """Return how deep a surface and each obstacle it overlaps reach into each other: the deeper of its vertices in the obstacle and the reverse."""
    depths = {name: max(obstacle.depth(surface.points), surface.depth(obstacle.points)) for name, obstacle in obstacles.items()}
    return {name: depth for name, depth in depths.items() if depth > _TOLERANCE}


def _vertex_normals(surface: vtkPolyData) -> np.ndarray:
    """Return a closed surface's unit outward normal at each vertex, in its own point order."""
    normals = vtkPolyDataNormals()
    normals.SetInputData(surface)
    normals.AutoOrientNormalsOn()
    normals.ConsistencyOn()
    normals.SplittingOff()
    normals.Update()
    output = normals.GetOutput()
    if output.GetNumberOfPoints() != surface.GetNumberOfPoints():
        raise ValueError("The surface's vertex normals do not match its vertices")
    return vtk_to_numpy(output.GetPointData().GetNormals()).astype(np.float64)


def _outward_normals(contour: ContourSlice) -> np.ndarray:
    """Return the unit outward normal, in its plane, at each point of a contour, which runs counterclockwise about the plane's normal."""
    tangents = np.roll(contour.points, -1, axis=0) - np.roll(contour.points, 1, axis=0)
    normals = np.cross(tangents, contour.normal)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def _contour_spacing(slices: Sequence[ContourSlice]) -> float:
    """Return the diagonal of a typical gap between contour points: the median spacing along the curve and around the contours."""
    centers = np.array([contour.center for contour in slices])
    along = float(np.median(np.linalg.norm(np.diff(centers, axis=0), axis=1))) if len(slices) > 1 else 0.0
    around = float(np.median(np.concatenate([np.linalg.norm(contour.points - np.roll(contour.points, 1, axis=0), axis=1) for contour in slices])))
    return float(np.hypot(along, around))


def _share_cartilage_overlaps(cartilages: Mapping[str, Adjustable], clearance: float) -> dict[str, Adjustable]:
    """Lower overlapping cartilages' articular sides, each by its share of the local thicknesses, until none overlaps another."""
    if len(cartilages) < 2:
        return dict(cartilages)
    base = {name: structure.sheet for name, structure in cartilages.items() if structure.sheet is not None}
    grids = {name: _articular_grid(sheet) for name, sheet in base.items()}
    lowering = {name: np.zeros((len(grid[0]), len(grid[1]))) for name, grid in grids.items()}
    current = dict(cartilages)
    sheets = dict(base)
    for iteration in range(_CARTILAGE_ITERATIONS):
        obstacles = {name: surface_obstacle(structure.geometry) for name, structure in current.items()}
        changed: set[str] = set()
        for name in base:
            s, _, domain, covered = grids[name]
            sheet = sheets[name]
            nodes = domain[covered]
            points = sheet.points(nodes, "articular")
            thickness = sheet.thickness(nodes)
            added = np.zeros(len(points))
            tree = KDTree(nodes)
            reach = _NEIGHBOR_REACH * np.sqrt(2) * float(s[1] - s[0])
            for other, obstacle in obstacles.items():
                if other == name or other not in sheets:
                    continue
                other_sheet = sheets[other]
                # Grid points of this articular side inside the other cartilage lower by their share of the overlap.
                signed, nearest = obstacle.distances(points, clearance + _TOLERANCE)
                short = signed < clearance - _TOLERANCE
                if short.any():
                    if iteration == 0:
                        logger.info("{} overlaps {} by up to {:.3g}", name, other, float(max(0.0, -signed.min())))
                    other_thickness = np.maximum(other_sheet.thickness(other_sheet.frame.local(nearest[short])[0]), 0.0)
                    share = thickness[short] / np.maximum(thickness[short] + other_thickness, 1e-12)
                    added[short] = np.maximum(added[short], share * (clearance - signed[short]))
                # The other's points inside this cartilage, between its grid points, lower the grid points around them.
                locations, moves = obstacles[name].recessions(obstacle.points, clearance)
                if not len(moves):
                    continue
                domain_points, heights = sheet.frame.local(locations)
                on_articular = np.abs(heights - sheet.articular.ev(domain_points[:, 0], domain_points[:, 1])) < _ARTICULAR_MATCH
                demands = np.linalg.norm(moves[on_articular], axis=1)
                own_thickness = np.maximum(sheet.thickness(domain_points[on_articular]), 0.0)
                other_thickness = np.maximum(other_sheet.thickness(other_sheet.frame.local(locations[on_articular])[0]), 0.0)
                demands *= own_thickness / np.maximum(own_thickness + other_thickness, 1e-12)
                for demand, neighbors in zip(demands, tree.query_ball_point(domain_points[on_articular], reach), strict=True):
                    for neighbor in neighbors:
                        added[neighbor] = max(added[neighbor], demand)
            if added.any():
                lowering[name].reshape(-1)[np.flatnonzero(covered)] += added
                changed.add(name)
        logger.debug("Round {} of sharing overlaps lowers {}", iteration + 1, ", ".join(sorted(changed)) or "nothing")
        if not changed:
            break
        for name in sorted(changed):
            s, t, _, _ = grids[name]
            sheets[name] = lower_articular(base[name], s, t, lowering[name])
            current[name] = replace(current[name], sheet=sheets[name], geometry=sheet_to_brep(sheets[name]))
    for name in base:
        if lowering[name].any():
            _, _, domain, covered = grids[name]
            sheet = sheets[name]
            thinnest = float(sheet.thickness(domain[covered]).min())
            logger.info("Lowered the articular side of {} by up to {:.3g}; it is at least {:.3g} thick", name, float(lowering[name].max()), thinnest)
            if thinnest < _THIN_WARNING * sheet.min_thickness:
                logger.warning("{} is thinner than half its minimum thickness where it was lowered", name)
    if any(lowering[name].any() for name in base):
        obstacles = {name: surface_obstacle(structure.geometry) for name, structure in current.items()}
        names = list(obstacles)
        for first, name in enumerate(names):
            later = {other: obstacles[other] for other in names[first + 1 :]}
            if residual := _overlaps(obstacles[name], later):
                _log_residual(name, residual)
    return current


def _articular_grid(sheet: CartilageSheet) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the sheet's surface grid's s and t, its (len(s) * len(t), 2) domain points, s-major, and which of them the sheet covers.

    Points within two grid spacings outside the perimeter count, so a
    lowering reaching the rim does not step down to zero across it.
    """
    s, t = surface_grid(sheet)
    domain = np.stack(np.meshgrid(s, t, indexing="ij"), axis=-1).reshape(-1, 2)
    return s, t, domain, sheet.covers(domain, margin=2 * float(s[1] - s[0]))


def _describe(depths: Mapping[str, float]) -> str:
    return ", ".join(f"{name} by up to {depth:.3g}" for name, depth in depths.items())


def _log_residual(name: str, residual: Mapping[str, float]) -> None:
    """Log what a moved structure still overlaps, warning if any is deeper than _RESIDUAL_WARNING."""
    level = "WARNING" if max(residual.values()) > _RESIDUAL_WARNING else "INFO"
    logger.log(level, "{} still overlaps {}", name, _describe(residual))


def _log_depths(name: str, depths: Mapping[str, float]) -> None:
    logger.info("{} overlaps {}", name, _describe(depths))


def _move_summary(before: np.ndarray, after: np.ndarray) -> tuple[int, float]:
    moves = np.linalg.norm(after - before, axis=1)
    return int(np.count_nonzero(moves > 0)), float(moves.max())


def _cut_indices(contour: ContourSlice) -> np.ndarray:
    """Return the indices of a clipped contour's cut, its two corners and the points between them, or none."""
    if contour.straight_from is None:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate((np.arange(contour.straight_from, len(contour.points)), [0]))


def _contour_area(contour: ContourSlice, points: np.ndarray) -> float:
    u = (points - contour.center) @ contour.u_axis
    v = (points - contour.center) @ contour.v_axis
    return 0.5 * abs(float(np.sum(u * np.roll(v, -1) - np.roll(u, -1) * v)))


def _polydata(points: np.ndarray, triangles: np.ndarray) -> vtkPolyData:
    surface = vtkPolyData()
    vtk_points = vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float64), deep=True))
    surface.SetPoints(vtk_points)
    cells = vtkCellArray()
    cells.SetData(3, numpy_to_vtk(np.ascontiguousarray(triangles, dtype=np.int64).ravel(), deep=True))
    surface.SetPolys(cells)
    return surface
