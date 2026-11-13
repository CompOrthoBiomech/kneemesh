"""Configuration models and parsing them from JSON."""

import json
from os import PathLike
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Self, get_args

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from kneemesh.cartilage import DEFAULT_SIDE_COSINE
from kneemesh.disk import BREP_SUFFIXES
from kneemesh.geo.sheet import (
    DEFAULT_HOLE_THICKNESS,
    DEFAULT_MIN_THICKNESS_HEXAHEDRAL,
    DEFAULT_MIN_THICKNESS_TETRAHEDRAL,
    DEFAULT_PERIMETER_RADIUS_HEXAHEDRAL,
    DEFAULT_PERIMETER_RADIUS_TETRAHEDRAL,
)
from kneemesh.mesh_optimization import (
    DEFAULT_FEATURE_ANGLE,
    DEFAULT_LAPLACIAN_ITERATIONS,
    DEFAULT_OPTIMIZATION_ITERATIONS,
    DEFAULT_SURFACE_DEVIATION,
    DEFAULT_SURFACE_TRAVEL,
)
from kneemesh.segmentation import DEFAULT_PADDING_VOXELS, Interpolator
from kneemesh.tet_meshing import Algorithm2D, Algorithm3D
from kneemesh.vol import DEFAULT_SMOOTHING_ITERATIONS, DEFAULT_SMOOTHING_PASS_BAND


class BaseConfig(BaseModel):
    """Base class for configuration models.

    Unknown fields are rejected so that misspelled options fail at load time
    instead of being silently ignored.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    @classmethod
    def from_json(cls, path: str | PathLike[str]) -> Self:
        """Parse and validate a configuration from a JSON file."""
        return cls.model_validate_json(Path(path).read_bytes())


def _none_from_text(value: Any) -> Any:
    """Read "none" or "null" as None, so command-line overrides can clear optional values."""
    return None if isinstance(value, str) and value.strip().lower() in ("none", "null") else value


def _json_from_text(value: Any) -> Any:
    """Parse JSON text, so command-line overrides can give mappings such as '{"tibia": 1}', and read "none" as None."""
    value = _none_from_text(value)
    return json.loads(value) if isinstance(value, str) else value


Structure = Literal[
    "lateral_meniscus",
    "medial_meniscus",
    "patellar_cartilage",
    "femoral_cartilage",
    "lateral_tibial_cartilage",
    "medial_tibial_cartilage",
    "femur",
    "tibia",
    "patella",
]
"""Names of the knee structures. Keys of labels, label_maps, and breps must be among them, and each picks its pipeline."""

STRUCTURES: tuple[Structure, ...] = get_args(Structure)

BONES: dict[Structure, Structure] = {
    "patellar_cartilage": "patella",
    "femoral_cartilage": "femur",
    "lateral_tibial_cartilage": "tibia",
    "medial_tibial_cartilage": "tibia",
}
"""The bone under each cartilage, which must be in the same segmentation to orient it."""

ElementType = Literal["hex8", "hex20", "hex27", "tet4", "tet10"]
"""Element shape and node count: 8-node, 20-node serendipity, or 27-node hexahedra, or 4- or 10-node tetrahedra."""

OptionalFloat = Annotated[float | None, BeforeValidator(_none_from_text)]
OptionalPath = Annotated[Path | None, BeforeValidator(_none_from_text)]
LabelValues = Annotated[dict[Structure, int] | None, BeforeValidator(_json_from_text)]
StructurePaths = Annotated[dict[Structure, Path] | None, BeforeValidator(_json_from_text)]


def _label_maps_field(summary: str) -> Any:
    return Field(
        default=None,
        description=f'{summary}, by structure name, as JSON such as {{"tibia": "tibia.nii"}}. Each file\'s nonzero voxels are its structure',
    )


def _interpolator_field() -> Any:
    return Field(default="label_gaussian", description="Resampling interpolator for assembling label_maps: label_gaussian or nearest_neighbor")


def _padding_field() -> Any:
    return Field(default=DEFAULT_PADDING_VOXELS, ge=0, description="Background voxels around the assembled label_maps on each side")


def _output_directory_field() -> Any:
    return Field(default=Path("."), description="Directory to write the output files in, created if missing")


def _check_suffix(suffix: str, extensions: frozenset[str] | None = None) -> None:
    """Raise ValueError unless suffix is a file name ending with an extension, from extensions if given."""
    if "/" in suffix or "\\" in suffix:
        raise ValueError(f"The output suffix is appended to each structure's name and cannot hold a directory, got {suffix!r}")
    extension = Path(f"structure{suffix}").suffix
    if not extension:
        raise ValueError(f"The output suffix needs a file extension, got {suffix!r}")
    if extensions is not None and extension not in extensions:
        raise ValueError(f"The output suffix needs extension {' or '.join(sorted(extensions))}, got {suffix!r}")


class LabelMapAssembly(BaseConfig):
    """Settings for assembling single-label LabelMaps into one multilabel LabelMap.

    Structures get label values 1, 2, ... in the order of label_maps.
    """

    label_maps: StructurePaths = _label_maps_field("Single-label LabelMaps to assemble")
    interpolator: Interpolator = _interpolator_field()
    padding: int = _padding_field()
    output: Path = Field(
        default=Path("label_map.nrrd"),
        description="Multilabel LabelMap to write: .nii, .nrrd, or .mhd, with its labels in <stem>_labels.json beside it",
    )


class SurfaceBrep(BaseConfig):
    """Settings shared by every structure for extracting a smooth surface from its label."""

    sigma: OptionalFloat = Field(
        default=None,
        description="Gaussian smoothing of the distance field in physical units. None spans 3 voxel edges per axis, 0 disables",
    )
    level: float = Field(default=0.0, description="Distance of the surface from the label boundary; positive is outside")
    smoothing_iterations: int = Field(
        default=DEFAULT_SMOOTHING_ITERATIONS, ge=0, description="Windowed sinc smoothing iterations on the surface before slicing; 0 disables"
    )
    smoothing_pass_band: float = Field(
        default=DEFAULT_SMOOTHING_PASS_BAND, gt=0, le=2, description="Windowed sinc pass band in (0, 2]; lower smooths and shrinks more"
    )


class MeniscusBrep(SurfaceBrep):
    """Settings for building the lateral and medial meniscus B-reps, with the defaults of each step."""

    spacing: float = Field(default=5.0, description="Arc-length distance between slices along the centerline, in physical units")
    points_per_slice: int = Field(default=32, description="Points on each slice contour. Use about 10 for hexahedral blocks")
    spline_order: int = Field(default=3, description="Polynomial degree of the B-spline geometry")
    cap: Literal["none", "planar", "smooth"] = Field(default="planar", description="How to close the ends; ignored for hexahedral blocks")
    cap_offset: OptionalFloat = Field(
        default=None, description="Apex offset of smooth caps in physical units. None is a quarter of the equivalent radius"
    )
    hex_blocks: bool = Field(default=False, description="Build the five-block butterfly solid with planar caps that hexahedral meshing needs")


class CartilageBrep(SurfaceBrep):
    """Settings for building the B-reps of the lateral and medial tibial and the patellar cartilage, with the defaults of each step.

    Each is a sheet over the plane normal to its thinnest direction, between
    its articular side and the side against its bone, within a smooth
    perimeter. Defaults left as None depend on hex_blocks.
    """

    side_cosine: float = Field(
        default=DEFAULT_SIDE_COSINE,
        gt=0,
        lt=1,
        description="Cosine between a surface normal and the direction away from the bone beyond which a point is on the articular "
        + "or bone side. Higher values cut off more of the steep rim between them",
    )
    perimeter_radius: OptionalFloat = Field(
        default=None,
        gt=0,
        description="Smallest radius of curvature of the perimeter and hole outlines, in physical units. None is "
        + f"{DEFAULT_PERIMETER_RADIUS_HEXAHEDRAL} for hex_blocks, enough for hexahedra to follow the real lateral tibial cartilage's "
        + f"notch, and {DEFAULT_PERIMETER_RADIUS_TETRAHEDRAL} otherwise",
    )
    perimeter_inset: float = Field(
        default=0.0, ge=0, description="How far inside the overlap of the two sides to put the perimeter, in physical units"
    )
    min_thickness: OptionalFloat = Field(
        default=None,
        gt=0,
        lt=1,
        description="Minimum thickness as a fraction of the mean; thinner regions are thickened by lowering the bone side. None is "
        + f"{DEFAULT_MIN_THICKNESS_HEXAHEDRAL} for hex_blocks and {DEFAULT_MIN_THICKNESS_TETRAHEDRAL} otherwise",
    )
    hole_thickness: float = Field(
        default=DEFAULT_HOLE_THICKNESS,
        ge=0,
        lt=1,
        description="Thickness as a fraction of the mean below which the sheet gets a hole, as do regions with no cartilage. "
        + "Ignored for hex_blocks, which are only thickened; 0 makes no holes",
    )
    hex_blocks: bool = Field(default=False, description="Build the five-block butterfly solid that hexahedral meshing needs")

    @model_validator(mode="after")
    def _check_thicknesses(self) -> Self:
        if not self.hex_blocks and self.hole_thickness >= self.resolved_min_thickness:
            raise ValueError(f"hole_thickness {self.hole_thickness} must be below min_thickness {self.resolved_min_thickness}")
        return self

    @property
    def resolved_perimeter_radius(self) -> float:
        """perimeter_radius, or its default for hex_blocks."""
        if self.perimeter_radius is not None:
            return self.perimeter_radius
        return DEFAULT_PERIMETER_RADIUS_HEXAHEDRAL if self.hex_blocks else DEFAULT_PERIMETER_RADIUS_TETRAHEDRAL

    @property
    def resolved_min_thickness(self) -> float:
        """min_thickness, or its default for hex_blocks."""
        if self.min_thickness is not None:
            return self.min_thickness
        return DEFAULT_MIN_THICKNESS_HEXAHEDRAL if self.hex_blocks else DEFAULT_MIN_THICKNESS_TETRAHEDRAL

    @property
    def resolved_hole_thickness(self) -> float:
        """hole_thickness, or 0 for hex_blocks."""
        return 0.0 if self.hex_blocks else self.hole_thickness


class StructureMesh(BaseConfig):
    """Settings shared by every structure for tetrahedral meshing and for optimizing hexahedral meshes."""

    min_size: OptionalFloat = Field(
        default=None,
        description="Smallest tetrahedron size in physical units. None is 1/20 of the shortest side of the B-rep's oriented bounding box, "
        + "but no more than max_size. 0 sets no limit",
    )
    max_size: OptionalFloat = Field(
        default=None,
        description="Largest tetrahedron size in physical units. None is 1/4 of the shortest side of the B-rep's oriented bounding box, "
        + "but no less than min_size",
    )
    elements_per_circle: int = Field(default=0, description="Tetrahedra per circle of surface curvature; 0 disables curvature sizing")
    algorithm_2d: Algorithm2D = Field(default=Algorithm2D.FRONTAL_DELAUNAY, description="gmsh surface algorithm for tetrahedral meshing")
    algorithm_3d: Algorithm3D = Field(default=Algorithm3D.DELAUNAY, description="gmsh volume algorithm for tetrahedral meshing")
    optimize: bool = Field(default=False, description="Move hexahedral mesh nodes to raise the elements' scaled Jacobians after meshing")
    laplacian_iterations: int = Field(default=DEFAULT_LAPLACIAN_ITERATIONS, ge=0, description="Smart Laplacian sweeps when optimizing")
    optimization_iterations: int = Field(
        default=DEFAULT_OPTIMIZATION_ITERATIONS, ge=0, description="Gradient steps minimizing the sum of 1 / scaled Jacobian when optimizing"
    )
    surface_deviation: float = Field(
        default=DEFAULT_SURFACE_DEVIATION,
        ge=0,
        description="How far boundary nodes may move off the surface when optimizing, as a fraction of the local element size",
    )
    surface_travel: float = Field(
        default=DEFAULT_SURFACE_TRAVEL,
        ge=0,
        description="How far boundary nodes may slide along the surface when optimizing, as a fraction of the local element size",
    )
    feature_angle: float = Field(
        default=DEFAULT_FEATURE_ANGLE,
        gt=0,
        lt=180,
        description="Angle in degrees between boundary faces above which their shared edge is kept as a feature when optimizing",
    )
    preserve_volume: bool = Field(default=True, description="Keep the mesh's volume when optimizing by cancelling the net outward surface offset")
    aspect_weight: float = Field(
        default=0.0,
        ge=0,
        description="Weight of the elements' squared Frobenius aspect ratios against 1 / scaled Jacobian when optimizing. "
        + "0 optimizes the scaled Jacobian alone, which can stretch elements",
    )


class MeniscusMesh(StructureMesh):
    """Settings for meshing the lateral and medial meniscus B-reps, with the defaults of each meshing function.

    Hexahedral element types need B-reps built with hex_blocks.
    """

    elements_per_edge: int = Field(default=4, description="Hexahedra along each edge of a cap")
    elements_along: int = Field(default=20, description="Hexahedra along the solid between the caps")


DEFAULT_CARTILAGE_ASPECT_WEIGHT = 0.5
"""Default weight of the squared Frobenius aspect ratios when optimizing cartilage hexahedra."""


class CartilageMesh(StructureMesh):
    """Settings for meshing the B-reps of the lateral and medial tibial and the patellar cartilage.

    Hexahedral element types need B-reps built with hex_blocks.
    """

    elements_around: int = Field(default=8, description="Hexahedra along each quarter of the perimeter and each side of the inner block")
    elements_radial: int = Field(default=4, description="Hexahedra along each spoke from the perimeter to the inner block")
    elements_through: int = Field(default=3, description="Hexahedra through the thickness")
    aspect_weight: float = Field(
        default=DEFAULT_CARTILAGE_ASPECT_WEIGHT,
        ge=0,
        description="Weight of the elements' squared Frobenius aspect ratios against 1 / scaled Jacobian when optimizing. "
        + "0 optimizes the scaled Jacobian alone, which stretched the real cartilage's elements up to 75 times their thickness",
    )


class LabelMapBrep(BaseConfig):
    """Settings for building a B-rep of every structure in a segmented LabelMap.

    The segmentation is either one multilabel LabelMap, label_map, with labels
    naming its values, or several single-label LabelMaps, label_maps, that are
    assembled onto one grid. Give one or the other. Each structure named in
    labels or label_maps is built by its own pipeline, with the settings of
    its section such as meniscus, and written to
    <output_directory>/<structure><output>.
    """

    label_map: OptionalPath = Field(default=None, description="Multilabel LabelMap to read: .nii, .nrrd, or .mhd")
    labels: LabelValues = Field(
        default=None,
        description='Label value of each structure in label_map by name, as JSON such as {"tibia": 1, "medial_tibial_cartilage": 2}',
    )
    label_maps: StructurePaths = _label_maps_field("Single-label LabelMaps to assemble instead of label_map")
    interpolator: Interpolator = _interpolator_field()
    padding: int = _padding_field()
    assembled_label_map: OptionalPath = Field(
        default=None,
        description="Where to write the multilabel LabelMap assembled from label_maps, with its labels in <stem>_labels.json beside it. "
        + "None does not write it",
    )
    output: str = Field(
        default=".brep", description="Suffix of each B-rep file, appended to its structure's name, such as _hex.brep for lateral_meniscus_hex.brep"
    )
    output_directory: Path = _output_directory_field()
    meniscus: MeniscusBrep = Field(default_factory=MeniscusBrep, description="Settings for the lateral and medial menisci")
    cartilage: CartilageBrep = Field(
        default_factory=CartilageBrep, description="Settings for the lateral and medial tibial and the patellar cartilage"
    )

    @model_validator(mode="after")
    def _check_sources(self) -> Self:
        if self.label_map is not None and self.label_maps is not None:
            raise ValueError("Give label_map or label_maps, not both")
        if self.labels is not None and self.label_maps is not None:
            raise ValueError("labels names the values in label_map; label_maps are named by their keys")
        if self.label_map is not None and self.labels is None:
            raise ValueError("Give labels naming the structures in label_map")
        if self.assembled_label_map is not None and self.label_maps is None:
            raise ValueError("assembled_label_map is written only when assembling label_maps")
        structures = self.structures
        for structure in structures:
            bone = BONES.get(structure)
            if bone is not None and bone not in structures:
                raise ValueError(f"{structure} needs its bone, {bone}, in the segmentation; got {list(structures)}")
        _check_suffix(self.output, BREP_SUFFIXES)
        return self

    @property
    def structures(self) -> tuple[Structure, ...]:
        """The structures named in labels or label_maps, in order, or none if neither is given."""
        if self.labels is not None:
            return tuple(self.labels)
        return tuple(self.label_maps) if self.label_maps is not None else ()

    def output_path(self, structure: Structure) -> Path:
        """The B-rep file to write for structure."""
        return self.output_directory / f"{structure}{self.output}"


# Each element type's shape, polynomial order, and whether high-order elements omit face and interior nodes.
ELEMENT_TYPES: dict[ElementType, tuple[Literal["tetrahedral", "hexahedral"], int, bool]] = {
    "hex8": ("hexahedral", 1, False),
    "hex20": ("hexahedral", 2, True),
    "hex27": ("hexahedral", 2, False),
    "tet4": ("tetrahedral", 1, False),
    "tet10": ("tetrahedral", 2, False),
}


class BrepMesh(BaseConfig):
    """Settings for meshing the B-rep of each structure.

    Each structure in breps is meshed by its own pipeline, with the settings
    of its section such as meniscus, and written to
    <output_directory>/<structure><output>.
    """

    breps: StructurePaths = Field(
        default=None,
        description='B-rep file of each structure to mesh, by structure name, as JSON such as {"medial_meniscus": "medial_meniscus.brep"}',
    )
    element_type: ElementType = Field(
        default="tet4", description="Element shape and node count: hex8, hex20, or hex27 hexahedra, or tet4 or tet10 tetrahedra"
    )
    output: str = Field(
        default="_{element_type}.msh",
        description="Suffix of each mesh file, appended to its structure's name, where {element_type} is replaced by the element type. "
        + "gmsh infers the format from the extension",
    )
    output_directory: Path = _output_directory_field()
    meniscus: MeniscusMesh = Field(default_factory=MeniscusMesh, description="Settings for the lateral and medial menisci")
    cartilage: CartilageMesh = Field(
        default_factory=CartilageMesh, description="Settings for the lateral and medial tibial and the patellar cartilage"
    )

    @model_validator(mode="after")
    def _check_mesh(self) -> Self:
        try:
            suffix = self.output_suffix
        except (KeyError, IndexError, ValueError) as error:
            raise ValueError(f"The output suffix may hold only the {{element_type}} placeholder, got {self.output!r}") from error
        _check_suffix(suffix)
        if (self.meniscus.optimize or self.cartilage.optimize) and self.element_shape != "hexahedral":
            raise ValueError("optimize applies only to hexahedral meshes")
        return self

    @property
    def element_shape(self) -> Literal["tetrahedral", "hexahedral"]:
        """The shape of element_type."""
        return ELEMENT_TYPES[self.element_type][0]

    @property
    def element_order(self) -> int:
        """The polynomial order of element_type."""
        return ELEMENT_TYPES[self.element_type][1]

    @property
    def second_order_incomplete(self) -> bool:
        """Whether element_type omits face and interior nodes."""
        return ELEMENT_TYPES[self.element_type][2]

    @property
    def output_suffix(self) -> str:
        """output with its {element_type} placeholder replaced."""
        return self.output.format(element_type=self.element_type)

    def output_path(self, structure: Structure) -> Path:
        """The mesh file to write for structure."""
        return self.output_directory / f"{structure}{self.output_suffix}"
