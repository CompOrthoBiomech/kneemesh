"""Configuration models and parsing them from JSON."""

import json
from os import PathLike
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Self, get_args

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from kneemesh.disk import BREP_SUFFIXES
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


class MeniscusBrep(BaseConfig):
    """Settings for building the lateral and medial meniscus B-reps, with the defaults of each step."""

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
    spacing: float = Field(default=5.0, description="Arc-length distance between slices along the centerline, in physical units")
    points_per_slice: int = Field(default=32, description="Points on each slice contour. Use about 10 for hexahedral blocks")
    spline_order: int = Field(default=3, description="Polynomial degree of the B-spline geometry")
    cap: Literal["none", "planar", "smooth"] = Field(default="planar", description="How to close the ends; ignored for hexahedral blocks")
    cap_offset: OptionalFloat = Field(
        default=None, description="Apex offset of smooth caps in physical units. None is a quarter of the equivalent radius"
    )
    hex_blocks: bool = Field(default=False, description="Build the five-block butterfly solid with planar caps that hexahedral meshing needs")


class MeniscusMesh(BaseConfig):
    """Settings for meshing the lateral and medial meniscus B-reps, with the defaults of each meshing function.

    Hexahedral element types need B-reps built with hex_blocks.
    """

    min_size: OptionalFloat = Field(
        default=None,
        description="Smallest tetrahedron size in physical units. None is 1/20 of the longest side of the B-rep's bounding box, "
        + "but no more than max_size. 0 sets no limit",
    )
    max_size: OptionalFloat = Field(default=None, description="Largest tetrahedron size in physical units. None lets gmsh choose")
    elements_per_circle: int = Field(default=0, description="Tetrahedra per circle of surface curvature; 0 disables curvature sizing")
    algorithm_2d: Algorithm2D = Field(default=Algorithm2D.FRONTAL_DELAUNAY, description="gmsh surface algorithm for tetrahedral meshing")
    algorithm_3d: Algorithm3D = Field(default=Algorithm3D.DELAUNAY, description="gmsh volume algorithm for tetrahedral meshing")
    elements_per_edge: int = Field(default=4, description="Hexahedra along each edge of a cap")
    elements_along: int = Field(default=20, description="Hexahedra along the solid between the caps")
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
        default=None, description='B-rep file of each structure to mesh, by structure name, as JSON such as {"medial_meniscus": "medial_meniscus.brep"}'
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

    @model_validator(mode="after")
    def _check_mesh(self) -> Self:
        try:
            suffix = self.output_suffix
        except (KeyError, IndexError, ValueError) as error:
            raise ValueError(f"The output suffix may hold only the {{element_type}} placeholder, got {self.output!r}") from error
        _check_suffix(suffix)
        if self.meniscus.optimize and self.element_shape != "hexahedral":
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
