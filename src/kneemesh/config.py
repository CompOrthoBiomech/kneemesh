"""Configuration models and parsing them from JSON."""

import json
from os import PathLike
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from kneemesh.segmentation import DEFAULT_PADDING_VOXELS, Interpolator
from kneemesh.tet_meshing import Algorithm2D, Algorithm3D
from kneemesh.vol import DEFAULT_SMOOTHING_ITERATIONS, DEFAULT_SMOOTHING_PASS_BAND


class BaseConfig(BaseModel):
    """Base class for configuration models.

    Unknown fields are rejected so that misspelled options fail at load time
    instead of being silently ignored.
    """

    model_config = ConfigDict(extra="forbid")

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


OptionalFloat = Annotated[float | None, BeforeValidator(_none_from_text)]
OptionalPath = Annotated[Path | None, BeforeValidator(_none_from_text)]
LabelValues = Annotated[dict[str, int] | None, BeforeValidator(_json_from_text)]
LabelPaths = Annotated[dict[str, Path] | None, BeforeValidator(_json_from_text)]


class LabelMapBrep(BaseConfig):
    """Settings every B-rep pipeline shares: the segmentation to read, the structure to build, and the file to write.

    The segmentation is either one multilabel LabelMap, label_map, with labels
    naming its values, or several single-label LabelMaps, label_maps, that are
    assembled onto one grid. Give one or the other.
    """

    label_map: OptionalPath = Field(default=None, description="Multilabel LabelMap to read: .nii, .nrrd, or .mhd")
    labels: LabelValues = Field(
        default=None,
        description='Label value of each structure in label_map by name, as JSON such as {"tibia": 1, "medial_tibial_cartilage": 2}. '
        "None reads label_map as holding only this structure, with value 1",
    )
    label_maps: LabelPaths = Field(
        default=None,
        description='Single-label LabelMaps to assemble instead of label_map, by structure name, as JSON such as {"tibia": "tibia.nii"}. '
        "Each file's nonzero voxels are its structure",
    )
    interpolator: Interpolator = Field(
        default="label_gaussian", description="Resampling interpolator for assembling label_maps: label_gaussian or nearest_neighbor"
    )
    padding: int = Field(default=DEFAULT_PADDING_VOXELS, ge=0, description="Background voxels around the assembled label_maps on each side")
    name: str = Field(default="structure", description="Name of the structure to build, a key of labels or label_maps")
    output: Path = Field(default=Path("structure.brep"), description="B-rep file to write")

    @model_validator(mode="after")
    def _check_sources(self) -> Self:
        if self.label_map is not None and self.label_maps is not None:
            raise ValueError("Give label_map or label_maps, not both")
        if self.labels is not None and self.label_maps is not None:
            raise ValueError("labels names the values in label_map; label_maps are named by their keys")
        self.require_label(self.name)
        return self

    def require_label(self, name: str) -> None:
        """Raise ValueError if labels or label_maps is given and does not name name."""
        names = self.labels if self.labels is not None else self.label_maps
        if names is not None and name not in names:
            raise ValueError(f"No label named {name!r}; got {sorted(names)}")


class CartilageBrep(LabelMapBrep):
    """Settings every cartilage B-rep pipeline shares, adding the underlying bone that orients the cartilage."""

    bone: str = Field(default="bone", description="Name of the bone under the cartilage, a key of labels or label_maps")

    @model_validator(mode="after")
    def _check_bone(self) -> Self:
        if self.bone == self.name:
            raise ValueError(f"The cartilage and its bone need different labels, got {self.name!r} for both")
        self.require_label(self.bone)
        return self


class BrepMesh(BaseConfig):
    """Settings every meshing pipeline shares: the B-rep to read, the mesh file to write, and the element order."""

    brep: Path = Field(default=Path("structure.brep"), description="B-rep file to read")
    output: Path = Field(default=Path("structure.msh"), description="Mesh file to write; gmsh infers the format from the extension")
    element_order: int = Field(default=1, description="Polynomial order of the elements")
    second_order_incomplete: bool = Field(default=False, description="Omit face and interior nodes from high-order elements")


class MeniscusBrep(LabelMapBrep):
    """Settings for building a meniscus B-rep from a segmented LabelMap, with the defaults of each step."""

    name: str = Field(default="meniscus", description="Name of the structure to build, a key of labels or label_maps")
    output: Path = Field(default=Path("meniscus.brep"), description="B-rep file to write")
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


class MeniscusMesh(BrepMesh):
    """Settings for meshing a meniscus B-rep, with the defaults of each meshing function."""

    brep: Path = Field(default=Path("meniscus.brep"), description="B-rep file to read")
    output: Path = Field(default=Path("meniscus.msh"), description="Mesh file to write; gmsh infers the format from the extension")
    element: Literal["tetrahedral", "hexahedral"] = Field(
        default="tetrahedral", description="Element shape. Hexahedral needs a B-rep built with hex_blocks"
    )
    min_size: float = Field(default=0.0, description="Smallest tetrahedron size in physical units")
    max_size: OptionalFloat = Field(default=None, description="Largest tetrahedron size in physical units. None lets gmsh choose")
    elements_per_circle: int = Field(default=0, description="Tetrahedra per circle of surface curvature; 0 disables curvature sizing")
    algorithm_2d: Algorithm2D = Field(default=Algorithm2D.FRONTAL_DELAUNAY, description="gmsh surface algorithm for tetrahedral meshing")
    algorithm_3d: Algorithm3D = Field(default=Algorithm3D.DELAUNAY, description="gmsh volume algorithm for tetrahedral meshing")
    elements_per_edge: int = Field(default=4, description="Hexahedra along each edge of a cap")
    elements_along: int = Field(default=20, description="Hexahedra along the solid between the caps")


class TibialCartilageBrep(CartilageBrep):
    """Settings for building a tibial cartilage B-rep from a segmented LabelMap. Not implemented yet."""

    name: str = Field(
        default="tibial_cartilage",
        description="Name of the cartilage plate's label, such as lateral_tibial_cartilage. Both plates share one B-rep and meshing strategy",
    )
    bone: str = Field(default="tibia", description="Name of the bone under the cartilage, a key of labels or label_maps")
    output: Path = Field(default=Path("tibial_cartilage.brep"), description="B-rep file to write")


class TibialCartilageMesh(BrepMesh):
    """Settings for meshing a tibial cartilage B-rep. Not implemented yet."""

    brep: Path = Field(default=Path("tibial_cartilage.brep"), description="B-rep file to read")
    output: Path = Field(default=Path("tibial_cartilage.msh"), description="Mesh file to write; gmsh infers the format from the extension")
    name: str = Field(
        default="tibial_cartilage",
        description="Name of the cartilage plate, such as lateral or medial. Both plates share one B-rep and meshing strategy",
    )


class FemoralCartilageBrep(CartilageBrep):
    """Settings for building a femoral cartilage B-rep from a segmented LabelMap. Not implemented yet."""

    name: str = Field(default="femoral_cartilage", description="Name of the structure to build, a key of labels or label_maps")
    bone: str = Field(default="femur", description="Name of the bone under the cartilage, a key of labels or label_maps")
    output: Path = Field(default=Path("femoral_cartilage.brep"), description="B-rep file to write")


class FemoralCartilageMesh(BrepMesh):
    """Settings for meshing a femoral cartilage B-rep. Not implemented yet."""

    brep: Path = Field(default=Path("femoral_cartilage.brep"), description="B-rep file to read")
    output: Path = Field(default=Path("femoral_cartilage.msh"), description="Mesh file to write; gmsh infers the format from the extension")


class PatellarCartilageBrep(CartilageBrep):
    """Settings for building a patellar cartilage B-rep from a segmented LabelMap. Not implemented yet."""

    name: str = Field(default="patellar_cartilage", description="Name of the structure to build, a key of labels or label_maps")
    bone: str = Field(default="patella", description="Name of the bone under the cartilage, a key of labels or label_maps")
    output: Path = Field(default=Path("patellar_cartilage.brep"), description="B-rep file to write")


class PatellarCartilageMesh(BrepMesh):
    """Settings for meshing a patellar cartilage B-rep. Not implemented yet."""

    brep: Path = Field(default=Path("patellar_cartilage.brep"), description="B-rep file to read")
    output: Path = Field(default=Path("patellar_cartilage.msh"), description="Mesh file to write; gmsh infers the format from the extension")


class TibiaBrep(LabelMapBrep):
    """Settings for building a tibia B-rep from a segmented LabelMap. Not implemented yet."""

    name: str = Field(default="tibia", description="Name of the structure to build, a key of labels or label_maps")
    output: Path = Field(default=Path("tibia.brep"), description="B-rep file to write")


class TibiaMesh(BrepMesh):
    """Settings for meshing a tibia B-rep. Not implemented yet."""

    brep: Path = Field(default=Path("tibia.brep"), description="B-rep file to read")
    output: Path = Field(default=Path("tibia.msh"), description="Mesh file to write; gmsh infers the format from the extension")


class FemurBrep(LabelMapBrep):
    """Settings for building a femur B-rep from a segmented LabelMap. Not implemented yet."""

    name: str = Field(default="femur", description="Name of the structure to build, a key of labels or label_maps")
    output: Path = Field(default=Path("femur.brep"), description="B-rep file to write")


class FemurMesh(BrepMesh):
    """Settings for meshing a femur B-rep. Not implemented yet."""

    brep: Path = Field(default=Path("femur.brep"), description="B-rep file to read")
    output: Path = Field(default=Path("femur.msh"), description="Mesh file to write; gmsh infers the format from the extension")


class PatellaBrep(LabelMapBrep):
    """Settings for building a patella B-rep from a segmented LabelMap. Not implemented yet."""

    name: str = Field(default="patella", description="Name of the structure to build, a key of labels or label_maps")
    output: Path = Field(default=Path("patella.brep"), description="B-rep file to write")


class PatellaMesh(BrepMesh):
    """Settings for meshing a patella B-rep. Not implemented yet."""

    brep: Path = Field(default=Path("patella.brep"), description="B-rep file to read")
    output: Path = Field(default=Path("patella.msh"), description="Mesh file to write; gmsh infers the format from the extension")
