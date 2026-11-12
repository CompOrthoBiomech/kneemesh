"""Configuration models and parsing them from JSON."""

from os import PathLike
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

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


OptionalFloat = Annotated[float | None, BeforeValidator(_none_from_text)]


class LabelMapBrep(BaseConfig):
    """Settings every B-rep pipeline shares: the LabelMap to read, the structure's label in it, and the file to write."""

    label_map: Path = Field(default=Path("label_map.nii"), description="LabelMap to read: .nii, .nrrd, or .mhd")
    output: Path = Field(default=Path("structure.brep"), description="B-rep file to write")
    label: int = Field(default=1, description="Value of the structure in the LabelMap")


class BrepMesh(BaseConfig):
    """Settings every meshing pipeline shares: the B-rep to read, the mesh file to write, and the element order."""

    brep: Path = Field(default=Path("structure.brep"), description="B-rep file to read")
    output: Path = Field(default=Path("structure.msh"), description="Mesh file to write; gmsh infers the format from the extension")
    element_order: int = Field(default=1, description="Polynomial order of the elements")
    second_order_incomplete: bool = Field(default=False, description="Omit face and interior nodes from high-order elements")


class MeniscusBrep(LabelMapBrep):
    """Settings for building a meniscus B-rep from a segmented LabelMap, with the defaults of each step."""

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


class TibialCartilageBrep(LabelMapBrep):
    """Settings for building a tibial cartilage B-rep from a segmented LabelMap. Not implemented yet."""

    output: Path = Field(default=Path("tibial_cartilage.brep"), description="B-rep file to write")
    name: str = Field(
        default="tibial_cartilage",
        description="Name of the cartilage plate, such as lateral or medial. Both plates share one B-rep and meshing strategy",
    )


class TibialCartilageMesh(BrepMesh):
    """Settings for meshing a tibial cartilage B-rep. Not implemented yet."""

    brep: Path = Field(default=Path("tibial_cartilage.brep"), description="B-rep file to read")
    output: Path = Field(default=Path("tibial_cartilage.msh"), description="Mesh file to write; gmsh infers the format from the extension")
    name: str = Field(
        default="tibial_cartilage",
        description="Name of the cartilage plate, such as lateral or medial. Both plates share one B-rep and meshing strategy",
    )


class FemoralCartilageBrep(LabelMapBrep):
    """Settings for building a femoral cartilage B-rep from a segmented LabelMap. Not implemented yet."""

    output: Path = Field(default=Path("femoral_cartilage.brep"), description="B-rep file to write")


class FemoralCartilageMesh(BrepMesh):
    """Settings for meshing a femoral cartilage B-rep. Not implemented yet."""

    brep: Path = Field(default=Path("femoral_cartilage.brep"), description="B-rep file to read")
    output: Path = Field(default=Path("femoral_cartilage.msh"), description="Mesh file to write; gmsh infers the format from the extension")


class PatellarCartilageBrep(LabelMapBrep):
    """Settings for building a patellar cartilage B-rep from a segmented LabelMap. Not implemented yet."""

    output: Path = Field(default=Path("patellar_cartilage.brep"), description="B-rep file to write")


class PatellarCartilageMesh(BrepMesh):
    """Settings for meshing a patellar cartilage B-rep. Not implemented yet."""

    brep: Path = Field(default=Path("patellar_cartilage.brep"), description="B-rep file to read")
    output: Path = Field(default=Path("patellar_cartilage.msh"), description="Mesh file to write; gmsh infers the format from the extension")


class TibiaBrep(LabelMapBrep):
    """Settings for building a tibia B-rep from a segmented LabelMap. Not implemented yet."""

    output: Path = Field(default=Path("tibia.brep"), description="B-rep file to write")


class TibiaMesh(BrepMesh):
    """Settings for meshing a tibia B-rep. Not implemented yet."""

    brep: Path = Field(default=Path("tibia.brep"), description="B-rep file to read")
    output: Path = Field(default=Path("tibia.msh"), description="Mesh file to write; gmsh infers the format from the extension")


class FemurBrep(LabelMapBrep):
    """Settings for building a femur B-rep from a segmented LabelMap. Not implemented yet."""

    output: Path = Field(default=Path("femur.brep"), description="B-rep file to write")


class FemurMesh(BrepMesh):
    """Settings for meshing a femur B-rep. Not implemented yet."""

    brep: Path = Field(default=Path("femur.brep"), description="B-rep file to read")
    output: Path = Field(default=Path("femur.msh"), description="Mesh file to write; gmsh infers the format from the extension")


class PatellaBrep(LabelMapBrep):
    """Settings for building a patella B-rep from a segmented LabelMap. Not implemented yet."""

    output: Path = Field(default=Path("patella.brep"), description="B-rep file to write")


class PatellaMesh(BrepMesh):
    """Settings for meshing a patella B-rep. Not implemented yet."""

    brep: Path = Field(default=Path("patella.brep"), description="B-rep file to read")
    output: Path = Field(default=Path("patella.msh"), description="Mesh file to write; gmsh infers the format from the extension")
