"""Configuration models and parsing them from JSON."""

from os import PathLike
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from meniscus2brep.mesh import Algorithm2D, Algorithm3D
from meniscus2brep.vol import DEFAULT_SMOOTHING_ITERATIONS, DEFAULT_SMOOTHING_PASS_BAND


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


class BrepConfig(BaseConfig):
    """Settings for building a B-rep from a segmented LabelMap, with the defaults of each step."""

    label_map: Path = Field(default=Path("label_map.nii"), description="LabelMap to read: .nii, .nrrd, or .mhd")
    output: Path = Field(default=Path("meniscus.brep"), description="B-rep file to write")
    label: int = Field(default=1, description="Value of the meniscus in the LabelMap")
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


class MeshConfig(BaseConfig):
    """Settings for meshing a B-rep, with the defaults of each meshing function."""

    brep: Path = Field(default=Path("meniscus.brep"), description="B-rep file to read")
    output: Path = Field(default=Path("meniscus.msh"), description="Mesh file to write; gmsh infers the format from the extension")
    element: Literal["tetrahedral", "hexahedral"] = Field(
        default="tetrahedral", description="Element shape. Hexahedral needs a B-rep built with hex_blocks"
    )
    element_order: int = Field(default=1, description="Polynomial order of the elements")
    second_order_incomplete: bool = Field(default=False, description="Omit face and interior nodes from high-order elements")
    min_size: float = Field(default=0.0, description="Smallest tetrahedron size in physical units")
    max_size: OptionalFloat = Field(default=None, description="Largest tetrahedron size in physical units. None lets gmsh choose")
    elements_per_circle: int = Field(default=0, description="Tetrahedra per circle of surface curvature; 0 disables curvature sizing")
    algorithm_2d: Algorithm2D = Field(default=Algorithm2D.FRONTAL_DELAUNAY, description="gmsh surface algorithm for tetrahedral meshing")
    algorithm_3d: Algorithm3D = Field(default=Algorithm3D.DELAUNAY, description="gmsh volume algorithm for tetrahedral meshing")
    elements_per_edge: int = Field(default=4, description="Hexahedra along each edge of a cap")
    elements_along: int = Field(default=20, description="Hexahedra along the solid between the caps")
