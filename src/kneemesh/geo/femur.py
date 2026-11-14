"""B-rep generation for the femoral cartilage. The femur is not implemented yet.

The femoral cartilage wraps the condyles, so it is a sheet over a cylinder
about the knee's flexion axis rather than over a plane (CylinderFrame). Its
domain, unrolled, is a U: the trochlea joins two condylar legs around the
intercondylar notch. Its solid is built like the other sheets' (geo.sheet.sheet_to_brep).
"""

from kneemesh.cartilage import CartilageSides
from kneemesh.geo.sheet import (
    DEFAULT_MIN_THICKNESS_HEXAHEDRAL,
    DEFAULT_PERIMETER_RADIUS_HEXAHEDRAL,
    CartilageSheet,
    CylinderFrame,
    fit_sheet,
)


def fit_femoral_sheet(
    sides: CartilageSides,
    perimeter_radius: float = DEFAULT_PERIMETER_RADIUS_HEXAHEDRAL,
    perimeter_inset: float = 0.0,
    min_thickness: float = DEFAULT_MIN_THICKNESS_HEXAHEDRAL,
    hole_thickness: float = 0.0,
) -> CartilageSheet:
    """Fit the femoral cartilage as a sheet over the cylinder its bone side wraps.

    The arguments are fit_sheet's; see there. Heights are distances from the
    cylinder's axis, and the domain is the cylinder unrolled.

    Raises:
        ValueError: If a parameter is out of range, the sides wrap too far
            around to unroll, or they do not overlap enough.
    """
    return fit_sheet(sides, perimeter_radius, perimeter_inset, min_thickness, hole_thickness, CylinderFrame.fit(sides))
