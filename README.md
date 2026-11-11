# meniscus2brep

[![PyPI](https://img.shields.io/pypi/v/meniscus2brep)](https://pypi.org/project/meniscus2brep/)
[![Coverage](https://img.shields.io/endpoint?url=https://comporthobiomech.github.io/meniscus2brep/coverage.json)](https://github.com/CompOrthoBiomech/meniscus2brep/actions/workflows/release.yml)
[![Docs](https://img.shields.io/badge/docs-gh--pages-blue)](https://comporthobiomech.github.io/meniscus2brep/)

Generate B-rep geometry of a meniscus from a segmentation LabelMap, and mesh it with high-quality elements:

- **B-rep from segmentation labels.** Reads a LabelMap (`.nii`, `.nrrd`, or `.mhd`), fits a centerline through the labeled meniscus, slices its smoothed surface into cross-section contours, and lofts them into a solid B-spline B-rep (`.brep`) with planar or smooth end caps.
- **Tetrahedral meshing** with linear 4-node or quadratic 10-node tetrahedra.
- **Hexahedral meshing** with block-structured 8-node, 20-node, or 27-node hexahedra. The solid is split into a butterfly (O-grid) of five blocks, so elements stay well shaped at the meniscus's sharp corners.

Meshes are written in any format gmsh infers from the file extension, such as `.msh`, `.vtk`, or `.inp`.

**Documentation:** <https://comporthobiomech.github.io/meniscus2brep/>

## Installation

```sh
pip install meniscus2brep
```

## Usage

Two command-line pipelines take a JSON configuration, `--option value` overrides, or both. Run either with `--help` to list every option and its default.

```sh
meniscus2brep-brep --label-map knee.nii --output meniscus.brep --cap planar
meniscus2brep-mesh --brep meniscus.brep --output meniscus.msh --max-size 1.0
```

Each run saves the configuration it used, so it can be repeated with `--config`. Without `--config` it is saved as `settings<timestamp>.json` in the working directory. With `--config` and overrides, the changed configuration is saved beside the original as `<name>_modified.json`.

### Element types

| Elements | B-rep options | Mesh options |
|---|---|---|
| 4-node tetrahedra | | `--element tetrahedral` (the default) |
| 10-node tetrahedra | | `--element tetrahedral --element-order 2` |
| 8-node hexahedra | `--hex-blocks true --points-per-slice 10` | `--element hexahedral` |
| 20-node hexahedra | `--hex-blocks true --points-per-slice 10` | `--element hexahedral --element-order 2 --second-order-incomplete true` |
| 27-node hexahedra | `--hex-blocks true --points-per-slice 10` | `--element hexahedral --element-order 2` |

## Development

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run pytest
```

Build the documentation into `docs/_build/html`:

```sh
uv run --group docs sphinx-build -W -b html docs docs/_build/html
```

## License

Licensed under the [Apache License 2.0](LICENSE).
