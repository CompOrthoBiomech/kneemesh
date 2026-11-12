# kneemesh

[![PyPI](https://img.shields.io/pypi/v/kneemesh)](https://pypi.org/project/kneemesh/)
[![Coverage](https://img.shields.io/endpoint?url=https://comporthobiomech.github.io/kneemesh/coverage.json)](https://github.com/CompOrthoBiomech/kneemesh/actions/workflows/release.yml)
[![Docs](https://img.shields.io/badge/docs-gh--pages-blue)](https://comporthobiomech.github.io/kneemesh/)

Generate B-rep geometry of knee structures from a segmentation LabelMap, and mesh it with high-quality elements. The meniscus is supported now; lateral and medial tibial, femoral, and patellar cartilage and the bones are planned. For the meniscus:

- **B-rep from segmentation labels.** Reads a LabelMap (`.nii`, `.nrrd`, or `.mhd`), fits a centerline through the labeled meniscus, slices its smoothed surface into cross-section contours, and lofts them into a solid B-spline B-rep (`.brep`) with planar or smooth end caps.
- **Tetrahedral meshing** with linear 4-node or quadratic 10-node tetrahedra.
- **Hexahedral meshing** with block-structured 8-node, 20-node, or 27-node hexahedra. The solid is split into a butterfly (O-grid) of five blocks, so elements stay well shaped at the meniscus's sharp corners.

Meshes are written in any format gmsh infers from the file extension, such as `.msh`, `.vtk`, or `.inp`.

**Documentation:** <https://comporthobiomech.github.io/kneemesh/>

## Installation

```sh
pip install kneemesh
```

## Usage

Two command-line pipelines take the structure as their first argument, then a JSON configuration, `--option value` overrides, or both. Run `kneemesh-brep meniscus --help` to list every option and its default.

```sh
kneemesh-brep meniscus --label-map meniscus.nii --output meniscus.brep --cap planar
kneemesh-mesh meniscus --brep meniscus.brep --output meniscus.msh --max-size 1.0
```

A multilabel LabelMap takes `--labels` naming its values, such as `--labels '{"tibia": 1, "medial_meniscus": 3}' --name medial_meniscus`. Separate single-label LabelMaps can be given with `--label-maps '{"tibia": "tibia.nii", "medial_meniscus": "meniscus.nii"}'` instead, and are resampled onto one grid. See the documentation's usage page.

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
