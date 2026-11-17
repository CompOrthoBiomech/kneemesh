# kneemesh

[![PyPI](https://img.shields.io/pypi/v/kneemesh)](https://pypi.org/project/kneemesh/)
[![Coverage](https://img.shields.io/endpoint?url=https://comporthobiomech.github.io/kneemesh/coverage.json)](https://github.com/CompOrthoBiomech/kneemesh/actions/workflows/release.yml)
[![Docs](https://img.shields.io/badge/docs-gh--pages-blue)](https://comporthobiomech.github.io/kneemesh/)

Generate B-rep geometry of knee structures from a segmentation LabelMap, and mesh it with high-quality elements. The menisci, the lateral and medial tibial, patellar, and femoral cartilage, and the femur, tibia, and patella are supported. For the meniscus:

- **B-rep from segmentation labels.** Reads a LabelMap (`.nii`, `.nrrd`, or `.mhd`), fits a centerline through the labeled meniscus, slices its smoothed surface into cross-section contours, and lofts them into a solid B-spline B-rep (`.brep`) with planar or smooth end caps.
- **Tetrahedral meshing** with linear 4-node or quadratic 10-node tetrahedra.
- **Hexahedral meshing** with 8-node, 20-node, or 27-node hexahedra. By default the largest cross-section is meshed with unstructured quadrilaterals about `--meniscus.transverse-size` across and swept along the meniscus in layers about `--meniscus.circumferential-size` apart; `--meniscus.hex-design butterfly` instead splits the solid into a butterfly (O-grid) of five blocks of structured hexahedra. Node optimization, on by default for the meniscus, untangles any inverted elements and raises the scaled Jacobians: swept hexahedra reach minimum scaled Jacobians of about 0.5 on the real test menisci, with aspect ratios under 9. For them the section's thin inner edge is cut off straight at a quarter of its outer height (`--meniscus.inner-clip-ratio`), so the cut's top and bottom are edges the elements follow. The slice contours keep each corner of the section, such as the sharp inner edge, between the same two points on every slice, so the geometry's lines along the meniscus follow its ridges.

For the tibial, patellar, and femoral cartilage:

- **B-rep from segmentation labels.** Tells the cartilage's articular side from the side against its bone, which must be in the segmentation, and fits both as heights over a plane, or for the femoral cartilage a cylinder, within a smoothed perimeter. Thin regions are thickened on the bone side, and for tetrahedra, the thinnest regions and gaps in the label become holes.
- **Tetrahedral meshing** as for the meniscus.
- **Hexahedral meshing** of the same solid: one side is meshed with quadrilaterals of about `--cartilage.element-size`, and hexahedra stack along straight columns to the other side. Optional node optimization (`--cartilage.optimize true`) untangles any inverted elements and weighs the elements' Frobenius aspect ratios along with their scaled Jacobians, keeping the worst under about 3.5 on the real test cartilage.

For the femur, tibia, and patella:

- **Surface from segmentation labels.** Fills the label's cavities, optionally cuts the shafts the scan cuts off shorter (`--femur.shaft-cut-fraction`, and likewise for each bone), and writes the smoothed, decimated surface as `.stl`; gmsh reparametrizes it for meshing.
- **Surface meshing** with 3- or 6-node triangles or 4-, 8-, or 9-node quadrilaterals (`--femur.element-type`, `tri3` by default).
- **Tetrahedral meshing** with 4- or 10-node tetrahedra, finest in the cortical shell (`--femur.cortical-element-size` within `--femur.cortical-distance` of the surface) and growing by `--femur.cancellous-grad` per unit of depth up to `--femur.max-size`.

Each bone has its own settings: `femur`, `tibia`, and `patella`. The menisci share theirs, as do the tibial and patellar cartilage, unless one is given its own, such as `--lateral-meniscus.spacing 2`, which takes the shared section's other values.

Every mesh has named node and element sets of its surface: each bone's surface, each cartilage's articular and bone sides, and each meniscus's anterior and posterior horns and body. `.vtu` and `.vtkhdf` (a multiblock) hold them; other formats gmsh infers from the file extension, such as `.msh`, `.vtk`, or `.inp`, hold the elements only. `--assembled-mesh knee.vtu` or `knee.vtkhdf` also writes every structure's mesh as one, numbered on through the structures. They also hold each element's quality, measured on its corner nodes: its scaled Jacobian and Frobenius aspect ratio, and each quadrilateral face's warp.

**Documentation:** <https://comporthobiomech.github.io/kneemesh/>

## Installation

```sh
pip install kneemesh
```

## Usage

The command-line pipelines take a JSON configuration, `--option value` overrides, or both. `kneemesh-brep` builds every structure named in the segmentation, and `kneemesh-mesh` every structure named in `--breps`; the names, such as `lateral_meniscus`, pick each structure's pipeline. Outputs are written as `<structure><output>`, where `--output` is a suffix such as `_hex8.msh`. Run `kneemesh-brep --help` to list every option and its default.

```sh
kneemesh-assemble --label-maps '{"lateral_meniscus": "lateral.nii", "medial_meniscus": "medial.nii"}' --output knee.nrrd
kneemesh-brep --label-map knee.nrrd --labels "$(cat knee_labels.json)"
kneemesh-brep --label-map meniscus.nii --labels '{"medial_meniscus": 1}' --meniscus.cap planar
kneemesh-mesh --breps '{"medial_meniscus": "medial_meniscus.brep"}' --element-type tet10 --meniscus.max-size 1.0
```

A multilabel LabelMap takes `--labels` naming its values, such as `--labels '{"lateral_meniscus": 2, "medial_meniscus": 3}'`; values it does not name are ignored. `kneemesh-assemble` resamples separate single-label LabelMaps onto one grid and writes the multilabel LabelMap with its labels in `<stem>_labels.json`. `kneemesh-brep` can also take `--label-maps` directly and assemble them within the run. Example configurations are in `examples/`. See the documentation's usage page.

Each run saves the configuration it used, so it can be repeated with `--config`. Without `--config` it is saved as `settings<timestamp>.json` in the working directory. With `--config` and overrides, the changed configuration is saved beside the original as `<name>_modified.json`.

### Element types

| Elements | B-rep options | Mesh options |
|---|---|---|
| 4-node tetrahedra | | `--element-type tet4` (the default) |
| 10-node tetrahedra | | `--element-type tet10` |
| 8-node hexahedra | `--meniscus.hex-blocks true` | `--element-type hex8` |
| 20-node hexahedra | `--meniscus.hex-blocks true` | `--element-type hex20` |
| 27-node hexahedra | `--meniscus.hex-blocks true` | `--element-type hex27` |

Add `--meniscus.hex-design butterfly --meniscus.points-per-slice 10` for the meniscus's structured butterfly blocks. For the cartilage, `--cartilage.hex-blocks true` builds the B-rep with the defaults that suit hexahedra; any cartilage B-rep can be meshed with either element shape.

## GUI

`kneemesh-gui` runs every stage interactively in a browser: import labels, configure, build, and mesh each structure, view the results, and import or export the pipelines' JSON configurations. It needs the `gui` extra:

```sh
pip install "kneemesh[gui]"
kneemesh-gui
```

### Docker

The `Dockerfile` builds a slim image that serves the GUI on port 8080 and also has the console scripts. Its working directory, `/data`, is where outputs are written:

```sh
docker build -t kneemesh .
docker run --rm -p 8080:8080 -v "$PWD:/data" kneemesh
```

Then open http://localhost:8080. The container runs as the owner of the mounted directory, so it can write wherever that owner can and its outputs belong to them. VTK renders offscreen with Mesa's software EGL, so the image needs no display or GPU. To keep the image small, `docker/prune.py` deletes every VTK library, SciPy subpackage, and trame client bundle that a traced run of the pipelines and the GUI (`docker/trace.py`) doesn't load. As a result, `import vtk` doesn't work inside the image; import from `vtkmodules` instead.

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

## Acknowledgments

The test data in `tests/dat` are segmentation labels of one knee from the [Open Knee(s)](https://simtk.org/projects/openknee) project, which provides specimen-specific imaging, segmentations, and models of the knee freely to the public. We thank the Open Knee(s) team for making their data openly available. If you use these data, please cite the Open Knee(s) project:

> Erdemir A. Open Knee: Open Source Modeling and Simulation in Knee Biomechanics. *J Knee Surg*. 2016;29(2):107-116. doi:[10.1055/s-0035-1564600](https://doi.org/10.1055/s-0035-1564600)

The labels were cropped to each structure's bounding box and converted to 8-bit LabelMaps for this repository.

## License

Licensed under the [Apache License 2.0](LICENSE).
