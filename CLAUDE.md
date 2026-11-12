# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

The project is managed with uv and requires Python 3.13 or newer. No linter or formatter is configured.

```sh
uv sync                                        # install dependencies
uv run pytest                                  # run all tests
uv run pytest tests/test_vol.py                # run one test file
uv run pytest tests/test_vol.py::test_name     # run one test
uv add <package>                               # add a runtime dependency
uv add --dev <package>                         # add a dev dependency
uv build                                       # build sdist and wheel
uv run kneemesh-brep meniscus --help           # LabelMap to B-rep pipeline
uv run kneemesh-mesh meniscus --help           # B-rep to mesh pipeline
uv run --group docs sphinx-build -W -b html docs docs/_build/html   # build the docs
```

## Docs

`docs/` is a Sphinx project with the Furo theme. Pages are Markdown through MyST, with `$...$` and `$$...$$` math rendered by MathJax. Napoleon reads the Google-style docstrings, and `autodoc_pydantic` renders config models with each field's type, default, `description`, and constraints, so document a config option in its `Field(description=...)`. The build treats warnings as errors.

## Architecture

The package converts segmented knee images into B-rep geometry and meshes, as a pipeline across `src/kneemesh/`. Only the meniscus is implemented; tibial (lateral and medial), femoral, and patellar cartilage and the three bones are planned. Root modules apply to every structure; the `geo/` and `mesh/` subpackages hold one module per structure (`meniscus.py`, `tibia.py`, `femur.py`, `patella.py`). The tibia, femur, and patella modules are docstring-only placeholders.

- `disk.py` reads and writes LabelMaps, `.vtp` poly data, and `.brep` files, and writes meshes in any volume format gmsh infers from the extension.
- `segmentation.py` defines `Segmentation`, a LabelMap with a `{name: value}` dict of its structures, and `assemble_label_maps`, which resamples named single-label LabelMaps onto one padded grid. Every B-rep pipeline reads a `Segmentation`, so cartilage can be oriented against its bone and anatomical neighbors can define surface and node sets.
- `vol.py` operates on LabelMaps: the signed distance field, isocontour extraction and smoothing, `require_3d_label_map`, and the `ContourSlice` value.
- `brep.py` holds the `BRep` value, `gmsh_session`, `export_brep`, the B-spline helpers that structure modules build geometry from (chord parameters, periodic and clamped fits, splitting, adding curves, wires, and swept surfaces to gmsh), and `contours_to_polydata`.
- `tet_meshing.py` meshes any solid B-rep with tetrahedra and defines the `Mesh` it returns: node coordinates, per-type connectivity in gmsh's node ordering, and gmsh element type numbers. Its `gmsh_options`, `import_solid`, `volume_mesh`, and `require_element_order` are shared with structure meshers. `write_mesh` rebuilds a `Mesh` as a discrete gmsh volume to write it.
- `geo/meniscus.py` fits the centerline, slices the surface into ordered contours, lofts them into a B-rep with gmsh's Open Cascade kernel, and builds the five-block solid that hexahedral meshing needs.
- `mesh/meniscus.py` meshes the five-block solid into block-structured hexahedra.
- `config.py` provides `BaseConfig`, the pydantic base class for configuration models parsed from JSON, which rejects unknown keys. `LabelMapBrep` (segmentation source, structure `name`, output) and `BrepMesh` (B-rep, output, element order) hold the settings every structure shares. Each structure has a `<Structure>Brep` and `<Structure>Mesh` model, such as `MeniscusBrep` and `MeniscusMesh`, holding every pipeline argument with its default; the planned structures' models hold only the shared fields. Cartilage B-rep models extend `CartilageBrep`, which adds the underlying `bone`; lateral and medial tibial plates share one strategy and differ only in `name`. Mapping fields (`labels`, `label_maps`) accept JSON text, so they work as command-line options. The gmsh algorithm enums are string enums named for readability, with gmsh's option number on `.number` and case-insensitive lookup, so configs store them as plain names. Optional numbers read "none" as None.
- `pipeline.py` runs the two console scripts. `STRUCTURES` maps each subcommand (`meniscus`, `tibial-cartilage`, ...) to its two config models. `configure` takes a subcommand-to-model mapping, builds an option for every field of each model, layers command-line values over `--config`, and saves the result: `settings<timestamp>.json` in the working directory without `--config`, or `<stem>_modified.json` beside it when options change it. Command-line values arrive as strings and rely on pydantic's lax coercion. `load_segmentation` turns a config's `label_map` + `labels` or `label_maps` into a `Segmentation`. `run_brep` and `run_mesh` dispatch on the config's type and raise `NotImplementedError` for planned structures, which the console scripts report as an error with exit code 1.
- Logging uses loguru. `__init__.py` disables the `kneemesh` logger so the library stays silent; modules log step details at DEBUG. The pipelines enable it for the run, sending INFO to stderr and DEBUG to `brep<timestamp>.log` or `mesh<timestamp>.log` in the working directory, or to `<config stem>.log` beside `--config`. They remove their handlers and disable the logger again on exit. Use `logger`, not `print`.

### LabelMap convention

A LabelMap is a scalar `sitk.Image` with pixel type `sitkUInt8`, `sitkUInt16`, or `sitkInt16` (`vol.LABEL_MAP_PIXEL_IDS`), not SimpleITK's run-length `sitkLabel*` pixel types. `disk.py` enforces this on both read and write and keeps the pixel type rather than casting. It accepts only `.nii`, `.nrrd`, and `.mhd` + `.raw`, so `.nii.gz` is rejected. `vol.signed_distance_field` and `geo.meniscus.centerline` additionally require a 3D image.

### Segmentations

A B-rep config gives either `label_map` with `labels` (`{name: value}`; None means the file holds only `name` with value 1) or `label_maps` (`{name: path}`), never both, and `name` (and a cartilage's `bone`) must be among the given names. `Segmentation` rejects value 0, duplicate values, and names whose value is absent from the image. Keep these points when changing `assemble_label_maps`:

- Every nonzero voxel of an input is its structure. Values are 1, 2, ... in input order, and a later structure overwrites earlier ones where they overlap, with a warning. A structure that vanishes in resampling or is entirely overwritten raises.
- The shared grid uses the first input's direction cosines. Each input axis counts toward the shared axis nearest it in direction, and each shared axis takes the smallest spacing from any input. Taking the minimum by index instead mixes up axes of inputs with permuted orientations.
- Along each axis the grid's voxel faces line up with the input that has the smallest spacing there, so that input resamples exactly when it shares the orientation. Anchoring to the bounding box's corner instead shifts it by up to half a voxel.
- `sitkLabelGaussian` is the default interpolator and `sitkNearestNeighbor` the alternative. Label Gaussian rounds thin corners even on an identical grid: it drops about 1% of the real test label.
- The default padding of 4 voxels keeps the distance field's Gaussian support inside the image around every structure.

### How `geo.meniscus.centerline` works

`sitk.BinaryThinning` only thins in 2D. Given a 3D image it thins slice by slice and returns a sheet, so it cannot produce the curve directly. It only provides the initial estimate:

1. Project the label along the image axis closest to its thinnest principal direction, then thin the 2D projection. The projection is padded with background first: thinning treats the image edge as an open end, so a label cut off by the edge would otherwise skeletonize into disconnected branches and the longest path would cover only part of the arc.
2. Take the longest path through the skeleton and trim one mean half-width from each end, because the skeleton forks into the label's corners there.
3. Fit a spline, then repeatedly replace it with a fit through centroids of cross-section slabs normal to the curve. Stations extend past both ends so the curve can grow back toward the tips, and partial sections at the tips are discarded.

Section spacing and smoothing tolerance scale with the structure's half-width as well as the voxel size. Tying them to the voxel size alone makes the curve hook at the tips on fine grids.

The result is a cubic `scipy.interpolate.BSpline` over u in [0, 1], in physical coordinates, and evaluating it at N parameters gives an `(N, 3)` array. `make_splprep` natively evaluates to `(3, N)`, so `_fit` rebuilds the spline to normalize this. Which end of the structure is u = 0 is unspecified, so compare curves by nearest-point distance rather than parameter by parameter.

### Cross-section contours

`signed_distance_field` feeds `extract_isocontour`, which returns a closed VTK triangle mesh from `vtkFlyingEdges3D`. `smooth_isocontour` smooths it with `vtkWindowedSincPolyDataFilter`, and `slice_isocontour` cuts the result along the centerline and returns `ContourSlice` objects holding ordered points from a closed Kochanek spline. Keep these points when changing this code:

- `sitk.SignedMaurerDistanceMap` alone is zero on boundary voxel centers, half a voxel inside the label. `signed_distance_field` averages it with the inverted mask's map so zero lies on the voxel edges.
- `signed_distance_field` then smooths with a recursive Gaussian to remove the voxel staircase. The default sigma, per axis, puts 3 voxel edges in the central 95%. It rounds sharp edges, costing the test fixture about 5% of its volume. Pass `sigma=0` for the exact field.
- `extract_isocontour` pads the field so the mesh stays closed where a label touches the image edge.
- `smooth_isocontour` turns boundary and feature-edge smoothing off and non-manifold smoothing on, and normalizes coordinates so its parameters don't depend on physical scale. The Gaussian leaves ripples in the mesh that make slice contours wiggle. The defaults, 50 iterations at pass band 0.001, cut the contours' excess turning by about 70% on the real test label and cost about 3% of the volume. A pass band of 0.001 at 100 iterations shrinks the fixture by 11%. `iterations=0` disables smoothing.
- A cutting plane is infinite and can also cut the far limb of the C-shaped structure, so each slice takes the loop nearest the centerline.
- Contours run counterclockwise about the slice normal. Each one's point order is rotated to match the previous slice, because a loft joins contours by parameter and a drifting start point makes the surface bulge at sharp corners.
- Slices are clipped only from the two ends, until one reaches 20% of the mean area. A narrow section between the ends is kept.

### B-reps through gmsh

gmsh holds its models in global state, so a `BRep` is a plain value holding the `.brep` file's text, and all gmsh work happens inside `gmsh_session()`. Keep these points when changing `geo.meniscus.contours_to_brep`:

- gmsh treats a B-spline whose first and last point tags match as periodic, so contours are built with periodic knots.
- `addThruSections(makeSolid=True)` returns a volume with no end faces and no error. Solids are instead built from the lofted surface, the two caps, and a sewn surface loop.
- A smooth cap reuses the end contour's poles and knots so its rim matches the loft's edge. Its quadratic meridians collapse to one apex by repeating the apex point tag for the last row of poles.
- Planar caps need contours projected exactly onto their planes, because VTK's single-precision points fall outside Open Cascade's tolerance.
- The loft's degree along the curve is capped at 3 regardless of `spline_order`, and only a ruled loft achieves degree 1.
- The loft uses `continuity="C1"`. Open Cascade's default C2 overshoots in the end spans where real sections taper toward the horns, folding the surface, and gmsh's 2D meshing then never finishes. `smoothing=True` in `addThruSections` hangs outright.
- `gmsh.write` exports only the highest-dimension shapes, so the input wires do not need removing.
- `occ.getMass` is unreliable on ruled surfaces (`spline_order=1`). Mesh the shape to check its volume.

### Hexahedral blocks

`contours_to_hex_blocks` splits each section with a butterfly (O-grid) partition: an inner quadrilateral block plus four outer blocks, one on each of four boundary arcs, joined by straight spokes. Every boundary point meets an inward spoke, so boundary cells stay near square even at the meniscus's sharp, smoothing-rounded corners. The Y split it replaced put block corners at those corners, which gave cells nearly 180° angles there and inverted elements on the real test label. Keep these points when changing it:

- The four splits divide the contour into equal-length arcs. `_butterfly_split_indices` rotates them to minimize the turning angle near them, averaged over all slices, so corners fall mid-arc. The first split is a point index because each contour is rolled to start there.
- Inner block corners lie halfway (`_INNER_BLOCK_SCALE`) from the section centroid to the split points. Mesh quality barely changes between 0.35 and 0.7.
- Each arc is sampled at `_PIECE_SAMPLES` points for its face's section curves. Fewer than 20 move the faces more than 1e-3 off the contour points.
- It does not use `addThruSections`. Separate lofts choose their own parameterization along the curve, so adjacent lofts cross between slices and fragment shatters them. Instead every face is a B-spline surface interpolating section curves fitted with a shared knot vector, with each column of poles interpolated along the curve at shared parameters. A face's edges then depend only on their own points, so neighboring faces match exactly.
- Each block is sewn separately, then the blocks are fragmented, which merges the identical copies of shared spoke faces. Trimming faces with existing curves (`wireTags`) does not share edges, because gmsh projects new ones.
- `hexahedral_mesh` makes every curve, surface, and volume transfinite. Curves on the planar caps get `elements_per_edge` and all others `elements_along`.
- Hexahedral tests use 10-point contours, which give valid elements and small B-reps. Contours need at least 8 points.
- gmsh's "initial mesh only" 2D algorithm crashes the process with a segmentation fault, and BAMG fails on these shapes, so `Algorithm2D` omits them. `Algorithm3D` omits R-tree, which writes `new_points.pos` into the working directory.

## Tests

Test files mirror the package: root modules have `tests/test_<module>.py`, and structure modules have `tests/geo/test_<structure>.py` and `tests/mesh/test_<structure>.py`. Duplicate basenames work because pytest runs with `--import-mode=importlib`. `tests/conftest.py` defines session-scoped fixtures that every test shares, so tests must not modify them in place:

- `meniscus_label` is a synthetic meniscus: a triangular cross-section swept 270° about the z axis, with the gap facing -x.
- `meniscus_centerline` is `centerline(meniscus_label)`. Use it instead of calling `centerline` on the same image again.
- `meniscus_distance_field`, `meniscus_mesh`, `meniscus_smoothed_mesh`, `meniscus_slices`, `meniscus_contours`, `meniscus_surface`, `meniscus_solid`, `meniscus_domed_solid`, `meniscus_hex_slices`, `meniscus_hex_blocks`, `meniscus_tetrahedra`, and `meniscus_hexahedra` are the later pipeline stages for the same label.
- `real_label` is the real meniscus in `tests/dat/mns_l.nii`, a single label. There is no multilabel test data yet; `tests/test_segmentation.py` builds multilabel cases from the fixture.

The fixture's voxelized cross-section centroid is r = 16.43, z = -1.67, not the ideal triangle's r = 16.67, z = -1.33. Its bottom slice lies exactly on the triangle's base and counts as a whole voxel layer, which also makes its voxel volume about 12% above the analytic value. Compare against the voxelized values or use a suitable tolerance.
