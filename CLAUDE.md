# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

The project is managed with uv and requires Python 3.13 or newer. Type checking uses basedpyright, a dev dependency configured under `[tool.basedpyright]` in `pyproject.toml`; keep it at 0 errors and 0 warnings. Ruff, also a dev dependency, formats code and sorts imports, configured under `[tool.ruff]` with a 150-character line length; run both commands below before committing.

```sh
uv sync                                        # install dependencies
uv run pytest                                  # run all tests
uv run pytest tests/test_vol.py                # run one test file
uv run pytest tests/test_vol.py::test_name     # run one test
uv run basedpyright                            # type check src and tests
uv run ruff check --fix                        # lint and sort imports
uv run ruff format                             # format code
uv add <package>                               # add a runtime dependency
uv add --dev <package>                         # add a dev dependency
uv build                                       # build sdist and wheel
uv run kneemesh-assemble --help                # single-label LabelMaps to a multilabel LabelMap
uv run kneemesh-brep --help                    # LabelMap to B-rep pipeline
uv run kneemesh-mesh --help                    # B-rep to mesh pipeline
uv run --group docs sphinx-build -W -b html docs docs/_build/html   # build the docs
```

`examples/` holds pipeline configurations for the real menisci in `tests/dat`, run from the repository root, such as `uv run kneemesh-assemble --config examples/menisci_assemble.json`. Its `.gitignore` tracks only the configurations, so outputs and logs written there stay untracked.

## Docs

`docs/` is a Sphinx project with the Furo theme. Pages are Markdown through MyST, with `$...$` and `$$...$$` math rendered by MathJax. Napoleon reads the Google-style docstrings, and `autodoc_pydantic` renders config models with each field's type, default, `description`, and constraints, so document a config option in its `Field(description=...)`. The build treats warnings as errors.

## Type checking

basedpyright runs in `recommended` mode with these exceptions:

- SimpleITK, VTK, and gmsh ship without type information, so the rules about `Any` and unknown types and missing stubs are off. Annotate our own functions fully anyway.
- `reportUnusedCallResult` is off because gmsh and VTK calls return tags and status values that are often not needed.
- Under `tests/`, unannotated fixture parameters and private helper use are allowed.

Keep these points when writing code it checks:

- Pydantic coerces inputs such as `str` to `Path` or JSON text to a dict, but constructor signatures use the parsed types. Tests of coercion go through `Model.model_validate({...})`, and other code passes the parsed types.
- A fixture used only for its side effect, such as `in_tmp_path` changing directory, is requested with `@pytest.mark.usefixtures(...)` rather than an unused parameter.
- Narrow optionals before use, for example `assert saved is not None` in tests.
- A runtime check of a value its type already rules out, kept for untyped callers, takes `# pyright: ignore[reportUnreachable]` on the unreachable line.

## Architecture

The package converts segmented knee images into B-rep geometry and meshes, as a pipeline across `src/kneemesh/`. Only the meniscus is implemented; tibial (lateral and medial), femoral, and patellar cartilage and the three bones are planned. Root modules apply to every structure; the `geo/` and `mesh/` subpackages hold one module per structure (`meniscus.py`, `tibia.py`, `femur.py`, `patella.py`). The tibia, femur, and patella modules are docstring-only placeholders; the plan for cartilage is under "Cartilage (planned, not implemented)" below.

- `disk.py` reads and writes LabelMaps, `.vtp` poly data, and `.brep` files, and writes meshes in any volume format gmsh infers from the extension.
- `segmentation.py` defines `Segmentation`, a LabelMap with a `{name: value}` dict of its structures, and `assemble_label_maps`, which resamples named single-label LabelMaps onto one padded grid. `disk.write_segmentation` writes one as a LabelMap plus `<stem>_labels.json`, which `read_segmentation` reads back; the pipelines write it to `assembled_label_map` when that is set. Every B-rep pipeline reads a `Segmentation`, so cartilage can be oriented against its bone and anatomical neighbors can define surface and node sets.
- `vol.py` operates on LabelMaps: the signed distance field, isocontour extraction and smoothing, `require_3d_label_map`, and the `ContourSlice` value.
- `brep.py` holds the `BRep` value, `gmsh_session`, `export_brep`, the B-spline helpers that structure modules build geometry from (chord parameters, periodic and clamped fits, splitting, adding curves, wires, and swept surfaces to gmsh), and `contours_to_polydata`.
- `tet_meshing.py` meshes any solid B-rep with tetrahedra, with sizes defaulting to fractions of the shortest side of the B-rep's oriented bounding box (`oriented_extents`, principal axes of points sampled on its curves and non-planar faces): `min_size` 1/20, capped at `max_size`, so small features and curvature sizing cannot make excessively small elements, and `max_size` 1/4, at least `min_size`, because gmsh's own default from the overall extent left the thin real medial meniscus with no tetrahedra. It raises ValueError when gmsh produces no tetrahedra, and defines the `Mesh` it returns: node coordinates, per-type connectivity in gmsh's node ordering, and gmsh element type numbers. Its `gmsh_options`, `import_solid`, `volume_mesh`, and `require_element_order` are shared with structure meshers. `write_mesh` rebuilds a `Mesh` as a discrete gmsh volume to write it.
- `mesh_optimization.py` moves the nodes of any hexahedral `Mesh` to raise its scaled Jacobians, and reports quality with `scaled_jacobians` and `mesh_quality`. `MeniscusMesh.optimize` (`--meniscus.optimize`) runs it after hexahedral meshing.
- `geo/meniscus.py` fits the centerline, slices the surface into ordered contours, lofts them into a B-rep with gmsh's Open Cascade kernel, and builds the five-block solid that hexahedral meshing needs.
- `mesh/meniscus.py` meshes the five-block solid into block-structured hexahedra.
- `config.py` provides `BaseConfig`, the pydantic base class for configuration models parsed from JSON, which rejects unknown keys. `Structure` is the `Literal` of the nine structure names (`lateral_meniscus`, `medial_meniscus`, `patellar_cartilage`, `femoral_cartilage`, `lateral_tibial_cartilage`, `medial_tibial_cartilage`, `femur`, `tibia`, `patella`), and the keys of every `labels`, `label_maps`, and `breps` mapping must be among them. `BONES` maps each cartilage to the bone that must share its segmentation. `LabelMapAssembly` configures `kneemesh-assemble`, `LabelMapBrep` `kneemesh-brep`, and `BrepMesh` `kneemesh-mesh`. The latter two hold the shared settings plus one section per kind of implemented structure, such as `meniscus: MeniscusBrep` and `meniscus: MeniscusMesh`, which hold every pipeline argument with its default and apply to both menisci. Their `output` is a suffix: each structure is written to `output_directory / f"{structure}{output}"` by `output_path`. B-rep suffixes must end in `.brep`; the mesh suffix defaults to `_{element_type}.msh`, kept as a template so a changed `element_type` renames the output. `element_type` (`hex8`, `hex20`, `hex27`, `tet4`, `tet10`) replaces separate shape and order settings; `element_shape`, `element_order`, and `second_order_incomplete` are properties derived from it through `ELEMENT_TYPES`. Mapping fields accept JSON text, so they work as command-line options. The gmsh algorithm enums are string enums named for readability, with gmsh's option number on `.number` and case-insensitive lookup, so configs store them as plain names. Optional numbers read "none" as None.
- `pipeline.py` runs the three console scripts, none of which take a subcommand. `configure` takes one model, builds an option for every field and `--section.field` options in an argument group for every section, layers command-line values over `--config` (merging section options into the file's section field by field), and saves the result: `settings<timestamp>.json` in the working directory without `--config`, or `<stem>_modified.json` beside it when options change it. Command-line values arrive as strings and rely on pydantic's lax coercion. `load_segmentation` turns a config's `label_map` + `labels` or `label_maps` into a `Segmentation`; `run_assembly` and B-rep configs with `label_maps` share one assembly step. `run_brep` builds every structure in the segmentation and `run_mesh` every structure in `breps`, each looking up the structure's name in `_BREP_BUILDERS` or `_MESHERS`. To implement a structure, add its section to the configs and its functions to those registries. Every structure is checked before any work starts, and a missing one raises `NotImplementedError`, which the console scripts report as an error with exit code 1. Each output is written as soon as it is made.
- Logging uses loguru. `__init__.py` disables the `kneemesh` logger so the library stays silent; modules log step details at DEBUG. The pipelines enable it for the run, sending INFO to stderr and DEBUG to `assemble<timestamp>.log`, `brep<timestamp>.log`, or `mesh<timestamp>.log` in the working directory, or to `<config stem>.log` beside `--config`. They remove their handlers and disable the logger again on exit. Use `logger`, not `print`.

### LabelMap convention

A LabelMap is a scalar `sitk.Image` with pixel type `sitkUInt8`, `sitkUInt16`, or `sitkInt16` (`vol.LABEL_MAP_PIXEL_IDS`), not SimpleITK's run-length `sitkLabel*` pixel types. `disk.py` enforces this on both read and write and keeps the pixel type rather than casting. It accepts only `.nii`, `.nrrd`, and `.mhd` + `.raw`, so `.nii.gz` is rejected. `vol.signed_distance_field` and `geo.meniscus.centerline` additionally require a 3D image.

### Segmentations

A B-rep config gives either `label_map` with the required `labels` (`{structure: value}`) or `label_maps` (`{structure: path}`), never both. Every structure named is built, and a cartilage's bone (`BONES`) must be named too. `Segmentation` rejects value 0, duplicate values, and names whose value is absent from the image. Keep these points when changing `assemble_label_maps`:

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
- The loft does not use `addThruSections`. Open Cascade's approximation overshoots in the end spans where real sections taper toward the horns, folding the surface, and gmsh's 2D meshing then never finishes; C1 continuity, every `parametrization`, and shared contour knots did not prevent it on the real lateral meniscus after label Gaussian assembly. Instead `brep.add_periodic_swept_surface` interpolates periodic contour splines that share one knot vector (fitted at the mean of the contours' chord parameters), with each column of poles interpolated along the curve at slice-center chord parameters, as for the hexahedral blocks. The surface then passes exactly through every contour, and its end rims are the end contours' curves.
- Solids are built from the lofted surface, the two caps, and a sewn surface loop.
- A smooth cap reuses the end contour's poles and knots so its rim matches the loft's edge. Its quadratic meridians collapse to one apex by repeating the apex point tag for the last row of poles.
- Planar caps need contours projected exactly onto their planes, because VTK's single-precision points fall outside Open Cascade's tolerance.
- The loft's degree along the curve is `min(spline_order, 3, slices - 1)`, so `spline_order=1` gives a single ruled (1, 1) face.
- `gmsh.write` exports only the highest-dimension shapes, so stray wires do not need removing.
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

### Hexahedral mesh optimization

`optimize_hexahedral_mesh` runs smart Laplacian smoothing, then projected gradient descent on the sum of 1 / SJ over all element corners. Keep these points when changing it:

- The objective is infinite for any corner with a determinant at or below zero, so a backtracking step can never invert an element. Optimization therefore needs a valid start, and is skipped with a warning if smoothing leaves an element inverted.
- 1 / SJ, unlike Knupp's condition number, does not penalize unequal edge lengths, so it leaves the deliberately long elements along the meniscus alone. It weighs the worst corners most.
- The gradient uses d log SJ / dE = E^-T - e_k / |e_k|^2 for the matrix E of a corner's edge rows, with E^-T from cross products rather than `np.linalg.inv`, and scatters with `np.bincount`; `np.add.at` and batched `inv` made it twice as slow. Steps are scaled by each node's local size squared so the step length is dimensionless.
- Boundary nodes are classified from the mesh's own boundary quadrilaterals, not the B-rep: feature edges are where adjacent boundary faces meet at more than `feature_angle`; nodes where feature edges end, branch, or turn are fixed. Surface nodes are limited to `surface_deviation` off and `surface_travel` along their starting tangent plane, as fractions of the mean incident edge length; feature edge nodes the reverse.
- Optimization pushes surface nodes outward: at `surface_deviation` 0.1 the real menisci swell 5-7%. `preserve_volume` removes the area-weighted mean normal offset in the projection, which keeps volume within 0.5% at the same quality.
- Smart Laplacian is a Jacobi sweep that undoes moves around any element that got worse, repeating until none did. It untangles the inverted test grid but barely changes gmsh's transfinite meshes; the optimization does most of the work there.
- High-order nodes move by trilinear interpolation of their element's corner displacements, using the local node coordinates gmsh reports, so curved edges keep their offsets. Quality is measured on corners only.
- On the real menisci at 4 by 20 elements, optimization takes about 2 s and raises the minimum scaled Jacobian from 0.08-0.17 to 0.36-0.40 and the mean from 0.64-0.72 to 0.80-0.83. 1000 steps instead of 200 add little.

### Cartilage (planned, not implemented)

This section records the agreed plan for cartilage B-reps and meshes; its points are decisions unless marked open. Nothing here exists in the code yet; update it as each piece lands, and move settled facts into the sections above. The lateral and medial tibial cartilage and the patellar cartilage share one strategy, since each is a roughly cap-shaped sheet over its bone. The femoral cartilage differs, because it wraps the condyles and joins them through the trochlea. Logic shared by every cartilage (side classification, thickness, perimeter smoothing, thin-region handling) should live in a root module rather than be duplicated in `geo/tibia.py`, `geo/patella.py`, and `geo/femur.py`. Every cartilage's bone (`BONES`) is in the same `Segmentation`, which is what the steps below use to tell the bone side from the articular side.

Anticipated problems:

- The isocontour of a cartilage label curves sharply around its perimeter, where the articular and bone-side surfaces meet at a thin edge. Hexahedra there degenerate, and tetrahedra refine to excessive element counts.
- Cartilage can be very thin in places, with its articular and bone-side surfaces nearly touching. Tetrahedra cope with this by refining; block-structured hexahedra do not.
- The meniscus defaults were tuned for a structure several millimeters across. Cartilage is often only a few voxels thick, so the default distance-field Gaussian (3 voxel edges in the central 95%) and windowed sinc smoothing will cost a larger fraction of its volume and may erase thin regions. Measure this on real cartilage before keeping the defaults.

Hexahedral B-rep, planned:

1. Extract and smooth the isocontour as for the meniscus, then classify its triangles as articular or bone side and threshold each, discarding the steep perimeter band between them. Classifying against the bone (distance from the bone's distance field, and the sign of the triangle normal against that field's gradient) works for all three cartilages, including the femoral one, where no single axis separates the sides.
2. The thresholds leave a jagged perimeter. Smooth it by one of: windowed sinc smoothing with `BoundarySmoothingOn`; fitting the boundary edge loop with a periodic spline and smoothing the spline; or, a strategy that has worked before, building the jagged boundary as a polyline, running it through `vtkTubeFilter` at a chosen radius, and clipping the surface with the tube. Prefer one smoothed perimeter shared by both sides (projected through the thickness) over two independent ones, so the side wall between them has positive height everywhere.
3. Fit a trimmed B-spline surface to each side, bounded by the smoothed perimeter, and close the solid by extruding and sewing a side wall between the two perimeters. The two perimeter curves need matching start points and directions, for the same reason contours are aligned before lofting. The B-rep must not fold or self-intersect.
4. Partition for transfinite meshing. Tibial and patellar cartilage: a butterfly partition of the articular and bone-side surfaces (an inner quadrilateral and four outer blocks), with the bone-side partition the image of the articular one through the thickness, so each block is a hexahedron bounded by a patch on each side and four walls, transfinite through the thickness like the meniscal horns. Femoral cartilage: likely a half butterfly on the trochlear groove glued to half butterflies on the medial and lateral condyles. The trochlear block edges must split where the condyles meet it and at the intercondylar notch, with matching element counts across each glued interface.
5. Lessons from the meniscus apply: build each block face as its own B-spline surface from shared points so neighboring faces match exactly, rather than lofting with `addThruSections` or trimming with `wireTags` (gmsh projects new curves, so trimmed faces do not share edges with their neighbors).

Tetrahedral B-rep, planned: tetrahedra refine adaptively, so the perimeter needs less aggressive treatment. Local smoothing or adjustment of extremely thin or sharply curved regions should suffice, and contouring followed by a loft (as in `contours_to_brep`, including the custom swept surface rather than `addThruSections`) is the likely construction. The femoral cartilage changes topology across slices near the intercondylar notch, so its slicing direction needs care.

Thin regions, planned: measure local thickness between the articular and bone-side surfaces and compare it with fractions of the mean thickness. Thickening moves the bone-side control points away from the articular surface until the thickness reaches the minimum, keeping the displacement smooth. Moving the bone side preserves the articular geometry that contact depends on.

- Hexahedra only thicken and never cut holes, because a hole adds an inner perimeter that the butterfly partitions cannot map. The minimum thickness defaults to 0.2 of the mean, which allows elements up to about 5 times more elongated there.
- Tetrahedra get two lower fractions (values open): below the upper one, thicken; below the lower one, cut a hole representing a cartilage defect.
- Thickened cartilage overlaps its bone. That is acceptable for most models: bone processing always runs downstream of cartilage, and a future assembly pipeline can add tooling that adjusts the bone geometry to the cartilage, never the reverse.

Synthetic fixtures with a bone label (for example a shell on a sphere for tibial and patellar cartilage, including a thin region) are still worth having for fast unit tests. Real cartilage and bone labels are in `tests/dat` (see Tests). The femur and tibia labels are cut off by the scan's field of view along the shaft, so bone B-reps will need a planar cut face there.

## Tests

Test files mirror the package: root modules have `tests/test_<module>.py`, and structure modules have `tests/geo/test_<structure>.py` and `tests/mesh/test_<structure>.py`. Duplicate basenames work because pytest runs with `--import-mode=importlib`. `tests/conftest.py` defines session-scoped fixtures that every test shares, so tests must not modify them in place:

- `meniscus_label` is a synthetic meniscus: a triangular cross-section swept 270° about the z axis, with the gap facing -x.
- `meniscus_centerline` is `centerline(meniscus_label)`. Use it instead of calling `centerline` on the same image again.
- `meniscus_distance_field`, `meniscus_mesh`, `meniscus_smoothed_mesh`, `meniscus_slices`, `meniscus_contours`, `meniscus_surface`, `meniscus_solid`, `meniscus_domed_solid`, `meniscus_hex_slices`, `meniscus_hex_blocks`, `meniscus_tetrahedra`, and `meniscus_hexahedra` are the later pipeline stages for the same label.
- `real_label` is the real lateral meniscus in `tests/dat/lateral_meniscus.nii`, and `real_medial_label` the medial meniscus of the same knee in `tests/dat/medial_meniscus.nii`. `tests/dat` is stored in Git LFS.

Every file in `tests/dat` is one structure of the same knee, named for its structure key: the two menisci plus `femoral_cartilage.nii`, `lateral_tibial_cartilage.nii`, `medial_tibial_cartilage.nii`, `patellar_cartilage.nii`, `femur.nii`, `tibia.nii`, and `patella.nii`, which no fixtures read yet. Each is a single label 1 in `sitkUInt8`, cropped to the label's bounding box (`sitk.LabelShapeStatisticsImageFilter`, with no margin). The grids share spacing and orientation and are offset by whole voxels, so nearest-neighbor assembly reproduces every label exactly. SimpleITK has no 1-bit pixel type and ITK does not write NIfTI's binary type, so `sitkUInt8` is the smallest.

The fixture's voxelized cross-section centroid is r = 16.43, z = -1.67, not the ideal triangle's r = 16.67, z = -1.33. Its bottom slice lies exactly on the triangle's base and counts as a whole voxel layer, which also makes its voxel volume about 12% above the analytic value. Compare against the voxelized values or use a suitable tolerance.
