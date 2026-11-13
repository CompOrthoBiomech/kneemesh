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

The package converts segmented knee images into B-rep geometry and meshes, as a pipeline across `src/kneemesh/`. The menisci and the lateral and medial tibial and patellar cartilage are implemented; the femoral cartilage and the three bones are planned. Root modules apply to every structure; the `geo/` and `mesh/` subpackages hold one module per structure or strategy (`meniscus.py`; `sheet.py` for the tibial and patellar cartilage; `tibia.py`, `femur.py`, `patella.py` for the bones and femoral cartilage). The tibia, femur, and patella modules are docstring-only placeholders.

- `disk.py` reads and writes LabelMaps, `.vtp` poly data, and `.brep` files, and writes meshes in any volume format gmsh infers from the extension.
- `segmentation.py` defines `Segmentation`, a LabelMap with a `{name: value}` dict of its structures, and `assemble_label_maps`, which resamples named single-label LabelMaps onto one padded grid. `disk.write_segmentation` writes one as a LabelMap plus `<stem>_labels.json`, which `read_segmentation` reads back; the pipelines write it to `assembled_label_map` when that is set. Every B-rep pipeline reads a `Segmentation`, so cartilage can be oriented against its bone and anatomical neighbors can define surface and node sets.
- `vol.py` operates on LabelMaps: the signed distance field, isocontour extraction and smoothing, `require_3d_label_map`, and the `ContourSlice` value.
- `cartilage.py` holds what every cartilage shares: `classify_sides` splits a cartilage surface into its articular side and the side against its bone (`CartilageSides`), and `field_value` and `field_gradient` sample a distance field at physical points.
- `brep.py` holds the `BRep` value, `gmsh_session`, `export_brep`, the B-spline helpers that structure modules build geometry from (chord parameters, periodic and clamped fits, splitting, adding curves, wires, and swept surfaces to gmsh), the loop helpers that butterfly partitions use (`resample_closed`, `butterfly_split_indices`, `turning_angles`), and `contours_to_polydata`.
- `tet_meshing.py` meshes any solid B-rep with tetrahedra, with sizes defaulting to fractions of the shortest side of the B-rep's oriented bounding box (`oriented_extents`, principal axes of points sampled on its curves and non-planar faces): `min_size` 1/20, capped at `max_size`, so small features and curvature sizing cannot make excessively small elements, and `max_size` 1/4, at least `min_size`, because gmsh's own default from the overall extent left the thin real medial meniscus with no tetrahedra. It raises ValueError when gmsh produces no tetrahedra, and defines the `Mesh` it returns: node coordinates, per-type connectivity in gmsh's node ordering, and gmsh element type numbers. Its `gmsh_options`, `import_solid`, `volume_mesh`, and `require_element_order` are shared with structure meshers. `write_mesh` rebuilds a `Mesh` as a discrete gmsh volume to write it.
- `mesh_optimization.py` moves the nodes of any hexahedral `Mesh` to raise its scaled Jacobians, and reports quality with `scaled_jacobians` and `mesh_quality`. Each mesh section's `optimize`, such as `--meniscus.optimize`, runs it after hexahedral meshing.
- `geo/meniscus.py` fits the centerline, slices the surface into ordered contours, lofts them into a B-rep with gmsh's Open Cascade kernel, and builds the five-block solid that hexahedral meshing needs.
- `mesh/meniscus.py` meshes the five-block solid into block-structured hexahedra.
- `geo/sheet.py` fits a cartilage sheet (`CartilageSheet`, heights of both sides over a plane within a perimeter) with `fit_sheet`, and builds its solid for tetrahedra (`sheet_to_brep`) or its five-block butterfly solid for hexahedra (`sheet_to_hex_blocks`).
- `mesh/sheet.py` meshes a sheet's five-block solid into block-structured hexahedra.
- `config.py` provides `BaseConfig`, the pydantic base class for configuration models parsed from JSON, which rejects unknown keys. `Structure` is the `Literal` of the nine structure names (`lateral_meniscus`, `medial_meniscus`, `patellar_cartilage`, `femoral_cartilage`, `lateral_tibial_cartilage`, `medial_tibial_cartilage`, `femur`, `tibia`, `patella`), and the keys of every `labels`, `label_maps`, and `breps` mapping must be among them. `BONES` maps each cartilage to the bone that must share its segmentation. `LabelMapAssembly` configures `kneemesh-assemble`, `LabelMapBrep` `kneemesh-brep`, and `BrepMesh` `kneemesh-mesh`. The latter two hold the shared settings plus one section per kind of implemented structure, `meniscus: MeniscusBrep` / `MeniscusMesh` for both menisci and `cartilage: CartilageBrep` / `CartilageMesh` for the tibial and patellar cartilage, which hold every pipeline argument with its default. B-rep sections extend `SurfaceBrep` (distance field and surface smoothing) and mesh sections `StructureMesh` (tetrahedral sizing and hexahedral optimization). `CartilageBrep` leaves `perimeter_radius` and `min_thickness` as None to pick defaults by `hex_blocks`, read through its `resolved_*` properties. Their `output` is a suffix: each structure is written to `output_directory / f"{structure}{output}"` by `output_path`. B-rep suffixes must end in `.brep`; the mesh suffix defaults to `_{element_type}.msh`, kept as a template so a changed `element_type` renames the output. `element_type` (`hex8`, `hex20`, `hex27`, `tet4`, `tet10`) replaces separate shape and order settings; `element_shape`, `element_order`, and `second_order_incomplete` are properties derived from it through `ELEMENT_TYPES`. Mapping fields accept JSON text, so they work as command-line options. The gmsh algorithm enums are string enums named for readability, with gmsh's option number on `.number` and case-insensitive lookup, so configs store them as plain names. Optional numbers read "none" as None.
- `pipeline.py` runs the three console scripts, none of which take a subcommand. `configure` takes one model, builds an option for every field and `--section.field` options in an argument group for every section, layers command-line values over `--config` (merging section options into the file's section field by field), and saves the result: `settings<timestamp>.json` in the working directory without `--config`, or `<stem>_modified.json` beside it when options change it. Command-line values arrive as strings and rely on pydantic's lax coercion. `load_segmentation` turns a config's `label_map` + `labels` or `label_maps` into a `Segmentation`; `run_assembly` and B-rep configs with `label_maps` share one assembly step. `run_brep` builds every structure in the segmentation and `run_mesh` every structure in `breps`, each looking up the structure's name in `_BREP_BUILDERS` or `_MESHERS`. A bone without a builder that is named only as a named cartilage's bone is used to classify that cartilage's sides and is not built. To implement a structure, add its section to the configs and its functions to those registries. Every structure is checked before any work starts, and a missing one raises `NotImplementedError`, which the console scripts report as an error with exit code 1. Each output is written as soon as it is made.
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

`optimize_hexahedral_mesh` runs smart Laplacian smoothing, then projected gradient descent on the sum of 1 / SJ over all element corners plus `aspect_weight` times the sum of their squared Frobenius aspect ratios. Keep these points when changing it:

- The objective is infinite for any corner with a determinant at or below zero, so a backtracking step can never invert an element. Optimization therefore needs a valid start, and is skipped with a warning if smoothing leaves an element inverted.
- 1 / SJ, unlike Knupp's condition number, does not penalize unequal edge lengths, so it leaves the deliberately long elements along the meniscus alone. It weighs the worst corners most. Alone it squares corners up by stretching elements: it raised the real medial meniscus's largest Frobenius aspect ratio from 24 to 89 and the medial tibial cartilage's from 20 to 75.
- The Frobenius aspect ratio of a corner is F = |E|_F^2 / (3 det(E)^(2/3)), Verdict's maximum aspect Frobenius over the eight corners for an element (`frobenius_aspects`, and `max_aspect` and `mean_aspect` in `MeshQuality`). It is 1 for a cube and also a barrier as det E falls to 0. Its gradient is d log F / dE = 2E / |E|_F^2 - (2/3) E^-T, reusing the cofactors. Squaring it weighs the worst corners more; F^2 lowered the real cartilage's largest aspect ratio about 10% more than F. `aspect_weight` defaults to 0 in `StructureMesh` and the function, so the meniscus keeps 1 / SJ alone: at 0.5 its mean scaled Jacobian fell from 0.80 to 0.68, fighting its deliberately long elements. `CartilageMesh` defaults to 0.5; between 0.5 and 2 the results barely differ.
- The gradient uses d log SJ / dE = E^-T - e_k / |e_k|^2 for the matrix E of a corner's edge rows, with E^-T from cross products rather than `np.linalg.inv`, and scatters with `np.bincount`; `np.add.at` and batched `inv` made it twice as slow. Steps are scaled by each node's local size squared so the step length is dimensionless.
- Boundary nodes are classified from the mesh's own boundary quadrilaterals, not the B-rep: feature edges are where adjacent boundary faces meet at more than `feature_angle`; nodes where feature edges end, branch, or turn are fixed. Surface nodes are limited to `surface_deviation` off and `surface_travel` along their starting tangent plane, as fractions of the mean incident edge length; feature edge nodes the reverse.
- Optimization pushes surface nodes outward: at `surface_deviation` 0.1 the real menisci swell 5-7%. `preserve_volume` removes the area-weighted mean normal offset in the projection, which keeps volume within 0.5% at the same quality.
- Smart Laplacian is a Jacobi sweep that undoes moves around any element that got worse, repeating until none did. It untangles the inverted test grid but barely changes gmsh's transfinite meshes; the optimization does most of the work there.
- High-order nodes move by trilinear interpolation of their element's corner displacements, using the local node coordinates gmsh reports, so curved edges keep their offsets. Quality is measured on corners only.
- On the real menisci at 4 by 20 elements, 1 / SJ optimization takes about 2 s and raises the minimum scaled Jacobian from 0.08-0.17 to 0.36-0.40 and the mean from 0.64-0.72 to 0.80-0.83. 1000 steps instead of 200 add little.

### Cartilage sheets

The lateral and medial tibial and the patellar cartilage are sheets: `classify_sides` splits the smoothed surface into its two sides, `fit_sheet` fits them as heights over the plane normal to the sides' thinnest principal direction, and `sheet_to_brep` or `sheet_to_hex_blocks` builds the solid. On the real labels, under 1% of the side points face against that normal, all near the rim, so heights over one plane suffice. The femoral cartilage wraps the condyles and is not a height field over any plane; it is still planned (see Femoral cartilage). Keep these points when changing this code:

- `classify_sides` compares each outward normal (`vtkPolyDataNormals` with `AutoOrientNormals` on the largest connected region) with the bone distance field's gradient, not with an axis, so it also works for the femoral cartilage. On the real labels the cosines are sharply bimodal. `DEFAULT_SIDE_COSINE` is 0.3: hexahedra stay valid there, and it keeps about 2% more volume than 0.5.
- Each side's footprint is the pixels within a closing distance of its points (4 median point spacings), shrunk back by that distance, so gaps narrower than that close. The sheet covers where both footprints overlap, so the steep rim between the sides is cut off. Regions inside the overlap's filled outline that neither side covers are defects.
- The perimeter is the overlap, moved `perimeter_inset` inward, closed and then opened by a disk of radius `perimeter_radius` from exact distance maps, and smoothed by a Gaussian of half that radius. Rolling the disk bounds the radius of curvature, concave and convex; the Gaussian alone left the real lateral tibial cartilage's notch tighter than one hexahedron, which inverted elements across it. The final Gaussian loosens the bound by under 15%. The defaults are 3 mm for hexahedra and 1 mm for tetrahedra. At the default counts, the real lateral tibial cartilage's minimum scaled Jacobian is -0.02 at 2 mm, 0.24 at 2.5 mm, and 0.35 at 3 mm.
- Heights are linear interpolation (`griddata`) on a grid of half the point spacing, nearest beyond the points, smoothed by a 2-pixel Gaussian and fitted with `RectBivariateSpline`. The perimeter lies inside both sides' points, so heights inside it are interpolated, not extrapolated.
- Thickening lowers the bone side to the articular height minus a smooth maximum of the thickness and the minimum, `(h + m + sqrt((h - m)^2 + b^2)) / 2` with `b` half the minimum. It is at least the minimum everywhere and leaves the articular side untouched. Thickened cartilage may overlap its bone; bone processing will run downstream and adjust the bone, never the cartilage.
- Hexahedra never get holes; `sheet_to_hex_blocks` ignores them with a warning, and the pipeline passes `hole_thickness` 0 for `hex_blocks`. For tetrahedra, connected regions thinner than `hole_thickness` of the mean, or defects, become holes when their rounded outlines lie `perimeter_radius` inside the perimeter. Rounding makes holes narrower than the radius vanish. `sheet_to_brep` cuts each with a prism along the normal.
- Volume: on the real labels the B-reps hold 92-96% of the voxel volume at either element shape's defaults. The distance-field Gaussian and windowed sinc smoothing cost 3-6%, and cutting off the rim the rest. None of the real labels gets holes at the defaults.

### Sheet butterfly partition

`_butterfly_blocks` lays a butterfly over the plane domain, as for the meniscus's sections, and both B-rep builders use it: the tetrahedral solid sews the ten patches and the four outer walls into one shell. Each block is a 20 by 20 grid of plane points; its patches are B-spline surfaces through the heights over them, and its walls ruled surfaces along the normal between matching patch edges. Because the walls run exactly along the normal, the B-rep cannot fold if the plane grid does not. Keep these points when changing it:

- The split cost is the signed turning angle: convex turns near a split cost, concave turns draw it, because a split at a concave notch shares its reflex angle between two blocks. `butterfly_split_indices` takes these costs through its `turns` argument; the meniscus still uses unsigned turning angles.
- The inner block is fixed, a bilinear quadrilateral with straight sides and corners halfway (`_INNER_BLOCK_SCALE`) from the perimeter's centroid to the splits. Freeing its sides rounded its corners to nearly 180 degrees.
- The ring of outer blocks between the perimeter and the inner block starts from a Laplace grid, which folds at the inner block's corners (270-degree re-entrant corners of the ring). Winslow's equations then relax it, which untangles those corners and bends the spokes around the perimeter's concavities. Straight spokes with Coons patches folded the real lateral tibial cartilage's blocks at its notch.
- After every Winslow sweep the second ring row is moved onto the perimeter's inward normals at its own distance, so columns leave the perimeter at right angles. Winslow alone let them leave the real medial tibial cartilage's perimeter at about 150 degrees, which made its worst element's Frobenius aspect ratio 20 instead of about 4 for a box with its edge lengths. Constraining more rows did not pay: 3 rows helped the medial cartilage but hurt the lateral, and 6 or more folded.
- Winslow crowds the rows toward the inner block, more than tenfold along the real spokes, and cubic patches through such rows at uniform parameters overshoot and fold. `_even_columns` respaces each ring column evenly by arc length afterward. Every block is checked for folded cells, raising ValueError.
- Patches and walls interpolate their grid points at uniform parameters with clamped cubic B-splines, so shared edges match exactly as in the meniscus's blocks; inner block sides 2 and 3 run opposite to their neighbors' and still match geometrically. Each block is sewn separately and then all are fragmented. A fragment yielding more than five volumes means blocks overlap.
- `mesh.sheet.hexahedral_mesh` finds edge counts from topology alone: opposite edges of four-sided faces form four classes, two around (arcs 0 and 2, and arcs 1 and 3, each with the inner sides opposite them), one through the thickness, and one along the spokes. The inner block is the one volume sharing four faces; the class common to its four walls runs through the thickness. It accepts the meniscus's blocks too, since their topology is the same.
- On the real cartilage at the defaults (8 around, 4 radial, 3 through), the largest Frobenius aspect ratio is 3.8-16 before optimization and 2.9-5.4 after, and the minimum scaled Jacobian 0.12-0.48 before and 0.40-0.56 after, with means of 0.89-0.94 after. The remaining worst elements are on thin stretches of the rim, where the wall is 0.5 mm tall under elements 3 mm across; `perimeter_inset` 1 mm brought the medial cartilage's largest aspect ratio from 5.4 to 3.0 at the cost of 14% of its area.
- Measured and not adopted: the inner block scale barely matters, with 0.5 within a few percent of the best between 0.3 and 0.7 at every density, before and after optimization, so it is not optimized per sheet. Columns along the local surface normal instead of the plane's normal crossed and overlapped blocks on the real lateral cartilage; along the bisector of both sides' normals they helped the medial cartilage but lowered other minimum scaled Jacobians to 0.25. Grading elements toward the rim cannot help, since a flat element's Frobenius aspect ratio is smallest with its radial edge near its tangential edge.

### Femoral cartilage (planned, not implemented)

The femoral cartilage wraps the condyles and joins them through the trochlea, so it is not a height field over one plane. Reuse `classify_sides`, the thickening, and the rounded outlines. Its hexahedral partition will likely be a half butterfly on the trochlear groove glued to half butterflies on the medial and lateral condyles; the trochlear block edges must split where the condyles meet it and at the intercondylar notch, with matching element counts across each glued interface. Tetrahedra may use contours and a loft instead, but the cartilage changes topology across slices near the intercondylar notch, so the slicing direction needs care. The femur and tibia labels are cut off by the scan's field of view along the shaft, so bone B-reps will need a planar cut face there.

## Tests

Test files mirror the package: root modules have `tests/test_<module>.py`, and structure modules have `tests/geo/test_<structure>.py` and `tests/mesh/test_<structure>.py`. Duplicate basenames work because pytest runs with `--import-mode=importlib`. `tests/conftest.py` defines session-scoped fixtures that every test shares, so tests must not modify them in place:

- `meniscus_label` is a synthetic meniscus: a triangular cross-section swept 270° about the z axis, with the gap facing -x.
- `meniscus_centerline` is `centerline(meniscus_label)`. Use it instead of calling `centerline` on the same image again.
- `meniscus_distance_field`, `meniscus_mesh`, `meniscus_smoothed_mesh`, `meniscus_slices`, `meniscus_contours`, `meniscus_surface`, `meniscus_solid`, `meniscus_domed_solid`, `meniscus_hex_slices`, `meniscus_hex_blocks`, `meniscus_tetrahedra`, and `meniscus_hexahedra` are the later pipeline stages for the same label.
- `real_label` is the real lateral meniscus in `tests/dat/lateral_meniscus.nii`, and `real_medial_label` the medial meniscus of the same knee in `tests/dat/medial_meniscus.nii`. `tests/dat` is stored in Git LFS.
- `cartilage_segmentation` is a synthetic tibial cartilage (label 2) on a spherical tibia (label 1) of radius 40 mm whose top touches z = 0: an ellipse of semi-axes 15 and 10 mm, 3 mm thick at its center and tapering to its rim, with a Gaussian dip at x = -6 mm thinned to about 15% of the mean once fitted. `cartilage_defect_segmentation` adds a full-thickness defect of radius 2.5 mm at x = 6 mm. `cartilage_sides`, `cartilage_defect_sides`, `cartilage_sheet` (hexahedral defaults), `cartilage_hex_blocks`, `cartilage_hexahedra`, and `cartilage_solid` (tetrahedral defaults) are later stages. The bone side lies about 0.35 mm outside the analytic sphere, from voxelization.
- `real_cartilage` holds the `CartilageSides` of each real tibial and patellar cartilage, assembled with its bone by nearest neighbor, and `real_cartilage_hex_blocks` their five-block solids at the hexahedral defaults. Building the three takes about 10 s, so share these instead of building again.

Every file in `tests/dat` is one structure of the same knee, named for its structure key: the two menisci plus `femoral_cartilage.nii`, `lateral_tibial_cartilage.nii`, `medial_tibial_cartilage.nii`, `patellar_cartilage.nii`, `femur.nii`, `tibia.nii`, and `patella.nii`; the femur and femoral cartilage are not read yet. Each is a single label 1 in `sitkUInt8`, cropped to the label's bounding box (`sitk.LabelShapeStatisticsImageFilter`, with no margin). The grids share spacing and orientation and are offset by whole voxels, so nearest-neighbor assembly reproduces every label exactly. SimpleITK has no 1-bit pixel type and ITK does not write NIfTI's binary type, so `sitkUInt8` is the smallest.

The fixture's voxelized cross-section centroid is r = 16.43, z = -1.67, not the ideal triangle's r = 16.67, z = -1.33. Its bottom slice lies exactly on the triangle's base and counts as a whole voxel layer, which also makes its voxel volume about 12% above the analytic value. Compare against the voxelized values or use a suitable tolerance.
