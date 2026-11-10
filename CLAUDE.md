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
uv run meniscus2brep-brep --help               # LabelMap to B-rep pipeline
uv run meniscus2brep-mesh --help               # B-rep to mesh pipeline
```

## Architecture

The package converts a segmented meniscus image into B-rep geometry, as a pipeline across `src/meniscus2brep/`:

- `disk.py` reads and writes LabelMaps, `.vtp` poly data, and `.brep` files, and writes meshes in any volume format gmsh infers from the extension.
- `vol.py` operates on LabelMaps. It produces the centerline spline and the ordered cross-section contours.
- `geo.py` lofts contour slices into a B-rep with gmsh's Open Cascade kernel, builds the three-block solid that hexahedral meshing needs, and can assemble slices into one `vtkPolyData` of closed polylines.
- `mesh.py` meshes solid B-reps with gmsh into tetrahedra or block-structured hexahedra, returning a `Mesh` of node coordinates, per-type connectivity in gmsh's node ordering, and gmsh element type numbers. `write_mesh` rebuilds a `Mesh` as a discrete gmsh volume to write it.
- `config.py` provides `BaseConfig`, the pydantic base class for configuration models parsed from JSON, which rejects unknown keys. `BrepConfig` and `MeshConfig` hold every pipeline argument with its default. The gmsh algorithm enums are string enums named for readability, with gmsh's option number on `.number` and case-insensitive lookup, so configs store them as plain names. Optional numbers read "none" as None.
- `pipeline.py` runs the two console scripts. `configure` builds an option for every config field, layers command-line values over `--config`, and saves the result: `settings<timestamp>.json` in the working directory without `--config`, or `<stem>_modified.json` beside it when options change it. Command-line values arrive as strings and rely on pydantic's lax coercion.
- Logging uses loguru. `__init__.py` disables the `meniscus2brep` logger so the library stays silent; modules log step details at DEBUG. The pipelines enable it for the run, sending INFO to stderr and DEBUG to `brep<timestamp>.log` or `mesh<timestamp>.log` in the working directory, or to `<config stem>.log` beside `--config`. They remove their handlers and disable the logger again on exit. Use `logger`, not `print`.

### LabelMap convention

A LabelMap is a scalar `sitk.Image` with pixel type `sitkUInt8`, `sitkUInt16`, or `sitkInt16` (`vol.LABEL_MAP_PIXEL_IDS`), not SimpleITK's run-length `sitkLabel*` pixel types. `disk.py` enforces this on both read and write and keeps the pixel type rather than casting. It accepts only `.nii`, `.nrrd`, and `.mhd` + `.raw`, so `.nii.gz` is rejected. `vol.centerline` additionally requires a 3D image.

### How `vol.centerline` works

`sitk.BinaryThinning` only thins in 2D. Given a 3D image it thins slice by slice and returns a sheet, so it cannot produce the curve directly. It only provides the initial estimate:

1. Project the label along the image axis closest to its thinnest principal direction, then thin the 2D projection. The projection is padded with background first: thinning treats the image edge as an open end, so a label cut off by the edge would otherwise skeletonize into disconnected branches and the longest path would cover only part of the arc.
2. Take the longest path through the skeleton and trim one mean half-width from each end, because the skeleton forks into the label's corners there.
3. Fit a spline, then repeatedly replace it with a fit through centroids of cross-section slabs normal to the curve. Stations extend past both ends so the curve can grow back toward the tips, and partial sections at the tips are discarded.

Section spacing and smoothing tolerance scale with the structure's half-width as well as the voxel size. Tying them to the voxel size alone makes the curve hook at the tips on fine grids.

The result is a cubic `scipy.interpolate.BSpline` over u in [0, 1], in physical coordinates, and evaluating it at N parameters gives an `(N, 3)` array. `make_splprep` natively evaluates to `(3, N)`, so `_fit` rebuilds the spline to normalize this. Which end of the structure is u = 0 is unspecified, so compare curves by nearest-point distance rather than parameter by parameter.

### Cross-section contours

`signed_distance_field` feeds `extract_isocontour`, which returns a closed VTK triangle mesh from `vtkFlyingEdges3D`. `slice_isocontour` cuts that mesh along the centerline and returns `ContourSlice` objects holding ordered points from a closed Kochanek spline. Keep these points when changing this code:

- `sitk.SignedMaurerDistanceMap` alone is zero on boundary voxel centers, half a voxel inside the label. `signed_distance_field` averages it with the inverted mask's map so zero lies on the voxel edges.
- `signed_distance_field` then smooths with a recursive Gaussian to remove the voxel staircase. The default sigma, per axis, puts 3 voxel edges in the central 95%. It rounds sharp edges, costing the test fixture about 5% of its volume. Pass `sigma=0` for the exact field.
- `extract_isocontour` pads the field so the mesh stays closed where a label touches the image edge.
- A cutting plane is infinite and can also cut the far limb of the C-shaped structure, so each slice takes the loop nearest the centerline.
- Contours run counterclockwise about the slice normal. Each one's point order is rotated to match the previous slice, because a loft joins contours by parameter and a drifting start point makes the surface bulge at sharp corners.
- Slices are clipped only from the two ends, until one reaches 20% of the mean area. A narrow section between the ends is kept.

### B-reps through gmsh

gmsh holds its models in global state, so a `BRep` is a plain value holding the `.brep` file's text, and all gmsh work happens inside `gmsh_session()`. Keep these points when changing `contours_to_brep`:

- gmsh treats a B-spline whose first and last point tags match as periodic, so contours are built with periodic knots.
- `addThruSections(makeSolid=True)` returns a volume with no end faces and no error. Solids are instead built from the lofted surface, the two caps, and a sewn surface loop.
- A smooth cap reuses the end contour's poles and knots so its rim matches the loft's edge. Its quadratic meridians collapse to one apex by repeating the apex point tag for the last row of poles.
- Planar caps need contours projected exactly onto their planes, because VTK's single-precision points fall outside Open Cascade's tolerance.
- The loft's degree along the curve is capped at 3 regardless of `spline_order`, and only a ruled loft achieves degree 1.
- The loft uses `continuity="C1"`. Open Cascade's default C2 overshoots in the end spans where real sections taper toward the horns, folding the surface, and gmsh's 2D meshing then never finishes. `smoothing=True` in `addThruSections` hangs outright.
- `gmsh.write` exports only the highest-dimension shapes, so the input wires do not need removing.
- `occ.getMass` is unreliable on ruled surfaces (`spline_order=1`). Mesh the shape to check its volume.

### Hexahedral blocks

A single four-sided block can't map the roughly triangular meniscus section: one block corner lands on a smooth side at about 180°, and the cells there invert. `contours_to_hex_blocks` therefore splits each section into three blocks, at its three sharpest corners (turning angle averaged over all slices) and the midpoints between them, with straight spokes to the section centroid. Keep these points when changing it:

- It does not use `addThruSections`. Separate lofts choose their own parameterization along the curve, so adjacent lofts cross between slices and fragment shatters them. Instead every face is a B-spline surface interpolating section curves fitted with a shared knot vector, with each column of poles interpolated along the curve at shared parameters. A face's edges then depend only on their own points, so neighboring faces match exactly.
- Each block is sewn separately, then the blocks are fragmented, which merges the identical copies of shared spoke faces. Trimming faces with existing curves (`wireTags`) does not share edges, because gmsh projects new ones.
- `hexahedral_mesh` makes every curve, surface, and volume transfinite. Curves on the planar caps get `elements_per_edge` and all others `elements_along`.
- Hexahedral tests use 10-point contours, which give valid elements and small B-reps.
- gmsh's "initial mesh only" 2D algorithm crashes the process with a segmentation fault, and BAMG fails on these shapes, so `Algorithm2D` omits them. `Algorithm3D` omits R-tree, which writes `new_points.pos` into the working directory.

## Tests

`tests/conftest.py` defines session-scoped fixtures that every test shares, so tests must not modify them in place:

- `meniscus_label` is a synthetic meniscus: a triangular cross-section swept 270° about the z axis, with the gap facing -x.
- `meniscus_centerline` is `centerline(meniscus_label)`. Use it instead of calling `centerline` on the same image again.
- `meniscus_distance_field`, `meniscus_mesh`, `meniscus_slices`, `meniscus_contours`, `meniscus_surface`, `meniscus_solid`, `meniscus_domed_solid`, `meniscus_hex_slices`, and `meniscus_hex_blocks` are the later pipeline stages for the same label.

The fixture's voxelized cross-section centroid is r = 16.43, z = -1.67, not the ideal triangle's r = 16.67, z = -1.33. Its bottom slice lies exactly on the triangle's base and counts as a whole voxel layer, which also makes its voxel volume about 12% above the analytic value. Compare against the voxelized values or use a suitable tolerance.
