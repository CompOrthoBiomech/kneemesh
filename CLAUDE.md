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
```

## Architecture

The package converts a segmented meniscus image into B-rep geometry, as a pipeline across `src/meniscus2brep/`:

- `disk.py` reads and writes LabelMaps.
- `vol.py` operates on LabelMaps. Its current output is the centerline spline.
- `geo.py` will generate the B-rep. It is an empty placeholder and no B-rep library has been chosen.
- `config.py` provides `BaseConfig`, the pydantic base class for configuration models parsed from JSON. It rejects unknown keys. No concrete configuration models exist yet.

### LabelMap convention

A LabelMap is a scalar `sitk.Image` with pixel type `sitk.sitkUInt8`, not SimpleITK's run-length `sitkLabel*` pixel types. `disk.py` enforces this on both read and write and does not cast other pixel types. It accepts only `.nii`, `.nrrd`, and `.mhd` + `.raw`, so `.nii.gz` is rejected. `vol.centerline` additionally requires a 3D image.

### How `vol.centerline` works

`sitk.BinaryThinning` only thins in 2D. Given a 3D image it thins slice by slice and returns a sheet, so it cannot produce the curve directly. It only provides the initial estimate:

1. Project the label along the image axis closest to its thinnest principal direction, then thin the 2D projection.
2. Take the longest path through the skeleton and trim one mean half-width from each end, because the skeleton forks into the label's corners there.
3. Fit a spline, then repeatedly replace it with a fit through centroids of cross-section slabs normal to the curve. Stations extend past both ends so the curve can grow back toward the tips, and partial sections at the tips are discarded.

Section spacing and smoothing tolerance scale with the structure's half-width as well as the voxel size. Tying them to the voxel size alone makes the curve hook at the tips on fine grids.

The result is a cubic `scipy.interpolate.BSpline` over u in [0, 1], in physical coordinates, and evaluating it at N parameters gives an `(N, 3)` array. `make_splprep` natively evaluates to `(3, N)`, so `_fit` rebuilds the spline to normalize this. Which end of the structure is u = 0 is unspecified, so compare curves by nearest-point distance rather than parameter by parameter.

## Tests

`tests/conftest.py` defines two session-scoped fixtures that every test shares, so tests must not modify them in place:

- `meniscus_label` is a synthetic meniscus: a triangular cross-section swept 270° about the z axis, with the gap facing -x.
- `meniscus_centerline` is `centerline(meniscus_label)`. Use it instead of calling `centerline` on the same image again.

The fixture's voxelized cross-section centroid is r = 16.43, z = -1.67, not the ideal triangle's r = 16.67, z = -1.33. Its bottom slice lies exactly on the triangle's base and counts as a whole voxel layer, which also makes its voxel volume about 12% above the analytic value. Compare against the voxelized values or use a suitable tolerance.
