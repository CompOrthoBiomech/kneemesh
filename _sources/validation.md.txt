# Validation

kneemesh is developed against one knee, the structures in `tests/dat`. To see how it generalizes, `examples/validation/all_oks.py` runs the full assembly, B-rep, and mesh pipeline on every structure of the [OpenKnee(s)](https://simtk.org/projects/openknee) subjects' segmentations, which `examples/validation/crop_oks.py` crops to their labels and stores as `sitkUInt8`, as `tests/dat` is. The script runs locally, since the data is not distributed with kneemesh, and writes this page.

Generated 2026-11-18 09:50 with kneemesh at commit [`d6b71754ad5c58cbd431abc7736243bb7a51edcd`](https://github.com/CompOrthoBiomech/kneemesh/commit/d6b71754ad5c58cbd431abc7736243bb7a51edcd), shapes hex, tet, 8 jobs, timeout 3600 s per step, 173 s in all.

Settings follow `examples/knee_*.json`; each subject's output directory holds the configurations to repeat it with the console scripts. `SJ` is the scaled Jacobian (1 ideal, <= 0 inverted), measured on corner nodes; `raw` is before hexahedral optimization. `aspect` is the Frobenius aspect ratio. `warp` is the boundary quadrilaterals' warp in degrees. `vol` is the mesh's volume over the assembled label's voxel volume; bones are cut at their shafts, so theirs is not comparable. Flags are heuristic: inverted elements, minimum SJ below 0.2 (hex) or 0.05 (tet), vol outside 0.85-1.08, or an expected set with no faces.

## Overview

Each cell is the minimum scaled Jacobian, or the failed stage.

| subject | femur | tibia | patella | lateral_meniscus hex8 | lateral_meniscus tet4 | medial_meniscus hex8 | medial_meniscus tet4 | lateral_tibial_cartilage hex8 | lateral_tibial_cartilage tet4 | medial_tibial_cartilage hex8 | medial_tibial_cartilage tet4 | patellar_cartilage hex8 | patellar_cartilage tet4 | femoral_cartilage hex8 | femoral_cartilage tet4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oks001 | 0.26 | 0.3 | 0.26 | 0.46 | 0.18 | 0.56 | 0.18 | 0.63 | 0.17 | 0.68 | 0.11 | 0.71 | 0.13 | 0.38 | 0.086 |
| oks002 | 0.27 | 0.27 | 0.26 | 0.55 | 0.15 | 0.25 | 0.16 | 0.67 | 0.049 ⚠ low-SJ | 0.75 | 0.078 | 0.6 | 0.15 | 0.4 | 0.083 |
| oks004 | 0.21 | 0.26 | 0.27 | 0.61 ⚠ volume | 0.17 ⚠ volume | 0.57 | 0.17 | 0.69 | 0.092 | 0.79 | 0.17 | 0.42 | 0.1 | 0.61 | 0.12 |
| oks006 | 0.31 | 0.23 | 0.27 | 0.63 | 0.18 | 0.46 | 0.16 | 0.71 | 0.15 | 0.73 | 0.17 | 0.36 | 0.13 | 0.53 | 0.17 |
| oks007 | 0.18 | 0.23 | 0.27 | 0.51 | 0.16 | 0.48 ⚠ volume | 0.19 | 0.66 | 0.15 | 0.66 | 0.17 | 0.61 | 0.16 | 0.35 | 0.088 |
| oks008 | 0.29 | 0.22 | 0.23 | 0.39 | 0.17 | 0.22 | 0.16 | 0.41 | 0.12 | 0.73 | 0.13 | 0.47 | 0.18 | 0.48 | 0.096 |
| oks009 | 0.28 | 0.24 | 0.28 | 0.45 | 0.18 | 0.34 | 0.15 | 0.58 | 0.064 | 0.47 | 0.17 | 0.67 | 0.13 | 0.53 | 0.048 ⚠ low-SJ |

105 of 105 structure meshes succeeded; 5 of them flagged.

## By structure

Values are min / median / max over the subjects that succeeded.

| structure | type | ok | raw min SJ | min SJ | mean SJ | max aspect | warp 99% | vol |
|---|---|---|---|---|---|---|---|---|
| femur | tet4 | 7/7 | - | 0.179 / 0.27 / 0.313 | 0.683 / 0.683 / 0.686 | 1.79 / 1.86 / 2.26 | - | 0.779 / 0.802 / 0.823 |
| tibia | tet4 | 7/7 | - | 0.222 / 0.245 / 0.3 | 0.682 / 0.685 / 0.686 | 1.81 / 1.91 / 2.12 | - | 0.811 / 0.833 / 0.996 |
| patella | tet4 | 7/7 | - | 0.232 / 0.267 / 0.281 | 0.668 / 0.673 / 0.681 | 1.74 / 1.84 / 1.98 | - | 0.986 / 0.987 / 0.991 |
| lateral_meniscus | hex8 | 7/7 | 0.261 / 0.299 / 0.401 | 0.39 / 0.508 / 0.626 | 0.873 / 0.899 / 0.926 | 2.48 / 3.22 / 4.07 | 6.95 / 9.06 / 9.67 | 0.823 / 0.875 / 0.913 |
| lateral_meniscus | tet4 | 7/7 | - | 0.152 / 0.171 / 0.18 | 0.599 / 0.604 / 0.606 | 2.5 / 2.67 / 3.24 | - | 0.848 / 0.903 / 0.928 |
| medial_meniscus | hex8 | 7/7 | 0.195 / 0.265 / 0.341 | 0.218 / 0.462 / 0.566 | 0.819 / 0.871 / 0.893 | 3.64 / 4.11 / 4.65 | 6.85 / 8.61 / 13.1 | 0.82 / 0.873 / 0.914 |
| medial_meniscus | tet4 | 7/7 | - | 0.147 / 0.16 / 0.194 | 0.588 / 0.601 / 0.618 | 2.54 / 2.65 / 2.85 | - | 0.86 / 0.894 / 0.942 |
| lateral_tibial_cartilage | hex8 | 7/7 | 0.23 / 0.426 / 0.526 | 0.409 / 0.66 / 0.713 | 0.951 / 0.967 / 0.974 | 2.82 / 3.11 / 3.52 | 6.62 / 9.07 / 13.1 | 0.872 / 0.94 / 0.968 |
| lateral_tibial_cartilage | tet4 | 7/7 | - | 0.0495 / 0.123 / 0.17 | 0.576 / 0.589 / 0.598 | 2.56 / 2.98 / 5.24 | - | 0.881 / 0.945 / 0.968 |
| medial_tibial_cartilage | hex8 | 7/7 | 0.199 / 0.511 / 0.711 | 0.471 / 0.733 / 0.789 | 0.958 / 0.963 / 0.975 | 2.83 / 3.39 / 3.82 | 8.74 / 11 / 13.8 | 0.893 / 0.928 / 0.957 |
| medial_tibial_cartilage | tet4 | 7/7 | - | 0.0782 / 0.165 / 0.175 | 0.56 / 0.579 / 0.6 | 2.69 / 3.1 / 3.65 | - | 0.902 / 0.937 / 0.962 |
| patellar_cartilage | hex8 | 7/7 | 0.263 / 0.392 / 0.581 | 0.357 / 0.595 / 0.709 | 0.923 / 0.934 / 0.965 | 2.51 / 3.02 / 4.18 | 8.42 / 10.3 / 13.5 | 0.866 / 0.937 / 0.961 |
| patellar_cartilage | tet4 | 7/7 | - | 0.102 / 0.133 / 0.18 | 0.579 / 0.596 / 0.598 | 2.55 / 2.72 / 3.78 | - | 0.882 / 0.938 / 0.963 |
| femoral_cartilage | hex8 | 7/7 | 0.204 / 0.3 / 0.371 | 0.353 / 0.477 / 0.615 | 0.916 / 0.942 / 0.956 | 3.68 / 3.98 / 7.6 | 10.1 / 10.9 / 13.9 | 0.933 / 0.955 / 0.974 |
| femoral_cartilage | tet4 | 7/7 | - | 0.0482 / 0.0879 / 0.17 | 0.559 / 0.588 / 0.643 | 2.73 / 4.27 / 8.05 | - | 0.929 / 0.954 / 0.971 |

## Failures

None.

## Subjects

### oks001

Assembled onto a 279 x 606 x 192 grid at spacing [0.3516, 0.3516, 0.5] in 1 s, from inputs at spacings (0.3516, 0.3516, 0.7), (0.5, 0.5, 0.5).

| structure | type | status | s brep/mesh | nodes | elements | raw min SJ | min SJ | 1% SJ | mean SJ | inverted | max aspect | warp 99% | vol | warnings | flags / error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| femur | tet4 | ok | 5.33 / 9.06 | 13099 | 57805 | - | 0.26 | 0.425 | 0.685 | 0 | 1.83 | - | 0.816 | 0 |  |
| tibia | tet4 | ok | 4.23 / 5.84 | 9122 | 39955 | - | 0.3 | 0.426 | 0.686 | 0 | 1.82 | - | 0.82 | 0 |  |
| patella | tet4 | ok | 2.75 / 1.1 | 1883 | 7528 | - | 0.263 | 0.384 | 0.668 | 0 | 1.84 | - | 0.989 | 0 |  |
| lateral_meniscus | hex8 | ok | 3.95 / 17.1 | 8099 | 7200 | 0.347 | 0.458 | 0.719 | 0.895 | 0 | 4.07 | 9.06 | 0.9 | 0 |  |
| lateral_meniscus | tet4 | ok | 2.37 / 2.58 | 16160 | 74070 | - | 0.18 | 0.275 | 0.606 | 0 | 2.56 | - | 0.924 | 0 |  |
| medial_meniscus | hex8 | ok | 4.09 / 21.6 | 10336 | 9375 | 0.267 | 0.564 | 0.711 | 0.891 | 0 | 4.11 | 8.61 | 0.909 | 0 |  |
| medial_meniscus | tet4 | ok | 2.57 / 1.8 | 9472 | 43032 | - | 0.181 | 0.274 | 0.603 | 0 | 2.65 | - | 0.922 | 0 |  |
| lateral_tibial_cartilage | hex8 | ok | 3.97 / 14.1 | 10050 | 7848 | 0.333 | 0.626 | 0.837 | 0.974 | 0 | 3.14 | 8.54 | 0.968 | 0 |  |
| lateral_tibial_cartilage | tet4 | ok | 3.28 / 1.22 | 4976 | 19742 | - | 0.17 | 0.273 | 0.595 | 0 | 2.83 | - | 0.968 | 0 |  |
| medial_tibial_cartilage | hex8 | ok | 3.43 / 9.96 | 10050 | 7848 | 0.516 | 0.682 | 0.835 | 0.961 | 0 | 3.51 | 11 | 0.957 | 0 |  |
| medial_tibial_cartilage | tet4 | ok | 3.14 / 0.973 | 4588 | 17282 | - | 0.113 | 0.265 | 0.588 | 0 | 3.59 | - | 0.962 | 0 |  |
| patellar_cartilage | hex8 | ok | 3.3 / 14 | 9510 | 7424 | 0.581 | 0.709 | 0.85 | 0.965 | 0 | 2.7 | 8.94 | 0.937 | 0 |  |
| patellar_cartilage | tet4 | ok | 2.92 / 1.12 | 4989 | 20375 | - | 0.133 | 0.276 | 0.597 | 0 | 2.55 | - | 0.938 | 0 |  |
| femoral_cartilage | hex8 | ok | 6.37 / 97.5 | 14935 | 11336 | 0.204 | 0.378 | 0.752 | 0.956 | 0 | 7.6 | 10.9 | 0.974 | 0 |  |
| femoral_cartilage | tet4 | ok | 5.93 / 2.12 | 4300 | 13269 | - | 0.0861 | 0.241 | 0.559 | 0 | 4.84 | - | 0.971 | 0 |  |

<details><summary>warnings</summary>

- `oks001/assemble/all/-`: 42 assembled voxels of 'medial_meniscus' overlap earlier structures and take its label
- `oks001/assemble/all/-`: 2457 assembled voxels of 'lateral_tibial_cartilage' overlap earlier structures and take its label
- `oks001/assemble/all/-`: 3247 assembled voxels of 'medial_tibial_cartilage' overlap earlier structures and take its label
- `oks001/assemble/all/-`: 157 assembled voxels of 'patellar_cartilage' overlap earlier structures and take its label
- `oks001/assemble/all/-`: 9414 assembled voxels of 'femoral_cartilage' overlap earlier structures and take its label

</details>

### oks002

Assembled onto a 260 x 520 x 124 grid at spacing [0.3516, 0.3516, 0.7] in 1 s, from inputs at spacings (0.3516, 0.3516, 0.7).

| structure | type | status | s brep/mesh | nodes | elements | raw min SJ | min SJ | 1% SJ | mean SJ | inverted | max aspect | warp 99% | vol | warnings | flags / error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| femur | tet4 | ok | 2.59 / 6.05 | 9155 | 40063 | - | 0.27 | 0.419 | 0.683 | 0 | 1.85 | - | 0.793 | 0 |  |
| tibia | tet4 | ok | 2.41 / 5.09 | 7649 | 33354 | - | 0.269 | 0.422 | 0.684 | 0 | 1.91 | - | 0.833 | 0 |  |
| patella | tet4 | ok | 1.3 / 1.03 | 1446 | 5728 | - | 0.264 | 0.383 | 0.671 | 0 | 1.81 | - | 0.986 | 0 |  |
| lateral_meniscus | hex8 | ok | 2.26 / 15.4 | 8160 | 7375 | 0.299 | 0.555 | 0.679 | 0.923 | 0 | 3.29 | 9.46 | 0.875 | 0 |  |
| lateral_meniscus | tet4 | ok | 1.21 / 1.54 | 8792 | 39250 | - | 0.152 | 0.277 | 0.604 | 0 | 3.24 | - | 0.89 | 0 |  |
| medial_meniscus | hex8 | ok | 2.34 / 11.4 | 5429 | 4800 | 0.243 | 0.251 | 0.463 | 0.852 | 0 | 3.85 | 9.97 | 0.865 | 0 |  |
| medial_meniscus | tet4 | ok | 1.07 / 1.49 | 8430 | 37966 | - | 0.16 | 0.274 | 0.601 | 0 | 2.66 | - | 0.888 | 0 |  |
| lateral_tibial_cartilage | hex8 | ok | 1.99 / 6.99 | 6340 | 4920 | 0.426 | 0.668 | 0.824 | 0.967 | 0 | 3.11 | 9.07 | 0.94 | 0 |  |
| lateral_tibial_cartilage | tet4 | ok | 1.91 / 0.861 | 2782 | 10324 | - | 0.0495 | 0.255 | 0.585 | 0 | 4.27 | - | 0.945 | 0 | low-SJ |
| medial_tibial_cartilage | hex8 | ok | 1.99 / 1.78 | 2765 | 2048 | 0.711 | 0.748 | 0.836 | 0.975 | 0 | 3.82 | 13.8 | 0.907 | 0 |  |
| medial_tibial_cartilage | tet4 | ok | 1.55 / 0.795 | 2642 | 8475 | - | 0.0782 | 0.27 | 0.576 | 0 | 3.46 | - | 0.915 | 0 |  |
| patellar_cartilage | hex8 | ok | 1.57 / 2.83 | 2615 | 1936 | 0.392 | 0.595 | 0.737 | 0.933 | 0 | 3.76 | 13.4 | 0.866 | 0 |  |
| patellar_cartilage | tet4 | ok | 1.85 / 1.03 | 2590 | 8546 | - | 0.151 | 0.274 | 0.596 | 0 | 3.78 | - | 0.882 | 0 |  |
| femoral_cartilage | hex8 | ok | 4.19 / 29.9 | 11790 | 8904 | 0.344 | 0.398 | 0.769 | 0.941 | 0 | 4.29 | 13.9 | 0.94 | 0 |  |
| femoral_cartilage | tet4 | ok | 3.95 / 1.8 | 2885 | 8201 | - | 0.083 | 0.236 | 0.616 | 0 | 4.86 | - | 0.939 | 0 |  |

<details><summary>warnings</summary>

- `oks002/assemble/all/-`: 216 assembled voxels of 'lateral_tibial_cartilage' overlap earlier structures and take its label
- `oks002/assemble/all/-`: 111 assembled voxels of 'medial_tibial_cartilage' overlap earlier structures and take its label
- `oks002/assemble/all/-`: 183 assembled voxels of 'patellar_cartilage' overlap earlier structures and take its label
- `oks002/assemble/all/-`: 2034 assembled voxels of 'femoral_cartilage' overlap earlier structures and take its label

</details>

### oks004

Assembled onto a 236 x 519 x 114 grid at spacing [0.3516, 0.3516, 0.7] in 1 s, from inputs at spacings (0.3516, 0.3516, 0.7).

| structure | type | status | s brep/mesh | nodes | elements | raw min SJ | min SJ | 1% SJ | mean SJ | inverted | max aspect | warp 99% | vol | warnings | flags / error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| femur | tet4 | ok | 2.58 / 5.93 | 8644 | 37662 | - | 0.211 | 0.418 | 0.683 | 0 | 1.86 | - | 0.823 | 0 |  |
| tibia | tet4 | ok | 1.95 / 4.17 | 6417 | 27742 | - | 0.258 | 0.42 | 0.683 | 0 | 1.81 | - | 0.876 | 0 |  |
| patella | tet4 | ok | 1.16 / 1.3 | 1518 | 6037 | - | 0.267 | 0.392 | 0.673 | 0 | 1.74 | - | 0.987 | 0 |  |
| lateral_meniscus | hex8 | ok | 1.8 / 4.99 | 3115 | 2720 | 0.401 | 0.61 | 0.704 | 0.922 | 0 | 2.7 | 7.88 | 0.823 | 0 | volume |
| lateral_meniscus | tet4 | ok | 1.03 / 1.2 | 5746 | 25557 | - | 0.171 | 0.275 | 0.601 | 0 | 2.5 | - | 0.848 | 0 | volume |
| medial_meniscus | hex8 | ok | 2.2 / 10.7 | 5874 | 5200 | 0.341 | 0.566 | 0.711 | 0.893 | 0 | 4.41 | 6.85 | 0.871 | 0 |  |
| medial_meniscus | tet4 | ok | 1.13 / 1.62 | 8290 | 36448 | - | 0.174 | 0.272 | 0.598 | 0 | 2.54 | - | 0.894 | 0 |  |
| lateral_tibial_cartilage | hex8 | ok | 1.65 / 2.23 | 2125 | 1560 | 0.457 | 0.691 | 0.792 | 0.957 | 0 | 3.52 | 10.2 | 0.872 | 0 |  |
| lateral_tibial_cartilage | tet4 | ok | 1.63 / 0.817 | 1734 | 5197 | - | 0.0916 | 0.252 | 0.576 | 0 | 5.24 | - | 0.881 | 0 |  |
| medial_tibial_cartilage | hex8 | ok | 1.61 / 10.7 | 7170 | 5576 | 0.536 | 0.789 | 0.839 | 0.963 | 0 | 3.79 | 10.9 | 0.893 | 0 |  |
| medial_tibial_cartilage | tet4 | ok | 1.64 / 1.01 | 2404 | 7327 | - | 0.165 | 0.257 | 0.575 | 0 | 3.65 | - | 0.902 | 0 |  |
| patellar_cartilage | hex8 | ok | 1.61 / 6.1 | 6560 | 5092 | 0.299 | 0.415 | 0.619 | 0.929 | 0 | 3.19 | 13.5 | 0.929 | 0 |  |
| patellar_cartilage | tet4 | ok | 1.57 / 0.802 | 2987 | 11130 | - | 0.102 | 0.265 | 0.579 | 0 | 3.75 | - | 0.936 | 0 |  |
| femoral_cartilage | hex8 | ok | 3.69 / 24.3 | 10565 | 7932 | 0.362 | 0.615 | 0.773 | 0.941 | 0 | 3.98 | 10.5 | 0.933 | 0 |  |
| femoral_cartilage | tet4 | ok | 3.7 / 1.62 | 2728 | 7713 | - | 0.117 | 0.237 | 0.643 | 0 | 3.45 | - | 0.929 | 0 |  |

<details><summary>warnings</summary>

- `oks004/assemble/all/-`: 5 assembled voxels of 'medial_meniscus' overlap earlier structures and take its label
- `oks004/assemble/all/-`: 80 assembled voxels of 'lateral_tibial_cartilage' overlap earlier structures and take its label
- `oks004/assemble/all/-`: 536 assembled voxels of 'medial_tibial_cartilage' overlap earlier structures and take its label
- `oks004/assemble/all/-`: 345 assembled voxels of 'patellar_cartilage' overlap earlier structures and take its label
- `oks004/assemble/all/-`: 1280 assembled voxels of 'femoral_cartilage' overlap earlier structures and take its label

</details>

### oks006

Assembled onto a 240 x 462 x 117 grid at spacing [0.3516, 0.3516, 0.7] in 1 s, from inputs at spacings (0.3516, 0.3516, 0.7).

Assembled over input voxel volume, where it changed by more than 1%: medial_meniscus 0.960.

| structure | type | status | s brep/mesh | nodes | elements | raw min SJ | min SJ | 1% SJ | mean SJ | inverted | max aspect | warp 99% | vol | warnings | flags / error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| femur | tet4 | ok | 2.38 / 5.88 | 9098 | 39701 | - | 0.313 | 0.418 | 0.683 | 0 | 1.91 | - | 0.807 | 0 |  |
| tibia | tet4 | ok | 2.27 / 4.34 | 7089 | 30776 | - | 0.225 | 0.423 | 0.686 | 0 | 2.12 | - | 0.996 | 0 |  |
| patella | tet4 | ok | 1.05 / 0.927 | 1475 | 5810 | - | 0.274 | 0.387 | 0.671 | 0 | 1.89 | - | 0.986 | 0 |  |
| lateral_meniscus | hex8 | ok | 2.38 / 10.3 | 5429 | 4800 | 0.393 | 0.626 | 0.783 | 0.926 | 0 | 2.48 | 6.95 | 0.873 | 0 |  |
| lateral_meniscus | tet4 | ok | 1.31 / 1.12 | 6395 | 27844 | - | 0.18 | 0.275 | 0.599 | 0 | 2.67 | - | 0.894 | 0 |  |
| medial_meniscus | hex8 | ok | 2.31 / 12.1 | 5874 | 5200 | 0.324 | 0.462 | 0.721 | 0.888 | 0 | 4.1 | 8.03 | 0.873 | 0 |  |
| medial_meniscus | tet4 | ok | 1.03 / 0.941 | 4955 | 20351 | - | 0.16 | 0.267 | 0.588 | 0 | 2.85 | - | 0.891 | 0 |  |
| lateral_tibial_cartilage | hex8 | ok | 1.49 / 7.41 | 6125 | 4752 | 0.526 | 0.713 | 0.85 | 0.974 | 0 | 2.82 | 6.62 | 0.94 | 0 |  |
| lateral_tibial_cartilage | tet4 | ok | 1.7 / 0.842 | 2804 | 10559 | - | 0.15 | 0.265 | 0.589 | 0 | 2.98 | - | 0.945 | 0 |  |
| medial_tibial_cartilage | hex8 | ok | 1.65 / 8.8 | 7170 | 5576 | 0.474 | 0.734 | 0.843 | 0.964 | 0 | 3.39 | 8.74 | 0.917 | 0 |  |
| medial_tibial_cartilage | tet4 | ok | 1.46 / 0.989 | 2646 | 8634 | - | 0.168 | 0.261 | 0.56 | 0 | 3.1 | - | 0.923 | 0 |  |
| patellar_cartilage | hex8 | ok | 1.71 / 11.9 | 7405 | 5760 | 0.263 | 0.357 | 0.796 | 0.965 | 0 | 4.18 | 10.3 | 0.892 | 0 |  |
| patellar_cartilage | tet4 | ok | 1.7 / 0.879 | 2851 | 9557 | - | 0.128 | 0.277 | 0.582 | 0 | 3.4 | - | 0.903 | 0 |  |
| femoral_cartilage | hex8 | ok | 3.87 / 27.1 | 12745 | 9672 | 0.25 | 0.535 | 0.707 | 0.942 | 0 | 3.97 | 10.6 | 0.948 | 0 |  |
| femoral_cartilage | tet4 | ok | 4.09 / 1.98 | 3014 | 8755 | - | 0.17 | 0.27 | 0.642 | 0 | 2.73 | - | 0.943 | 0 |  |

<details><summary>warnings</summary>

- `oks006/assemble/all/-`: 8 assembled voxels of 'lateral_tibial_cartilage' overlap earlier structures and take its label
- `oks006/assemble/all/-`: 310 assembled voxels of 'medial_tibial_cartilage' overlap earlier structures and take its label
- `oks006/assemble/all/-`: 550 assembled voxels of 'patellar_cartilage' overlap earlier structures and take its label
- `oks006/assemble/all/-`: 358 assembled voxels of 'femoral_cartilage' overlap earlier structures and take its label

</details>

### oks007

Assembled onto a 243 x 520 x 130 grid at spacing [0.3516, 0.3516, 0.7] in 1 s, from inputs at spacings (0.3516, 0.3516, 0.7).

| structure | type | status | s brep/mesh | nodes | elements | raw min SJ | min SJ | 1% SJ | mean SJ | inverted | max aspect | warp 99% | vol | warnings | flags / error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| femur | tet4 | ok | 3 / 7.78 | 10390 | 45483 | - | 0.179 | 0.421 | 0.683 | 0 | 2.26 | - | 0.779 | 0 |  |
| tibia | tet4 | ok | 2.53 / 5.4 | 8246 | 35972 | - | 0.228 | 0.419 | 0.682 | 0 | 1.88 | - | 0.823 | 0 |  |
| patella | tet4 | ok | 1.13 / 1.52 | 1905 | 7675 | - | 0.272 | 0.393 | 0.674 | 0 | 1.84 | - | 0.987 | 0 |  |
| lateral_meniscus | hex8 | ok | 2.69 / 5.87 | 3172 | 2700 | 0.261 | 0.508 | 0.679 | 0.899 | 0 | 3.05 | 6.95 | 0.878 | 0 |  |
| lateral_meniscus | tet4 | ok | 1.35 / 1.07 | 6904 | 30468 | - | 0.165 | 0.276 | 0.6 | 0 | 3 | - | 0.923 | 0 |  |
| medial_meniscus | hex8 | ok | 2.23 / 7.73 | 3432 | 2925 | 0.265 | 0.475 | 0.56 | 0.871 | 0 | 3.64 | 6.88 | 0.82 | 0 | volume |
| medial_meniscus | tet4 | ok | 1.41 / 1.25 | 7725 | 34333 | - | 0.194 | 0.278 | 0.603 | 0 | 2.68 | - | 0.86 | 0 |  |
| lateral_tibial_cartilage | hex8 | ok | 1.98 / 2.06 | 2205 | 1600 | 0.517 | 0.66 | 0.775 | 0.951 | 0 | 2.83 | 13.1 | 0.931 | 0 |  |
| lateral_tibial_cartilage | tet4 | ok | 1.7 / 0.984 | 3135 | 11849 | - | 0.153 | 0.267 | 0.587 | 0 | 2.56 | - | 0.944 | 0 |  |
| medial_tibial_cartilage | hex8 | ok | 1.89 / 7.49 | 7640 | 5944 | 0.402 | 0.664 | 0.757 | 0.962 | 0 | 2.83 | 11.4 | 0.933 | 0 |  |
| medial_tibial_cartilage | tet4 | ok | 1.98 / 1.03 | 3292 | 11650 | - | 0.175 | 0.261 | 0.579 | 0 | 2.87 | - | 0.94 | 0 |  |
| patellar_cartilage | hex8 | ok | 2.32 / 9.15 | 8550 | 6664 | 0.439 | 0.608 | 0.793 | 0.96 | 0 | 2.57 | 9.01 | 0.946 | 0 |  |
| patellar_cartilage | tet4 | ok | 1.94 / 0.934 | 4768 | 19595 | - | 0.16 | 0.267 | 0.597 | 0 | 2.72 | - | 0.951 | 0 |  |
| femoral_cartilage | hex8 | ok | 4.75 / 43.1 | 15425 | 11776 | 0.3 | 0.353 | 0.746 | 0.95 | 0 | 3.75 | 11.2 | 0.956 | 0 |  |
| femoral_cartilage | tet4 | ok | 4.5 / 1.84 | 3518 | 10457 | - | 0.0879 | 0.25 | 0.562 | 0 | 3.85 | - | 0.955 | 0 |  |

<details><summary>warnings</summary>

- `oks007/assemble/all/-`: 4 assembled voxels of 'lateral_tibial_cartilage' overlap earlier structures and take its label
- `oks007/assemble/all/-`: 1 assembled voxels of 'medial_tibial_cartilage' overlap earlier structures and take its label
- `oks007/assemble/all/-`: 2 assembled voxels of 'patellar_cartilage' overlap earlier structures and take its label
- `oks007/assemble/all/-`: 39 assembled voxels of 'femoral_cartilage' overlap earlier structures and take its label

</details>

### oks008

Assembled onto a 272 x 520 x 139 grid at spacing [0.3516, 0.3516, 0.7] in 1 s, from inputs at spacings (0.3516, 0.3516, 0.7).

Assembled over input voxel volume, where it changed by more than 1%: medial_meniscus 0.989.

| structure | type | status | s brep/mesh | nodes | elements | raw min SJ | min SJ | 1% SJ | mean SJ | inverted | max aspect | warp 99% | vol | warnings | flags / error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| femur | tet4 | ok | 3.46 / 8.53 | 12457 | 55032 | - | 0.29 | 0.428 | 0.686 | 0 | 1.79 | - | 0.795 | 0 |  |
| tibia | tet4 | ok | 2.4 / 5.46 | 8868 | 38954 | - | 0.222 | 0.425 | 0.685 | 0 | 2.09 | - | 0.811 | 0 |  |
| patella | tet4 | ok | 1.38 / 1.68 | 2416 | 9974 | - | 0.232 | 0.398 | 0.681 | 0 | 1.98 | - | 0.991 | 0 |  |
| lateral_meniscus | hex8 | ok | 3 / 20.4 | 10200 | 9250 | 0.278 | 0.39 | 0.624 | 0.873 | 0 | 4.04 | 9.59 | 0.913 | 0 |  |
| lateral_meniscus | tet4 | ok | 1.51 / 1.89 | 11202 | 50419 | - | 0.168 | 0.277 | 0.604 | 0 | 2.67 | - | 0.928 | 0 |  |
| medial_meniscus | hex8 | ok | 2.84 / 14.7 | 6764 | 6000 | 0.195 | 0.218 | 0.374 | 0.819 | 0 | 4.65 | 13.1 | 0.914 | 0 |  |
| medial_meniscus | tet4 | ok | 1.13 / 1.76 | 11495 | 52054 | - | 0.159 | 0.273 | 0.601 | 0 | 2.6 | - | 0.942 | 0 |  |
| lateral_tibial_cartilage | hex8 | ok | 2.23 / 5.92 | 4015 | 3016 | 0.23 | 0.409 | 0.68 | 0.951 | 0 | 3.33 | 12 | 0.948 | 0 |  |
| lateral_tibial_cartilage | tet4 | ok | 2.54 / 1.04 | 4972 | 19202 | - | 0.123 | 0.272 | 0.598 | 0 | 2.68 | - | 0.951 | 0 |  |
| medial_tibial_cartilage | hex8 | ok | 2.21 / 10.7 | 10050 | 7848 | 0.511 | 0.733 | 0.826 | 0.958 | 0 | 3.15 | 8.99 | 0.942 | 0 |  |
| medial_tibial_cartilage | tet4 | ok | 1.73 / 0.907 | 4264 | 15111 | - | 0.132 | 0.273 | 0.6 | 0 | 2.69 | - | 0.947 | 0 |  |
| patellar_cartilage | hex8 | ok | 2.2 / 14.6 | 10805 | 8448 | 0.356 | 0.473 | 0.651 | 0.934 | 0 | 3.02 | 10.8 | 0.961 | 0 |  |
| patellar_cartilage | tet4 | ok | 2.15 / 1.47 | 5950 | 25168 | - | 0.18 | 0.268 | 0.598 | 0 | 2.55 | - | 0.963 | 0 |  |
| femoral_cartilage | hex8 | ok | 4.77 / 53.5 | 15915 | 12112 | 0.293 | 0.477 | 0.708 | 0.916 | 0 | 3.68 | 11 | 0.957 | 0 |  |
| femoral_cartilage | tet4 | ok | 4.49 / 1.81 | 3841 | 11192 | - | 0.0955 | 0.267 | 0.572 | 0 | 4.27 | - | 0.954 | 0 |  |

<details><summary>warnings</summary>

- `oks008/assemble/all/-`: 486 assembled voxels of 'lateral_tibial_cartilage' overlap earlier structures and take its label
- `oks008/assemble/all/-`: 2029 assembled voxels of 'medial_tibial_cartilage' overlap earlier structures and take its label
- `oks008/assemble/all/-`: 555 assembled voxels of 'patellar_cartilage' overlap earlier structures and take its label
- `oks008/assemble/all/-`: 110 assembled voxels of 'femoral_cartilage' overlap earlier structures and take its label

</details>

### oks009

Assembled onto a 272 x 518 x 135 grid at spacing [0.3516, 0.3516, 0.7] in 1 s, from inputs at spacings (0.3516, 0.3516, 0.7).

| structure | type | status | s brep/mesh | nodes | elements | raw min SJ | min SJ | 1% SJ | mean SJ | inverted | max aspect | warp 99% | vol | warnings | flags / error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| femur | tet4 | ok | 3.05 / 7.65 | 11644 | 51385 | - | 0.278 | 0.427 | 0.686 | 0 | 1.86 | - | 0.802 | 0 |  |
| tibia | tet4 | ok | 2.48 / 5.41 | 8448 | 37007 | - | 0.245 | 0.418 | 0.685 | 0 | 2.05 | - | 0.85 | 0 |  |
| patella | tet4 | ok | 1.47 / 1.59 | 2260 | 9173 | - | 0.281 | 0.401 | 0.675 | 0 | 1.8 | - | 0.99 | 0 |  |
| lateral_meniscus | hex8 | ok | 3.45 / 6.58 | 4160 | 3555 | 0.264 | 0.446 | 0.55 | 0.881 | 0 | 3.22 | 9.67 | 0.855 | 0 |  |
| lateral_meniscus | tet4 | ok | 1.68 / 2.13 | 14433 | 64804 | - | 0.18 | 0.278 | 0.605 | 0 | 2.58 | - | 0.903 | 0 |  |
| medial_meniscus | hex8 | ok | 3.08 / 6.64 | 3640 | 3105 | 0.221 | 0.335 | 0.563 | 0.868 | 0 | 4.19 | 12.6 | 0.886 | 0 |  |
| medial_meniscus | tet4 | ok | 1.28 / 4.66 | 29948 | 147895 | - | 0.147 | 0.28 | 0.618 | 0 | 2.58 | - | 0.938 | 0 |  |
| lateral_tibial_cartilage | hex8 | ok | 2.2 / 10.5 | 8300 | 6468 | 0.369 | 0.584 | 0.82 | 0.967 | 0 | 2.82 | 8.69 | 0.953 | 0 |  |
| lateral_tibial_cartilage | tet4 | ok | 2.29 / 0.967 | 3718 | 14339 | - | 0.0642 | 0.27 | 0.593 | 0 | 3.75 | - | 0.957 | 0 |  |
| medial_tibial_cartilage | hex8 | ok | 2.09 / 2.37 | 2775 | 2040 | 0.199 | 0.471 | 0.774 | 0.969 | 0 | 3.03 | 12.6 | 0.928 | 0 |  |
| medial_tibial_cartilage | tet4 | ok | 1.87 / 1.12 | 3216 | 11419 | - | 0.172 | 0.271 | 0.594 | 0 | 3.03 | - | 0.937 | 0 |  |
| patellar_cartilage | hex8 | ok | 2.29 / 8.31 | 9510 | 7424 | 0.521 | 0.666 | 0.77 | 0.923 | 0 | 2.51 | 8.42 | 0.954 | 0 |  |
| patellar_cartilage | tet4 | ok | 2.43 / 0.966 | 4843 | 19797 | - | 0.129 | 0.269 | 0.594 | 0 | 2.6 | - | 0.96 | 0 |  |
| femoral_cartilage | hex8 | ok | 4.87 / 19.8 | 15915 | 12112 | 0.371 | 0.529 | 0.764 | 0.943 | 0 | 4.21 | 10.1 | 0.955 | 0 |  |
| femoral_cartilage | tet4 | ok | 4.52 / 1.45 | 3806 | 11020 | - | 0.0482 | 0.253 | 0.588 | 0 | 8.05 | - | 0.955 | 0 | low-SJ |

<details><summary>warnings</summary>

- `oks009/assemble/all/-`: 3 assembled voxels of 'lateral_tibial_cartilage' overlap earlier structures and take its label
- `oks009/assemble/all/-`: 2 assembled voxels of 'medial_tibial_cartilage' overlap earlier structures and take its label
- `oks009/assemble/all/-`: 4 assembled voxels of 'patellar_cartilage' overlap earlier structures and take its label
- `oks009/assemble/all/-`: 5 assembled voxels of 'femoral_cartilage' overlap earlier structures and take its label

</details>
