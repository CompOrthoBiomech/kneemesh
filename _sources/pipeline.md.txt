# Pipeline

For the meniscus, {py:func}`kneemesh.pipeline.run_brep` turns a LabelMap into a B-rep in these steps:

1. {py:func}`~kneemesh.geo.meniscus.centerline` fits a cubic B-spline through the centers of the label's cross-sections.
2. {py:func}`~kneemesh.vol.signed_distance_field` computes a signed distance field whose zero level lies on the label's voxel edges, then smooths it with a Gaussian.
3. {py:func}`~kneemesh.vol.extract_isocontour` extracts a closed triangle mesh at a level of that field.
4. {py:func}`~kneemesh.vol.smooth_isocontour` smooths the mesh with a windowed sinc filter.
5. {py:func}`~kneemesh.geo.meniscus.slice_isocontour` cuts the mesh with planes normal to the centerline.
6. {py:func}`~kneemesh.geo.meniscus.contours_to_brep` lofts the slices into a capped solid, or {py:func}`~kneemesh.geo.meniscus.contours_to_hex_blocks` builds the five-block butterfly solid for hexahedral meshing.

## Distance field smoothing

By default, the Gaussian's central 95% spans three voxel edges along each axis. A Gaussian's central 95% is $\pm 1.96\,\sigma$, so on an axis with voxel spacing $h$

$$
\sigma = \frac{3h}{2 \times 1.96} \approx 0.77\,h .
$$
