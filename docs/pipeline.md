# Pipeline

For the meniscus, {py:func}`kneemesh.pipeline.run_brep` turns a LabelMap into a B-rep in these steps:

1. {py:func}`~kneemesh.geo.meniscus.centerline` fits a cubic B-spline through the centers of the label's cross-sections.
2. {py:func}`~kneemesh.vol.signed_distance_field` computes a signed distance field whose zero level lies on the label's voxel edges, then smooths it with a Gaussian.
3. {py:func}`~kneemesh.vol.extract_isocontour` extracts a closed triangle mesh at a level of that field.
4. {py:func}`~kneemesh.vol.smooth_isocontour` smooths the mesh with a windowed sinc filter.
5. {py:func}`~kneemesh.geo.meniscus.slice_isocontour` cuts the mesh with planes normal to the centerline.
6. {py:func}`~kneemesh.geo.meniscus.contours_to_brep` lofts the slices into a capped solid, or {py:func}`~kneemesh.geo.meniscus.contours_to_hex_blocks` builds the five-block butterfly solid for hexahedral meshing.

For the lateral and medial tibial and the patellar cartilage, each a sheet over its bone, it takes these steps:

1. {py:func}`~kneemesh.vol.signed_distance_field`, {py:func}`~kneemesh.vol.extract_isocontour`, and {py:func}`~kneemesh.vol.smooth_isocontour` extract the cartilage's smoothed surface as for the meniscus, and the bone's distance field too.
2. {py:func}`~kneemesh.cartilage.classify_sides` splits the surface into its articular side and the side against the bone, by the angle between each point's normal and the bone's distance gradient, dropping the steep rim between them.
3. {py:func}`~kneemesh.geo.sheet.fit_sheet` fits both sides as heights over the plane normal to the cartilage's thinnest direction, within a perimeter where the two sides overlap. Rolling a disk of radius `perimeter_radius` around the perimeter, inside and out, keeps it from curving more tightly than that radius. Regions thinner than `min_thickness` of the mean are thickened by lowering the bone side, and for tetrahedra, regions thinner than `hole_thickness`, or missing from the label, become holes.
4. {py:func}`~kneemesh.geo.sheet.sheet_to_brep` builds the solid for tetrahedral meshing, or {py:func}`~kneemesh.geo.sheet.sheet_to_hex_blocks` the five-block butterfly solid for hexahedral meshing, which never has holes.

The bone under each cartilage must be in the segmentation. It is used only to tell the cartilage's sides apart; bone B-reps are not implemented yet, so the bone is not built.

The butterfly's inner quadrilateral is fixed with corners halfway from the perimeter's centroid to the four split points. Between it and the perimeter, the four outer blocks form a ring of grid lines relaxed by Winslow's equations, which keep them from crossing in the perimeter's concavities and at the inner block's corners. Straight spokes instead fold the blocks at the real lateral tibial cartilage's notch.

## Distance field smoothing

By default, the Gaussian's central 95% spans three voxel edges along each axis. A Gaussian's central 95% is $\pm 1.96\,\sigma$, so on an axis with voxel spacing $h$

$$
\sigma = \frac{3h}{2 \times 1.96} \approx 0.77\,h .
$$
