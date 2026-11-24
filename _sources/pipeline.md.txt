# Pipeline

For the meniscus, {py:func}`kneemesh.pipeline.run_brep` turns a LabelMap into a B-rep in these steps:

1. {py:func}`~kneemesh.geo.meniscus.centerline` fits a cubic B-spline through the centers of the label's cross-sections.
2. {py:func}`~kneemesh.vol.signed_distance_field` computes a signed distance field whose zero level lies on the label's voxel edges, then smooths it with a Gaussian.
3. {py:func}`~kneemesh.vol.extract_isocontour` extracts a closed triangle mesh at a level of that field.
4. {py:func}`~kneemesh.vol.smooth_isocontour` smooths the mesh with a windowed sinc filter.
5. {py:func}`~kneemesh.geo.meniscus.slice_isocontour` cuts the mesh with planes normal to the centerline.
6. {py:func}`~kneemesh.geo.meniscus.contours_to_brep` lofts the slices into a capped solid, which tetrahedral meshing and swept hexahedra take, or {py:func}`~kneemesh.geo.meniscus.contours_to_hex_blocks` builds the five-block butterfly solid for structured hexahedra. {py:func}`kneemesh.mesh.meniscus.hexahedral_mesh` takes either: it sweeps quadrilaterals of the loft's middle cross-section along it, or meshes the blocks with transfinite hexahedra.

For the lateral and medial tibial and the patellar cartilage, each a sheet over its bone, it takes these steps:

1. {py:func}`~kneemesh.vol.signed_distance_field`, {py:func}`~kneemesh.vol.extract_isocontour`, and {py:func}`~kneemesh.vol.smooth_isocontour` extract the cartilage's smoothed surface as for the meniscus, and the bone's distance field too.
2. {py:func}`~kneemesh.cartilage.classify_sides` splits the surface into its articular side and the side against the bone, by the angle between each point's normal and the bone's distance gradient, dropping the steep rim between them.
3. {py:func}`~kneemesh.geo.sheet.fit_sheet` fits both sides as heights over the plane normal to the cartilage's thinnest direction, within a perimeter where the two sides overlap. Rolling a disk of radius `perimeter_radius` around the perimeter, inside and out, keeps it from curving more tightly than that radius. Regions thinner than `min_thickness` of the mean are thickened by lowering the bone side, and for tetrahedra, regions thinner than `hole_thickness`, or missing from the label, become holes.
4. {py:func}`~kneemesh.geo.sheet.sheet_to_brep` builds the solid: each side one B-spline surface over the domain, trimmed by the perimeter and any holes, and straight walls ruled between the sides' edges. The same solid serves tetrahedral and hexahedral meshing; `hex_blocks` only picks the defaults that suit hexahedra.

The femoral cartilage takes the same steps over a cylinder instead of a plane:

1. {py:func}`~kneemesh.geo.femur.fit_femoral_sheet` fits a {py:class}`~kneemesh.geo.sheet.CylinderFrame` to the side against the femur, along the direction its surface normals vary least, and fits both sides as distances from that axis over the cylinder unrolled. The unrolled domain is a U.
2. {py:func}`~kneemesh.geo.sheet.sheet_to_brep` builds its solid as for the other cartilage.

The bone under each cartilage must be in the segmentation. It tells the cartilage's sides apart, and its own surface is built too.

Both sides' surfaces take the domain coordinates as their parameters, so a point of one side and the point of the other at the same parameters lie on one straight column through the sheet, and the walls are made of such columns. {py:func}`kneemesh.mesh.sheet.hexahedral_mesh` relies on this to partition the sheet when it meshes it: it meshes one side with quadrilaterals by gmsh's Frontal-Delaunay algorithm, or its quasi-structured one, which follows a cross field aligned with the boundary, and runs each node's column to the other side, stacking hexahedra along the columns. {py:func}`kneemesh.mesh.sheet.sheet_partition_brep` builds the blocks the quadrilaterals' irregular nodes divide the sheet into, as a B-rep of one volume per block.

For the femur, tibia, and patella, {py:func}`kneemesh.pipeline.run_brep` writes a closed, triangulated surface rather than a B-rep:

1. {py:func}`~kneemesh.geo.bone.bone_mask` fills the label's cavities, and cuts shafts the scan cuts off ({py:func}`~kneemesh.geo.bone.find_shaft_ends`) shorter by `shaft_cut_fraction`, in the voxel map.
2. {py:func}`~kneemesh.vol.signed_distance_field`, {py:func}`~kneemesh.vol.extract_isocontour`, and {py:func}`~kneemesh.vol.smooth_isocontour` extract its smoothed surface.
3. {py:func}`~kneemesh.geo.bone.decimate_surface` reduces the surface to triangles about `triangle_size` across.

{py:func}`kneemesh.mesh.bone.surface_mesh` and {py:func}`kneemesh.mesh.bone.tetrahedral_mesh` load the surface into gmsh, which splits it into patches it can map to a plane and builds those maps, then mesh it with triangles or quadrilaterals, or the solid it encloses with tetrahedra graded from the cortical shell inward.

## Overclosure

Each structure is built from its own label, and smoothing, thickening, and fitting let neighbors overlap a little. On the example knee the medial meniscus reached 0.45 mm into its tibial cartilage, the menisci 0.1 mm into the femoral cartilage, the cartilages 0.13-0.15 mm into their bones, and the femoral cartilage 0.06-0.11 mm into the tibial and patellar cartilage. With `overclosure.enabled`, {py:func}`kneemesh.pipeline.run_brep` builds every structure, then {py:func}`~kneemesh.overclosure.resolve_overclosure` moves them apart, leaving `overclosure.clearance` between them, before writing any:

1. Cartilage is thin, so it never moves for a meniscus or bone. Where two cartilages overlap, both articular sides are lowered toward their bones by the overlap's depth, $d$, each by its share of the two local thicknesses:

   $$
   \delta_a = \frac{t_a}{t_a + t_b}\,(d + c), \qquad \delta_b = \frac{t_b}{t_a + t_b}\,(d + c),
   $$

   with $c$ the clearance. {py:func}`~kneemesh.geo.sheet.lower_articular` lowers each sheet and {py:func}`~kneemesh.geo.sheet.sheet_to_brep` rebuilds it, in rounds until neither reaches into the other.
2. Every other structure, in `overclosure.priority` order, is moved out of every structure ranked above it. A meniscus's contour points move within their slice planes and are lofted again as they were built ({py:func}`~kneemesh.overclosure.push_meniscus_out`), so its planar caps, clipped cut, and hexahedral blocks are kept. A bone's surface vertices move ({py:func}`~kneemesh.overclosure.push_surface_out`).

Overlaps are measured on each B-rep's outer boundary triangulated at 0.5 mm ({py:func}`~kneemesh.overclosure.brep_boundary`), both ways: a structure's points inside the other, and the other's points inside it, which catch overlap between a loft's contours or a bone's vertices. A point measured between them moves the contour points or vertices around it that face it. On the example knee every overlap ended within 0.006 mm, and the hexahedral meshes' minimum scaled Jacobians changed by at most 0.033.

## Distance field smoothing

By default, the Gaussian's central 95% spans three voxel edges along each axis. A Gaussian's central 95% is $\pm 1.96\,\sigma$, so on an axis with voxel spacing $h$

$$
\sigma = \frac{3h}{2 \times 1.96} \approx 0.77\,h .
$$

## Sets and assembly

Every mesh carries its boundary faces and named sets of them and of their nodes ({py:mod}`kneemesh.mesh_sets`). {py:func}`~kneemesh.mesh_sets.boundary_faces` finds the faces of the volume elements that belong to only one, and the meshers name groups of them by the B-rep faces gmsh meshed them on: {py:func}`kneemesh.mesh.meniscus.meniscus_face_groups` the meniscus's caps and loft, and {py:func}`kneemesh.mesh.sheet.sheet_face_groups` a cartilage sheet's two sides, telling the articular side by its normal, which agrees with its parameterization's. {py:func}`kneemesh.disk.write_mesh` writes the sets to `.vtu` and `.vtkhdf`, and {py:func}`kneemesh.disk.write_meshes`, for `assembled_mesh`, writes every structure's mesh as one, numbering nodes, elements, and sets on through the structures.
