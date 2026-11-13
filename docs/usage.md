# Usage

`kneemesh-assemble` combines single-label LabelMaps into one multilabel LabelMap. `kneemesh-brep` builds a B-rep of every structure in a segmentation, and `kneemesh-mesh` meshes each structure's B-rep. Each takes a JSON configuration, `--option value` overrides, or both, and `--help` lists every option and its default. Every option is a field of the command's model, {py:class}`~kneemesh.config.LabelMapAssembly`, {py:class}`~kneemesh.config.LabelMapBrep`, or {py:class}`~kneemesh.config.BrepMesh`, with underscores written as dashes. Settings for one kind of structure sit in a section, such as `meniscus`, and are given on the command line as `--meniscus.points-per-slice 10`.

The structure names, keys of `labels`, `label_maps`, and `breps`, pick each structure's pipeline: `lateral_meniscus`, `medial_meniscus`, `patellar_cartilage`, `femoral_cartilage`, `lateral_tibial_cartilage`, `medial_tibial_cartilage`, `femur`, `tibia`, and `patella`. The menisci and the tibial and patellar cartilage are implemented so far; naming any other structure fails before any work starts, except a bone named only for its cartilage.

Each structure's output is written to `<output-directory>/<structure><output>`, where `--output` is a suffix. B-rep suffixes end in `.brep`, `.brep` by default. Mesh suffixes default to `_{element_type}.msh`, with `{element_type}` replaced by `--element-type`: `hex8`, `hex20`, `hex27`, `tet4` (the default), or `tet10`. gmsh infers the mesh format from the extension.

```sh
kneemesh-brep --label-map meniscus.nii --labels '{"medial_meniscus": 1}' --meniscus.cap planar
kneemesh-brep --label-map knee.nii --labels '{"lateral_meniscus": 2, "medial_meniscus": 3}'
kneemesh-mesh --breps '{"medial_meniscus": "medial_meniscus.brep"}' --meniscus.max-size 1.0
```

The last command writes `medial_meniscus_tet4.msh`. Each run saves the configuration it used, so it can be repeated with `--config`. Without `--config` it is saved as `settings<timestamp>.json` in the working directory. With `--config` and overrides, the changed configuration is saved beside the original as `<name>_modified.json`. Section overrides change only the fields given, leaving the section's other values from `--config`.

For a hexahedral mesh, build the B-reps with `--meniscus.hex-blocks true --meniscus.points-per-slice 10` or `--cartilage.hex-blocks true` and mesh them with `--element-type hex8`, `hex20`, or `hex27`. Add `--meniscus.optimize true` to move the nodes afterward for better scaled Jacobians with {py:func}`~kneemesh.mesh_optimization.optimize_hexahedral_mesh`. Boundary nodes may then leave the B-rep's surface by up to `--meniscus.surface-deviation` times the local element size, 0.1 by default, without changing the volume overall; `--meniscus.surface-deviation 0` keeps them on its tangent planes.

## Segmentations

When the structures are segmented in separate files, assemble them once with `kneemesh-assemble`, a pipeline configured by {py:class}`~kneemesh.config.LabelMapAssembly`:

```sh
kneemesh-assemble --label-maps '{"lateral_meniscus": "lateral.nii", "medial_meniscus": "medial.nii"}' --output knee.nrrd
```

It writes `knee.nrrd` and its labels in `knee_labels.json` beside it, logging to `assemble<timestamp>.log`. Structures get the values 1, 2, ... in the order given.

`kneemesh-brep` reads a segmentation and builds every structure named in it. Give the segmentation one of two ways:

- **A multilabel LabelMap** with `--label-map`, and `--labels` naming its values as JSON, such as `'{"lateral_meniscus": 2, "medial_meniscus": 3}'`. `--labels` is required, and values it does not name are ignored. An assembled LabelMap is read with `--label-map knee.nrrd --labels "$(cat knee_labels.json)"`.
- **Single-label LabelMaps** with `--label-maps`, assembled within the run as `kneemesh-assemble` does, one file per structure by name, such as `'{"lateral_meniscus": "lateral.nii", "medial_meniscus": "medial.nii"}'`. Every nonzero voxel of a file is its structure. {py:func}`~kneemesh.segmentation.assemble_label_maps` resamples them onto one grid, in the first file's orientation, with the smallest voxel spacing along each axis and a background margin of `--padding` voxels. `--interpolator` chooses `label_gaussian` (the default) or `nearest_neighbor`. Where structures overlap after resampling, the one listed later wins. `--assembled-label-map knee.nrrd` also writes the assembled LabelMap and `knee_labels.json`, as `kneemesh-assemble` does. {py:func}`~kneemesh.disk.read_segmentation` reads them in Python.

Every structure in the segmentation is built, so name only implemented structures for now. A cartilage needs the bone beneath it in the segmentation too: `patella` for `patellar_cartilage`, `femur` for `femoral_cartilage`, and `tibia` for either tibial cartilage. Bone B-reps are not implemented yet, so a bone named only for its cartilage is used to tell the cartilage's sides apart and is not built.

```json
{
  "label_maps": {"lateral_meniscus": "lateral_meniscus.nii", "medial_meniscus": "medial_meniscus.nii"},
  "assembled_label_map": "knee.nrrd",
  "output": "_hex.brep",
  "meniscus": {"points_per_slice": 10, "hex_blocks": true}
}
```

## Cartilage

The tibial and patellar cartilage are built as sheets between their articular side and the side against their bone. Their `cartilage` settings pick defaults by `hex_blocks`. For hexahedra, the perimeter curves no more tightly than a 3 mm radius, so elements can follow the real lateral tibial cartilage's notch, and regions thinner than 0.2 of the mean thickness are thickened by lowering the bone side. For tetrahedra, the perimeter follows the label more closely, with a 1 mm radius, regions thinner than 0.1 of the mean are thickened, and those thinner than 0.05 of the mean, or missing from the label, become holes. Thickened cartilage may overlap its bone.

```sh
kneemesh-brep --label-maps '{"tibia": "tibia.nii", "lateral_tibial_cartilage": "lateral.nii"}' --cartilage.hex-blocks true --output _hex.brep
kneemesh-mesh --breps '{"lateral_tibial_cartilage": "lateral_tibial_cartilage_hex.brep"}' --element-type hex8 --cartilage.optimize true
```

`--cartilage.elements-around`, `--cartilage.elements-radial`, and `--cartilage.elements-through` set the hexahedra along each quarter of the perimeter, along each spoke toward the inner block, and through the thickness. Optimizing cartilage hexahedra also weighs their Frobenius aspect ratios, by `--cartilage.aspect-weight`, 0.5 by default: optimizing the scaled Jacobian alone squares up corners by stretching elements, up to 75 times their thickness on the real medial tibial cartilage. The thinnest stretches of the rim hold the worst elements; `--cartilage.perimeter-inset` cuts them off at the cost of area.
