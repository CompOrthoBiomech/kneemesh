# Usage

`kneemesh-assemble` combines single-label LabelMaps into one multilabel LabelMap. Two further pipelines, `kneemesh-brep` and `kneemesh-mesh`, take the structure as their first argument, then a JSON configuration, `--option value` overrides, or both. The structures are `meniscus`, `tibial-cartilage`, `femoral-cartilage`, `patellar-cartilage`, `tibia`, `femur`, and `patella`; only `meniscus` is implemented so far. Run `kneemesh-brep meniscus --help` to list every option and its default. Every option is a field of the structure's model, such as {py:class}`~kneemesh.config.MeniscusBrep` or {py:class}`~kneemesh.config.MeniscusMesh`, with underscores written as dashes.

```sh
kneemesh-brep meniscus --label-map meniscus.nii --output meniscus.brep --cap planar
kneemesh-brep meniscus --label-map knee.nii --labels '{"tibia": 1, "medial_meniscus": 3}' --name medial_meniscus
kneemesh-mesh meniscus --brep meniscus.brep --output meniscus.msh --max-size 1.0
```

Each run saves the configuration it used, so it can be repeated with `--config`, still naming the structure. Without `--config` it is saved as `settings<timestamp>.json` in the working directory. With `--config` and overrides, the changed configuration is saved beside the original as `<name>_modified.json`.

For a hexahedral mesh, build the B-rep with `--hex-blocks true --points-per-slice 10` and mesh it with `--element hexahedral`.

## Segmentations

When the structures are segmented in separate files, assemble them once with `kneemesh-assemble`, a pipeline configured by {py:class}`~kneemesh.config.LabelMapAssembly`:

```sh
kneemesh-assemble --label-maps '{"tibia": "tibia.nii", "medial_meniscus": "meniscus.nii"}' --output knee.nrrd
```

It writes `knee.nrrd` and its labels in `knee_labels.json` beside it, logging to `assemble<timestamp>.log`. Structures get the values 1, 2, ... in the order given.

Every B-rep pipeline reads a segmentation and builds the structure that `--name` names in it. Give the segmentation one of two ways:

- **A multilabel LabelMap** with `--label-map`, and `--labels` naming its values as JSON, such as `'{"tibia": 1, "medial_tibial_cartilage": 2, "medial_meniscus": 3}'`. Without `--labels`, the LabelMap is read as holding only the structure, with value 1. An assembled LabelMap is read with `--label-map knee.nrrd --labels "$(cat knee_labels.json)"`.
- **Single-label LabelMaps** with `--label-maps`, assembled within the run as `kneemesh-assemble` does, one file per structure by name, such as `'{"tibia": "tibia.nii", "medial_meniscus": "meniscus.nii"}'`. Every nonzero voxel of a file is its structure. {py:func}`~kneemesh.segmentation.assemble_label_maps` resamples them onto one grid, in the first file's orientation, with the smallest voxel spacing along each axis and a background margin of `--padding` voxels. `--interpolator` chooses `label_gaussian` (the default) or `nearest_neighbor`. Where structures overlap after resampling, the one listed later wins. `--assembled-label-map knee.nrrd` also writes the assembled LabelMap and `knee_labels.json`, as `kneemesh-assemble` does. {py:func}`~kneemesh.disk.read_segmentation` reads them in Python.

Cartilage structures also name the bone beneath them with `--bone`, which must be in the segmentation too.

```json
{
  "label_maps": {"tibia": "tibia.nii", "medial_meniscus": "medial_meniscus.nii"},
  "assembled_label_map": "knee.nrrd",
  "name": "medial_meniscus",
  "output": "medial_meniscus.brep"
}
```
