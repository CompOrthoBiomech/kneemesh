# GUI

`kneemesh-gui` runs every stage of the pipeline interactively in a browser: import labels, configure and build each structure's geometry, mesh it, and view the results in 3D. It is a [trame](https://kitware.github.io/trame/) app with an optional dependency:

```sh
pip install "kneemesh[gui]"
kneemesh-gui                                   # opens a browser tab
kneemesh-gui --server --port 8080              # serves without opening one
kneemesh-gui --config examples/knee_brep_hex.json   # starts from a configuration
```

The server runs the pipeline on its own machine, and the file dialogs browse that machine's files, so paths in exported configurations are paths there. Relative paths are relative to the directory the GUI was started in, as for the command-line pipelines.

## Layout

The drawer on the left holds a tree above a settings panel; drag the bar between them to resize them. The view on the right shows the structures, with the log of running jobs beneath it.

- **Settings** holds what applies to every structure: the output directory, the B-rep suffix (`output`), how several label files are assembled (`interpolator`, `padding`), where to write the composed segmentation (`assembled_label_map`), the default mesh element type, the mesh suffix, and `assembled_mesh`.
- **Labels** lists the imported labels. **Add Label** opens a file dialog for a `.nii`, `.nrrd`, or `.mhd` LabelMap. Each value in the file becomes a branch with a **Structure Name**, a **Label Value**, and an **Include Label** checkbox. A file holding one value takes the next label value not yet used, so single-label files of value 1 are numbered 1, 2, 3, ... as they are added; the values of a multilabel file keep their own where free. Structures are picked from a `<stem>_labels.json` beside the file, as {py:func}`~kneemesh.disk.write_segmentation` writes, or from a file named for its structure, such as `tibia.nii`.
- **Structures** lists the nine structures, each with **Geometry** and **Mesh**. Selecting one shows its settings below, with a button to build or mesh that structure alone. A structure's settings are its configuration section, so the two menisci share theirs, as do the tibial and patellar cartilage; the panel says which structures share them. A mesh page also sets the structure's own element type (`element_types`) and the geometry file it meshes, by default the one the geometry step writes. The icons beside a structure show whether its geometry and mesh are built, and the eye hides it in the view.

Every field's tooltip is its description from {py:class}`~kneemesh.config.LabelMapBrep` or {py:class}`~kneemesh.config.BrepMesh`, and invalid values are flagged as they are typed. Empty optional fields, shown as *auto* or *none*, take the pipeline's default.

## Running

The toolbar builds the geometry of every included structure, meshes every structure whose geometry exists, or shows geometry already written. Building composes the included labels into one segmentation: labels from one file are relabelled in place, keeping its grid, and labels from several files are assembled onto one grid as `kneemesh-assemble` does. A cartilage needs its bone among the included labels, though building the cartilage alone does not rebuild the bone.

Each job runs in a separate process, so the GUI stays responsive, the toolbar's stop button cancels it, and a crash in gmsh ends only the job. Outputs are written as the command-line pipelines write them. The view shows each structure's latest stage, its label, geometry, or mesh, or one stage for all of them, as the toolbar's selector says.

## Configurations

The toolbar imports a `kneemesh-brep`, `kneemesh-mesh`, or `kneemesh-assemble` JSON configuration, telling which by the fields it holds, and exports the settings as configurations for `kneemesh-brep` and `kneemesh-mesh`. A geometry configuration names the labels as `kneemesh-brep` reads them: `label_map` and `labels` when they all come from one file at their own values, or `label_maps` when each is the only value of its own file (`kneemesh-brep` then numbers them 1, 2, ... in order). Any other labels are written as the segmentation they compose, `<name>_segmentation.nrrd` beside the configuration, for it to read. A mesh configuration's `breps` are the geometry files of the included structures and of any structure given its own file.
