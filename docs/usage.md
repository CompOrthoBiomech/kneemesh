# Usage

Two command-line pipelines, `kneemesh-brep` and `kneemesh-mesh`, take the structure as their first argument, then a JSON configuration, `--option value` overrides, or both. The structures are `meniscus`, `tibial-cartilage`, `femoral-cartilage`, `patellar-cartilage`, `tibia`, `femur`, and `patella`; only `meniscus` is implemented so far. Run `kneemesh-brep meniscus --help` to list every option and its default. Every option is a field of the structure's model, such as {py:class}`~kneemesh.config.MeniscusBrep` or {py:class}`~kneemesh.config.MeniscusMesh`, with underscores written as dashes.

```sh
kneemesh-brep meniscus --label-map knee.nii --output meniscus.brep --cap planar
kneemesh-mesh meniscus --brep meniscus.brep --output meniscus.msh --max-size 1.0
```

Each run saves the configuration it used, so it can be repeated with `--config`, still naming the structure. Without `--config` it is saved as `settings<timestamp>.json` in the working directory. With `--config` and overrides, the changed configuration is saved beside the original as `<name>_modified.json`.

For a hexahedral mesh, build the B-rep with `--hex-blocks true --points-per-slice 10` and mesh it with `--element hexahedral`.
