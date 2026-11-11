# Usage

Two command-line pipelines take a JSON configuration, `--option value` overrides, or both. Run either with `--help` to list every option and its default. Every option is a field of {py:class}`~meniscus2brep.config.BrepConfig` or {py:class}`~meniscus2brep.config.MeshConfig`, with underscores written as dashes.

```sh
meniscus2brep-brep --label-map knee.nii --output meniscus.brep --cap planar
meniscus2brep-mesh --brep meniscus.brep --output meniscus.msh --max-size 1.0
```

Each run saves the configuration it used, so it can be repeated with `--config`. Without `--config` it is saved as `settings<timestamp>.json` in the working directory. With `--config` and overrides, the changed configuration is saved beside the original as `<name>_modified.json`.

For a hexahedral mesh, build the B-rep with `--hex-blocks true --points-per-slice 10` and mesh it with `--element hexahedral`.
