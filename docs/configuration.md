# Configuration

Each pipeline reads one of these models from JSON with {py:meth}`~kneemesh.config.BaseConfig.from_json`. Unknown keys are rejected. {py:class}`~kneemesh.config.LabelMapBrep` and {py:class}`~kneemesh.config.BrepMesh` hold the settings every structure shares, and a section for each kind of implemented structure holds its own settings. Keys of `labels`, `label_maps`, and `breps` must be names from {py:data}`~kneemesh.config.Structure`.

```{eval-rst}
.. autopydantic_model:: kneemesh.config.LabelMapAssembly
   :noindex:

.. autopydantic_model:: kneemesh.config.LabelMapBrep
   :noindex:

.. autopydantic_model:: kneemesh.config.BrepMesh
   :noindex:
```

## Meniscus

The `meniscus` sections apply to both `lateral_meniscus` and `medial_meniscus`.

```{eval-rst}
.. autopydantic_model:: kneemesh.config.MeniscusBrep
   :noindex:

.. autopydantic_model:: kneemesh.config.MeniscusMesh
   :noindex:
```
