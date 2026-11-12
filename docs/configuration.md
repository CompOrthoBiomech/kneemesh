# Configuration

Each pipeline reads one of these models from JSON with {py:meth}`~kneemesh.config.BaseConfig.from_json`. Unknown keys are rejected. Every B-rep model extends {py:class}`~kneemesh.config.LabelMapBrep` and every meshing model extends {py:class}`~kneemesh.config.BrepMesh`.

```{eval-rst}
.. autopydantic_model:: kneemesh.config.LabelMapBrep
   :noindex:

.. autopydantic_model:: kneemesh.config.BrepMesh
   :noindex:

.. autopydantic_model:: kneemesh.config.CartilageBrep
   :noindex:

.. autopydantic_model:: kneemesh.config.MeniscusBrep
   :noindex:

.. autopydantic_model:: kneemesh.config.MeniscusMesh
   :noindex:
```

## Planned structures

These models hold only the shared settings until their pipelines are implemented. Cartilage models extend {py:class}`~kneemesh.config.CartilageBrep`, which names the underlying bone. Lateral and medial tibial cartilage share one strategy and are told apart by `name`.

```{eval-rst}
.. autopydantic_model:: kneemesh.config.TibialCartilageBrep
   :noindex:

.. autopydantic_model:: kneemesh.config.TibialCartilageMesh
   :noindex:

.. autopydantic_model:: kneemesh.config.FemoralCartilageBrep
   :noindex:

.. autopydantic_model:: kneemesh.config.FemoralCartilageMesh
   :noindex:

.. autopydantic_model:: kneemesh.config.PatellarCartilageBrep
   :noindex:

.. autopydantic_model:: kneemesh.config.PatellarCartilageMesh
   :noindex:

.. autopydantic_model:: kneemesh.config.TibiaBrep
   :noindex:

.. autopydantic_model:: kneemesh.config.TibiaMesh
   :noindex:

.. autopydantic_model:: kneemesh.config.FemurBrep
   :noindex:

.. autopydantic_model:: kneemesh.config.FemurMesh
   :noindex:

.. autopydantic_model:: kneemesh.config.PatellaBrep
   :noindex:

.. autopydantic_model:: kneemesh.config.PatellaMesh
   :noindex:
```
