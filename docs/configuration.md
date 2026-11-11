# Configuration

Each pipeline reads one of these models from JSON with {py:meth}`~meniscus2brep.config.BaseConfig.from_json`. Unknown keys are rejected.

```{eval-rst}
.. autopydantic_model:: meniscus2brep.config.BrepConfig
   :noindex:

.. autopydantic_model:: meniscus2brep.config.MeshConfig
   :noindex:
```
