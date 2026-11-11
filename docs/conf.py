"""Sphinx configuration for the meniscus2brep documentation."""

from importlib.metadata import metadata

_metadata = metadata("meniscus2brep")

project = _metadata["Name"]
author = "Scott Sibole"
copyright = f"2026, {author}"
release = _metadata["Version"]

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.autodoc_pydantic",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build"]

# MyST: $inline$ and $$display$$ math, and ::: fenced directives.
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence"]

# The docstrings use Google style.
napoleon_google_docstring = True
napoleon_numpy_docstring = False

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {"members": True, "show-inheritance": True}

# Pydantic models list each field with its type, default, description, and constraints.
autodoc_pydantic_model_show_json = False
autodoc_pydantic_model_show_config_summary = False
autodoc_pydantic_model_show_validator_summary = False
autodoc_pydantic_model_show_validator_members = False
autodoc_pydantic_model_show_field_summary = True
autodoc_pydantic_model_member_order = "bysource"
autodoc_pydantic_field_list_validators = False
autodoc_pydantic_field_show_constraints = True
autodoc_pydantic_field_show_default = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "pydantic": ("https://docs.pydantic.dev/latest", None),
}

html_theme = "furo"
html_title = "meniscus2brep"
