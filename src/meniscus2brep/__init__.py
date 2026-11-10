"""meniscus2brep."""

from importlib.metadata import version

from loguru import logger

__version__ = version("meniscus2brep")

# A library stays silent unless its user opts in with logger.enable("meniscus2brep").
logger.disable("meniscus2brep")
