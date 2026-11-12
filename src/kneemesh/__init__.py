"""kneemesh: B-rep geometry and meshes of knee structures from segmented images."""

from importlib.metadata import version

from loguru import logger

__version__ = version("kneemesh")

# A library stays silent unless its user opts in with logger.enable("kneemesh").
logger.disable("kneemesh")
