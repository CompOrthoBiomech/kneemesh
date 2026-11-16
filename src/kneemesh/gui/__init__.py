"""A trame GUI that runs the pipeline interactively: import labels, configure and build each structure's geometry and mesh, and view them.

The app imports trame, an optional dependency (``pip install kneemesh[gui]``),
so it is loaded only by main; the other modules, which the GUI's worker
process imports, do not need it.
"""

import importlib
import sys


def main() -> None:
    """Run the kneemesh GUI (the kneemesh-gui console script)."""
    try:
        # Imported by name so that the worker, which imports this package, never loads trame.
        app = importlib.import_module("kneemesh.gui.app")
    except ImportError as error:
        sys.exit(f"The kneemesh GUI needs the gui extra, such as `uv sync --extra gui` or `pip install kneemesh[gui]`: {error}")
    app.main()
