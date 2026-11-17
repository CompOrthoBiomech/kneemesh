"""Delete what the kneemesh GUI image never loads from a virtual environment's site-packages.

Run with the environment's Python. It runs trace.py, which exercises the
pipelines and the GUI, and keeps what that loaded:

- VTK: every shared object in vtkmodules the trace did not map, the extension
  modules among them, and the subpackages and type stubs nothing imported.
  vtkmodules.all, and so ``import vtk``, no longer works; import from
  vtkmodules.<module> as kneemesh does.
- SciPy: top-level subpackages nothing imported.
- trame: static bundles for clients other than Vue 3 with Vuetify 3.
- Everywhere: test suites, type stubs, and bytecode caches.

The remaining shared objects are stripped of symbols they do not export.
SimpleITK is one shared object holding every filter, so it can only be
stripped, not pruned.
"""

import json
import shutil
import subprocess
import sys
import sysconfig
from collections.abc import Collection
from pathlib import Path

SITE = Path(sysconfig.get_paths()["purelib"])
# trame's static bundles: the GUI runs Vue 3 (client_type="vue3") with Vuetify 3.
UNUSED_BUNDLES = [
    "trame_client/module/vue2-www",
    "trame_client/module/react-www",
    "trame_vuetify/module/v2-serve",
    "trame_vuetify/module/v3-lab-serve",
    "trame_vuetify/module/v4-serve",
    "trame_vuetify/module/v4-lab-serve",
]


def trace() -> tuple[set[Path], set[str]]:
    script = Path(__file__).with_name("trace.py")
    result = subprocess.run([sys.executable, str(script)], check=True, stdout=subprocess.PIPE, text=True)
    loaded = json.loads(result.stdout)
    return {Path(path).resolve() for path in loaded["objects"]}, set(loaded["modules"])


def remove(path: Path) -> int:
    size = sum(file.stat().st_size for file in path.rglob("*") if file.is_file()) if path.is_dir() else path.stat().st_size
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return size


def unimported_subpackages(package: Path, modules: set[str], keep: Collection[str] = ()) -> list[Path]:
    """Subpackages of package, directories with an __init__.py, that no imported module lies in."""
    return [
        child
        for child in sorted(package.iterdir())
        if (child / "__init__.py").exists() and child.name not in keep and f"{package.name}.{child.name}" not in modules
    ]


def prune_vtk(objects: set[Path], modules: set[str]) -> int:
    package = SITE / "vtkmodules"
    freed = sum(remove(path) for path in sorted(package.glob("*.so*")) if path.resolve() not in objects)
    freed += sum(remove(path) for path in unimported_subpackages(package, modules))
    return freed + sum(remove(path) for path in [package / "all.py", package / "generate_pyi.py", SITE / "vtk.py"] if path.exists())


def prune_scipy(modules: set[str]) -> int:
    # _lib holds the shared helpers every subpackage imports lazily.
    return sum(remove(path) for path in unimported_subpackages(SITE / "scipy", modules, keep={"_lib"}))


def prune_everywhere() -> int:
    freed = sum(remove(SITE / bundle) for bundle in UNUSED_BUNDLES if (SITE / bundle).exists())
    for pattern in ("tests", "test", "__pycache__"):
        freed += sum(remove(path) for path in sorted(SITE.rglob(pattern), reverse=True) if path.is_dir() and path.exists())
    return freed + sum(remove(path) for path in SITE.rglob("*.pyi"))


def strip(root: Path) -> int:
    freed = 0
    for path in sorted(root.rglob("*.so*")):
        if path.is_file() and not path.is_symlink():
            before = path.stat().st_size
            if subprocess.run(["strip", "--strip-unneeded", str(path)], check=False, capture_output=True).returncode == 0:
                freed += before - path.stat().st_size
    return freed


def main() -> None:
    objects, modules = trace()
    environment = Path(sys.prefix)
    for name, freed in [
        ("VTK", prune_vtk(objects, modules)),
        ("SciPy", prune_scipy(modules)),
        ("tests, stubs, caches, and unused trame bundles", prune_everywhere()),
        ("symbols", strip(environment)),
    ]:
        print(f"Removed {freed / 2**20:.0f} MiB of {name}", file=sys.stderr)


if __name__ == "__main__":
    main()
