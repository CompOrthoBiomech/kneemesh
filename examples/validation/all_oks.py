"""Run the full assembly > B-rep > mesh pipeline on every OpenKnee(s) subject and report how well it generalizes.

Each subject directory under --data (dat/oks/oks*/ by default) holds the nine
single-label LabelMaps, named for their structures. For each subject this
script:

1. assembles them into one multilabel LabelMap (kneemesh-assemble),
2. builds every structure's B-rep, or a bone's surface (kneemesh-brep), once
   per element shape: hexahedral B-reps (hex_blocks) and tetrahedral ones,
   the bones' surfaces being shared,
3. meshes every B-rep (kneemesh-mesh): hex8 or tet4, bones with tet4 either way.

Every step of every structure runs in its own spawned process, because gmsh
holds global state and can crash or hang the process; a crash, a timeout, or
an exception fails only that step, and the steps that depend on it are
skipped. Steps run in parallel up to --jobs at a time.

The settings follow examples/knee_*.json. Each subject's directory under
--output gets the JSON configurations, so a failure can be repeated with the
console scripts, along with every output and one log per step, and
results.json and results.csv. The report, the docs' Validation page
(docs/validation.md, --report), records the commit it ran at and summarizes per
subject and structure: failures and their errors, warnings, element counts,
scaled Jacobians and Frobenius aspect ratios (before and after optimization
for hexahedra), inverted elements, surface warp, empty node and element sets,
and the mesh's volume against the label's voxel volume.

Run from the repository root:

    uv run python examples/validation/all_oks.py
    uv run python examples/validation/all_oks.py --subjects oks001 oks002 --shapes hex --jobs 6
    uv run python examples/validation/all_oks.py --report-only   # rebuild the report from results.json

The LabelMaps are cropped to their labels and stored as sitkUInt8 first, by
examples/validation/crop_oks.py. Commit before running, so the recorded commit
reproduces the results.
"""

import argparse
import csv
import json
import math
import multiprocessing
import re
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

STRUCTURES = (
    "femur",
    "tibia",
    "patella",
    "lateral_meniscus",
    "medial_meniscus",
    "lateral_tibial_cartilage",
    "medial_tibial_cartilage",
    "patellar_cartilage",
    "femoral_cartilage",
)
BONES = ("femur", "tibia", "patella")
MENISCI = ("lateral_meniscus", "medial_meniscus")
# Sets every mesh of a structure should have, with at least one face.
EXPECTED_SETS = {
    **{bone: ("surface",) for bone in BONES},
    **{meniscus: ("surface", "anterior_horn", "posterior_horn", "body") for meniscus in MENISCI},
    **{
        cartilage: ("surface", "articular_side", "bone_side")
        for cartilage in ("lateral_tibial_cartilage", "medial_tibial_cartilage", "patellar_cartilage", "femoral_cartilage")
    },
}

type Shape = Literal["hex", "tet"]
SHAPES: tuple[Shape, ...] = ("hex", "tet")
ELEMENT_TYPES: dict[Shape, str] = {"hex": "hex8", "tet": "tet4"}

# Heuristic flags for the report, from what the test knee reaches (see CLAUDE.md); they mark rows to look at, not failures.
FLAG_MIN_SCALED_JACOBIAN = {"hex": 0.2, "tet": 0.05}
FLAG_VOLUME_RATIO = (0.85, 1.08)

# Settings as in examples/knee_brep_*.json and examples/knee_mesh_*.json.
BONE_BREP = {"femur": {"shaft_cut_fraction": 0.4}, "tibia": {"shaft_cut_fraction": 0.3}}
BONE_MESH = {
    "element_type": "tet4",
    "cortical_element_size": 2.0,
    "cortical_distance": 4.0,
    "cancellous_grad": 0.5,
    "max_size": 5.0,
}
BREP_SECTIONS: dict[Shape, dict[str, Any]] = {
    "hex": {
        "meniscus": {"hex_blocks": True, "hex_design": "butterfly"},
        "cartilage": {"hex_blocks": True},
        "femoral_cartilage": {"hex_blocks": True},
    },
    "tet": {},
}
MESH_SECTIONS: dict[Shape, dict[str, Any]] = {
    "hex": {
        "meniscus": {"transverse_size": 1.0, "circumferential_size": 1.0, "optimize": True},
        "cartilage": {"element_size": 1.5, "elements_through": 4, "optimize": True},
        "femoral_cartilage": {"element_size": 1.5, "elements_through": 4, "optimize": True},
    },
    "tet": {
        "meniscus": {"max_size": 1.0, "elements_per_circle": 8},
        "cartilage": {"max_size": 1.0},
    },
}

_OPTIMIZING = re.compile(
    r"Optimizing (\d+) hexahedra with minimum scaled Jacobian (\S+), mean (\S+), maximum Frobenius aspect ratio (\S+), "
    + r"and surface warp (\S+) degrees at the 99th percentile, (\S+) at most"
)


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------


def assemble_config(data: Path, out: Path) -> dict[str, Any]:
    return {
        "label_maps": {structure: str(data / f"{structure}.nii") for structure in STRUCTURES},
        "interpolator": "nearest_neighbor",
        "output": str(out / "knee.nrrd"),
    }


def brep_config(out: Path, shape: Shape, labels: dict[str, int]) -> dict[str, Any]:
    return {
        "label_map": str(out / "knee.nrrd"),
        "labels": labels,
        "output": f"_{shape}.brep",
        "output_directory": str(out / "brep"),
        **BREP_SECTIONS[shape],
        **BONE_BREP,
    }


def brep_path(out: Path, structure: str, shape: Shape) -> Path:
    return out / "brep" / (f"{structure}.stl" if structure in BONES else f"{structure}_{shape}.brep")


def mesh_config(out: Path, shape: Shape, structures: tuple[str, ...]) -> dict[str, Any]:
    return {
        "breps": {structure: str(brep_path(out, structure, shape)) for structure in structures},
        "element_type": ELEMENT_TYPES[shape],
        "output": "_{element_type}.vtu",
        "output_directory": str(out / "mesh"),
        **MESH_SECTIONS[shape],
        **{bone: BONE_MESH for bone in BONES},
    }


def write_configs(data: Path, out: Path, shapes: tuple[Shape, ...]) -> None:
    """Write the subject's configurations, which repeat this run with the console scripts from the repository root."""
    out.mkdir(parents=True, exist_ok=True)
    labels = {structure: value for value, structure in enumerate(STRUCTURES, start=1)}
    configs: dict[str, dict[str, Any]] = {"assemble.json": assemble_config(data, out)}
    for shape in shapes:
        configs[f"brep_{shape}.json"] = brep_config(out, shape, labels)
        configs[f"mesh_{ELEMENT_TYPES[shape]}.json"] = mesh_config(out, shape, STRUCTURES) | {
            "assembled_mesh": str(out / "mesh" / f"knee_{ELEMENT_TYPES[shape]}.vtu")
        }
    for name, config in configs.items():
        _ = (out / name).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Steps, each run in its own process
# ---------------------------------------------------------------------------


@dataclass
class Step:
    """One step of one subject: assembly, or a structure's B-rep or mesh for an element shape."""

    subject: str
    stage: Literal["assemble", "brep", "mesh"]
    structure: str  # "all" for assembly
    shape: str  # "hex", "tet", or "-" for steps shared by both shapes
    data: str
    out: str
    depends: list[str] = field(default_factory=list[str])

    @property
    def key(self) -> str:
        return f"{self.subject}/{self.stage}/{self.structure}/{self.shape}"

    @property
    def directory(self) -> Path:
        return Path(self.out) / "steps" / f"{self.stage}_{self.structure}_{self.shape}"


def run_step(step: Step) -> None:
    """Run a step in this process, writing its result to result.json in its directory, and its log beside it."""
    from loguru import logger

    import kneemesh  # noqa: F401  # pyright: ignore[reportUnusedImport] - importing it disables its logger, so do so before enabling

    directory = step.directory
    directory.mkdir(parents=True, exist_ok=True)
    messages: list[dict[str, str]] = []

    def collect(message: Any) -> None:
        record = message.record
        messages.append({"level": record["level"].name, "message": record["message"]})

    logger.remove()
    _ = logger.add(directory / "step.log", level="DEBUG", format="{time:HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}")
    _ = logger.add(collect, level="INFO", format="{message}")
    logger.enable("kneemesh")

    result: dict[str, Any] = {"status": "ok"}
    start = time.perf_counter()
    try:
        result |= _STEP_RUNNERS[step.stage](step)
    except Exception as error:  # noqa: BLE001 - any failure is a result to report
        result |= {"status": "error", "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
        logger.exception("Step failed")
    result["seconds"] = time.perf_counter() - start
    result["warnings"] = [m["message"] for m in messages if m["level"] in ("WARNING", "ERROR", "CRITICAL") and m["message"] != "Step failed"]
    result |= _optimization_start([m["message"] for m in messages])
    _ = (directory / "result.json").write_text(json.dumps(result, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _optimization_start(messages: list[str]) -> dict[str, Any]:
    """The hexahedra's quality before optimization, from the pipeline's log message."""
    for message in messages:
        if match := _OPTIMIZING.search(message):
            _, minimum, mean, aspect, warp_99, max_warp = match.groups()
            return {
                "raw_min_sj": float(minimum),
                "raw_mean_sj": float(mean),
                "raw_max_aspect": float(aspect),
                "raw_warp_99": float(warp_99),
                "raw_max_warp": float(max_warp),
            }
    return {}


def _assemble(step: Step) -> dict[str, Any]:
    import SimpleITK as sitk

    from kneemesh.config import LabelMapAssembly
    from kneemesh.disk import read_label_map
    from kneemesh.pipeline import run_assembly

    data, out = Path(step.data), Path(step.out)
    sources: dict[str, Any] = {}
    for structure in STRUCTURES:
        image = read_label_map(data / f"{structure}.nii")
        voxels = int(np.count_nonzero(sitk.GetArrayViewFromImage(image)))
        sources[structure] = {
            "size": list(image.GetSize()),
            "spacing": [round(s, 4) for s in image.GetSpacing()],
            "pixel_type": image.GetPixelIDTypeAsString(),
            "voxel_volume": voxels * math.prod(image.GetSpacing()),
        }
    segmentation = run_assembly(LabelMapAssembly.model_validate(assemble_config(data, out)))
    array = sitk.GetArrayViewFromImage(segmentation.image)
    voxel = math.prod(segmentation.image.GetSpacing())
    volumes = {name: float(np.count_nonzero(array == value)) * voxel for name, value in segmentation.labels.items()}
    return {
        "labels": dict(segmentation.labels),
        "grid": list(segmentation.image.GetSize()),
        "spacing": [round(s, 4) for s in segmentation.image.GetSpacing()],
        "sources": sources,
        "assembled_volume": volumes,
    }


def _brep(step: Step) -> dict[str, Any]:
    from kneemesh.config import LabelMapBrep
    from kneemesh.disk import read_segmentation
    from kneemesh.pipeline import build_structure

    out = Path(step.out)
    segmentation = read_segmentation(out / "knee.nrrd")
    shape = cast(Shape, "hex" if step.shape == "-" else step.shape)
    config = LabelMapBrep.model_validate(brep_config(out, shape, dict(segmentation.labels)))
    geometry = build_structure(segmentation, cast(Any, step.structure), config)
    path = config.output_path(cast(Any, step.structure))
    result: dict[str, Any] = {"path": str(path), "bytes": path.stat().st_size}
    if step.structure in BONES:
        result["triangles"] = int(cast(Any, geometry).GetNumberOfCells())
    return result


def _mesh(step: Step) -> dict[str, Any]:
    from kneemesh.config import BrepMesh
    from kneemesh.mesh_optimization import FROBENIUS_ASPECT, SCALED_JACOBIAN, WARP
    from kneemesh.pipeline import run_mesh

    out = Path(step.out)
    shape: Shape = "tet" if step.shape == "-" else cast(Shape, step.shape)
    config = BrepMesh.model_validate(mesh_config(out, shape, (step.structure,)))
    mesh = run_mesh(config)[cast(Any, step.structure)]
    element_count = sum(len(rows) for rows in mesh.elements.values())
    scaled = mesh.quality[SCALED_JACOBIAN][:element_count]
    aspects = mesh.quality[FROBENIUS_ASPECT][:element_count]
    result: dict[str, Any] = {
        "path": str(config.output_path(cast(Any, step.structure))),
        "element_type": next(iter(mesh.elements)),
        "nodes": len(mesh.nodes),
        "elements": element_count,
        "min_sj": float(np.min(scaled)),
        "p01_sj": float(np.percentile(scaled, 1)),
        "mean_sj": float(np.mean(scaled)),
        "inverted": int(np.count_nonzero(scaled <= 0)),
        "max_aspect": float(np.max(aspects)),
        "p99_aspect": float(np.percentile(aspects[np.isfinite(aspects)], 99)) if np.isfinite(aspects).any() else math.inf,
        "mean_aspect": float(np.mean(aspects[np.isfinite(aspects)])) if np.isfinite(aspects).any() else math.inf,
        "volume": mesh_volume(mesh.nodes, mesh.elements),
        "sets": {name: {"faces": len(mesh.element_sets[name]), "nodes": len(mesh.node_sets[name])} for name in mesh.element_sets},
    }
    if WARP in mesh.quality:
        warps = mesh.quality[WARP][~np.isnan(mesh.quality[WARP])]
        result |= {"warp_99": float(np.percentile(warps, 99)), "max_warp": float(np.max(warps))}
    return result


# Six positively oriented tetrahedra about the diagonal 0-6 of a gmsh hexahedron.
_HEX_TETS = np.array([(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)])


def mesh_volume(nodes: np.ndarray, elements: dict[str, np.ndarray]) -> float:
    """The volume of a mesh's tetrahedra and hexahedra, measured on their corners; 0 for a surface mesh."""
    total = 0.0
    for name, rows in elements.items():
        if name.startswith("Tetrahedron"):
            tets = rows[:, None, :4]
        elif name.startswith("Hexahedron"):
            tets = rows[:, _HEX_TETS]
        else:
            continue
        points = nodes[tets]
        total += float(np.linalg.det(points[..., 1:, :] - points[..., :1, :]).sum()) / 6
    return total


_STEP_RUNNERS: dict[str, Callable[[Step], dict[str, Any]]] = {"assemble": _assemble, "brep": _brep, "mesh": _mesh}


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------


def plan(subjects: list[Path], output: Path, shapes: tuple[Shape, ...]) -> list[Step]:
    steps: list[Step] = []
    for data in subjects:
        subject, out = data.name, output / data.name
        assemble = Step(subject, "assemble", "all", "-", str(data), str(out))
        steps.append(assemble)
        for structure in STRUCTURES:
            if structure in BONES:
                brep = Step(subject, "brep", structure, "-", str(data), str(out), [assemble.key])
                mesh = Step(subject, "mesh", structure, "-", str(data), str(out), [brep.key])
                steps += [brep, mesh]
                continue
            for shape in shapes:
                brep = Step(subject, "brep", structure, shape, str(data), str(out), [assemble.key])
                mesh = Step(subject, "mesh", structure, shape, str(data), str(out), [brep.key])
                steps += [brep, mesh]
    return steps


def execute(steps: list[Step], jobs: int, timeout: float) -> dict[str, dict[str, Any]]:
    """Run steps in spawned processes, at most jobs at once, each after the steps it depends on succeeded."""
    context = multiprocessing.get_context("spawn")
    results: dict[str, dict[str, Any]] = {}
    pending = list(steps)
    running: dict[str, tuple[Step, BaseProcess, float]] = {}
    total, started = len(steps), time.perf_counter()

    def finish(step: Step, result: dict[str, Any]) -> None:
        results[step.key] = result
        detail = result.get("error") or ""
        print(
            f"[{len(results):>3}/{total}] {time.perf_counter() - started:7.0f} s  {result['status']:<8} {step.key}"
            + (f"  ({result.get('seconds', 0):.0f} s)" if "seconds" in result else "")
            + (f"  {detail.splitlines()[0][:120]}" if detail else ""),
            flush=True,
        )

    while pending or running:
        for step in list(pending):
            failed = [key for key in step.depends if key in results and results[key]["status"] != "ok"]
            if failed:
                pending.remove(step)
                finish(step, {"status": "skipped", "error": f"skipped because {', '.join(failed)} did not succeed"})
            elif len(running) < jobs and all(key in results for key in step.depends):
                pending.remove(step)
                step.directory.mkdir(parents=True, exist_ok=True)
                (step.directory / "result.json").unlink(missing_ok=True)
                process = context.Process(target=run_step, args=(step,), name=step.key)
                process.start()
                running[step.key] = (step, process, time.perf_counter())
        for key, (step, process, start) in list(running.items()):
            elapsed = time.perf_counter() - start
            if process.is_alive() and elapsed < timeout:
                continue
            if process.is_alive():
                process.kill()
                process.join()
                result = {"status": "timeout", "error": f"killed after {timeout:.0f} s", "seconds": elapsed}
            else:
                process.join()
                path = step.directory / "result.json"
                if path.exists():
                    result = json.loads(path.read_text(encoding="utf-8"))
                else:
                    code = process.exitcode
                    reason = f"signal {-code}" if code is not None and code < 0 else f"exit code {code}"
                    result = {"status": "crashed", "error": f"process died with {reason} (see {step.directory / 'step.log'})", "seconds": elapsed}
            del running[key]
            finish(step, result)
        time.sleep(0.2)
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class Row:
    """One structure of one subject for one element shape: its B-rep and mesh steps."""

    subject: str
    structure: str
    shape: str
    element_type: str
    status: str
    failed_stage: str = ""
    error: str = ""
    brep_seconds: float = math.nan
    mesh_seconds: float = math.nan
    nodes: int = 0
    elements: int = 0
    raw_min_sj: float = math.nan
    min_sj: float = math.nan
    p01_sj: float = math.nan
    mean_sj: float = math.nan
    inverted: int = 0
    max_aspect: float = math.nan
    p99_aspect: float = math.nan
    warp_99: float = math.nan
    max_warp: float = math.nan
    voxel_volume: float = math.nan
    volume: float = math.nan
    volume_ratio: float = math.nan
    empty_sets: str = ""
    warnings: int = 0
    flags: str = ""


def rows_of(steps: list[Step], results: dict[str, dict[str, Any]]) -> list[Row]:
    rows: list[Row] = []
    meshes = [step for step in steps if step.stage == "mesh"]
    for mesh_step in meshes:
        brep_key = mesh_step.depends[0]
        assemble_key = f"{mesh_step.subject}/assemble/all/-"
        brep, mesh, assemble = results.get(brep_key, {}), results.get(mesh_step.key, {}), results.get(assemble_key, {})
        shape = "tet" if mesh_step.shape == "-" else mesh_step.shape
        row = Row(mesh_step.subject, mesh_step.structure, mesh_step.shape, mesh.get("element_type") or ELEMENT_TYPES[cast(Shape, shape)], "ok")
        for stage, result in (("assemble", assemble), ("brep", brep), ("mesh", mesh)):
            if result.get("status", "missing") != "ok":
                row.status, row.failed_stage = result.get("status", "missing"), stage
                row.error = (result.get("error") or "").splitlines()[0] if result.get("error") else ""
                break
        row.brep_seconds = brep.get("seconds", math.nan)
        row.mesh_seconds = mesh.get("seconds", math.nan)
        row.warnings = len(brep.get("warnings", [])) + len(mesh.get("warnings", []))
        row.voxel_volume = assemble.get("assembled_volume", {}).get(mesh_step.structure, math.nan)
        if mesh.get("status") == "ok":
            for name in ("nodes", "elements", "raw_min_sj", "min_sj", "p01_sj", "mean_sj", "inverted", "max_aspect", "p99_aspect"):
                if name in mesh:
                    setattr(row, name, mesh[name])
            row.warp_99, row.max_warp = mesh.get("warp_99", math.nan), mesh.get("max_warp", math.nan)
            row.volume = mesh["volume"]
            row.volume_ratio = row.volume / row.voxel_volume if row.voxel_volume else math.nan
            sets = mesh.get("sets", {})
            row.empty_sets = " ".join(name for name in EXPECTED_SETS[mesh_step.structure] if sets.get(name, {}).get("faces", 0) == 0)
            row.flags = " ".join(flags_of(row, shape))
        rows.append(row)
    return rows


def flags_of(row: Row, shape: str) -> list[str]:
    flags: list[str] = []
    if row.inverted:
        flags.append("inverted")
    if row.structure not in BONES and row.min_sj < FLAG_MIN_SCALED_JACOBIAN[cast(Shape, shape)]:
        flags.append("low-SJ")
    low, high = FLAG_VOLUME_RATIO
    if row.structure not in BONES and not low <= row.volume_ratio <= high:
        flags.append("volume")
    if row.empty_sets:
        flags.append("empty-set")
    return flags


def _fmt(value: float, digits: int = 3) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "-"
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return f"{value:.{digits}g}" if isinstance(value, float) else str(value)


def _table(header: list[str], body: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|", *("| " + " | ".join(row) + " |" for row in body)]


REPOSITORY_URL = "https://github.com/CompOrthoBiomech/kneemesh"
# Paths whose uncommitted changes would make the recorded commit misleading.
_TRACKED_PATHS = ("src", "examples/validation")


def _git_commit() -> str:
    """HEAD's full hash, marked when the package or this script has uncommitted changes."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", *_TRACKED_PATHS], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit + (" (modified)" if dirty else "")


def _commit_link(commit: str) -> str:
    """The commit as a Markdown link to it on GitHub, with its modified note after the link."""
    hash_, _, note = commit.partition(" ")
    if not re.fullmatch(r"[0-9a-f]{7,40}", hash_):
        return commit
    return f"[`{hash_}`]({REPOSITORY_URL}/commit/{hash_})" + (f" {note}" if note else "")


def write_report(report: Path, output: Path, steps: list[Step], results: dict[str, dict[str, Any]], meta: dict[str, Any]) -> None:
    """Write the report to report, a Markdown page of the docs, and results.csv to output."""
    rows = rows_of(steps, results)
    subjects = sorted({row.subject for row in rows})
    modified = " (modified)" in meta["commit"]
    lines = [
        "# Validation",
        "",
        "kneemesh is developed against one knee, the structures in `tests/dat`. To see how it generalizes, "
        + "`examples/validation/all_oks.py` runs the full assembly, B-rep, and mesh pipeline on every structure of the "
        + "[OpenKnee(s)](https://simtk.org/projects/openknee) subjects' segmentations, which `examples/validation/crop_oks.py` "
        + "crops to their labels and stores as `sitkUInt8`, as `tests/dat` is. The script runs locally, since the data is not "
        + "distributed with kneemesh, and writes this page.",
        "",
        f"Generated {meta['generated']} with kneemesh at commit {_commit_link(meta['commit'])}, shapes {', '.join(meta['shapes'])}, "
        + f"{meta['jobs']} jobs, timeout {meta['timeout']:.0f} s per step, {meta['wall_seconds']:.0f} s in all."
        + (
            " The package or the validation script had uncommitted changes, so the commit does not reproduce these results exactly."
            if modified
            else ""
        ),
        "",
        "Settings follow `examples/knee_*.json`; each subject's output directory holds the configurations to repeat it with the console scripts. "
        + "`SJ` is the scaled Jacobian (1 ideal, <= 0 inverted), measured on corner nodes; `raw` is before hexahedral optimization. "
        + "`aspect` is the Frobenius aspect ratio. `warp` is the boundary quadrilaterals' warp in degrees. `vol` is the mesh's volume "
        + "over the assembled label's voxel volume; bones are cut at their shafts, so theirs is not comparable. "
        + f"Flags are heuristic: inverted elements, minimum SJ below {FLAG_MIN_SCALED_JACOBIAN['hex']} (hex) or "
        + f"{FLAG_MIN_SCALED_JACOBIAN['tet']} (tet), vol outside {FLAG_VOLUME_RATIO[0]}-{FLAG_VOLUME_RATIO[1]}, or an expected set with no faces.",
        "",
    ]

    # Overview matrix.
    lines += ["## Overview", ""]
    columns = [(structure, shape) for structure in STRUCTURES for shape in (("-",) if structure in BONES else meta["shapes"])]
    by_key = {(row.subject, row.structure, row.shape): row for row in rows}
    header = ["subject", *(f"{structure}" + ("" if shape == "-" else f" {ELEMENT_TYPES[cast(Shape, shape)]}") for structure, shape in columns)]
    body: list[list[str]] = []
    for subject in subjects:
        cells = [subject]
        for structure, shape in columns:
            row = by_key.get((subject, structure, shape))
            if row is None:
                cells.append("")
            elif row.status != "ok":
                cells.append(f"**{row.status.upper()}** ({row.failed_stage})")
            else:
                cells.append(f"{_fmt(row.min_sj, 2)}" + (f" ⚠ {row.flags}" if row.flags else ""))
        body.append(cells)
    lines += ["Each cell is the minimum scaled Jacobian, or the failed stage.", "", *_table(header, body), ""]

    ok = [row for row in rows if row.status == "ok"]
    lines += [f"{len(ok)} of {len(rows)} structure meshes succeeded; {sum(1 for row in ok if row.flags)} of them flagged.", ""]

    # Per structure across subjects.
    lines += ["## By structure", ""]
    body = []
    for structure, shape in columns:
        group = [row for row in rows if row.structure == structure and row.shape == shape]
        good = [row for row in group if row.status == "ok"]

        def spread(name: str, rows: list[Row] = good) -> str:
            values = [getattr(row, name) for row in rows if not math.isnan(getattr(row, name))]
            return f"{_fmt(min(values))} / {_fmt(float(np.median(values)))} / {_fmt(max(values))}" if values else "-"

        body.append(
            [
                structure,
                ELEMENT_TYPES[cast(Shape, "tet" if shape == "-" else shape)],
                f"{len(good)}/{len(group)}",
                spread("raw_min_sj"),
                spread("min_sj"),
                spread("mean_sj"),
                spread("max_aspect"),
                spread("warp_99"),
                spread("volume_ratio"),
            ]
        )
    lines += [
        "Values are min / median / max over the subjects that succeeded.",
        "",
        *_table(["structure", "type", "ok", "raw min SJ", "min SJ", "mean SJ", "max aspect", "warp 99%", "vol"], body),
        "",
    ]

    # Failures.
    failures = [(key, result) for key, result in results.items() if result["status"] not in ("ok", "skipped")]
    lines += ["## Failures", ""]
    if not failures:
        lines += ["None.", ""]
    for key, result in sorted(failures):
        step = next(step for step in steps if step.key == key)
        lines += [f"### {key}: {result['status']}", "", f"Log: `{step.directory / 'step.log'}`", "", "```", result.get("error", ""), "```", ""]
        if tb := result.get("traceback"):
            lines += ["<details><summary>traceback</summary>", "", "```", tb.rstrip(), "```", "", "</details>", ""]
    skipped = sorted(key for key, result in results.items() if result["status"] == "skipped")
    if skipped:
        lines += [f"Skipped after a failure: {', '.join(f'`{key}`' for key in skipped)}.", ""]

    # Per subject.
    lines += ["## Subjects", ""]
    for subject in subjects:
        assemble = results.get(f"{subject}/assemble/all/-", {})
        lines += [f"### {subject}", ""]
        if assemble.get("status") == "ok":
            spacings = sorted({tuple(source["spacing"]) for source in assemble["sources"].values()})
            lines += [
                f"Assembled onto a {' x '.join(map(str, assemble['grid']))} grid at spacing {assemble['spacing']} "
                + f"in {assemble['seconds']:.0f} s, from inputs at spacings {', '.join(map(str, spacings))}.",
                "",
            ]
            lost = [
                f"{name} {assemble['assembled_volume'][name] / source['voxel_volume']:.3f}"
                for name, source in assemble["sources"].items()
                if abs(assemble["assembled_volume"][name] / source["voxel_volume"] - 1) > 0.01
            ]
            if lost:
                lines += [f"Assembled over input voxel volume, where it changed by more than 1%: {', '.join(lost)}.", ""]
        body = []
        for row in (row for row in rows if row.subject == subject):
            body.append(
                [
                    row.structure,
                    row.element_type.replace("Tetrahedron ", "tet").replace("Hexahedron ", "hex"),
                    row.status if row.status == "ok" else f"**{row.status}** ({row.failed_stage})",
                    f"{_fmt(row.brep_seconds, 3)} / {_fmt(row.mesh_seconds, 3)}",
                    str(row.nodes or "-"),
                    str(row.elements or "-"),
                    _fmt(row.raw_min_sj),
                    _fmt(row.min_sj),
                    _fmt(row.p01_sj),
                    _fmt(row.mean_sj),
                    str(row.inverted),
                    _fmt(row.max_aspect),
                    _fmt(row.warp_99),
                    _fmt(row.volume_ratio),
                    str(row.warnings),
                    row.flags or row.error[:80],
                ]
            )
        lines += _table(
            [
                "structure",
                "type",
                "status",
                "s brep/mesh",
                "nodes",
                "elements",
                "raw min SJ",
                "min SJ",
                "1% SJ",
                "mean SJ",
                "inverted",
                "max aspect",
                "warp 99%",
                "vol",
                "warnings",
                "flags / error",
            ],
            body,
        )
        warned = [
            (key, message)
            for key, result in results.items()
            if key.startswith(f"{subject}/")
            for message in dict.fromkeys(result.get("warnings", []))
            if result["status"] == "ok"
        ]
        if warned:
            lines += ["", "<details><summary>warnings</summary>", "", *(f"- `{key}`: {message}" for key, message in warned), "", "</details>"]
        lines.append("")

    report.parent.mkdir(parents=True, exist_ok=True)
    _ = report.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(rows[0]))) if rows else None
        if writer is not None:
            writer.writeheader()
            writer.writerows(asdict(row) for row in rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("dat/oks"), help="directory of oks*/ subject directories")
    parser.add_argument("--output", type=Path, default=Path("dat/oks_validation"), help="directory for outputs and the report")
    parser.add_argument("--subjects", nargs="*", help="subject names to run, by default every oks*/ under --data")
    parser.add_argument("--shapes", nargs="+", choices=SHAPES, default=list(SHAPES), help="element shapes of the soft tissue")
    parser.add_argument("--jobs", type=int, default=4, help="steps run at once; each may take a few GB on full-size images")
    parser.add_argument("--timeout", type=float, default=900.0, help="seconds before a step is killed; gmsh can stall on a folded surface")
    parser.add_argument("--report", type=Path, default=Path("docs/validation.md"), help="the report, the docs' Validation page")
    parser.add_argument("--report-only", action="store_true", help="rebuild the report from --output's results.json without running")
    arguments = parser.parse_args()

    output: Path = arguments.output
    shapes = cast(tuple[Shape, ...], tuple(shape for shape in SHAPES if shape in arguments.shapes))
    results_path = output / "results.json"
    if arguments.report_only:
        saved = json.loads(results_path.read_text(encoding="utf-8"))
        steps = [Step(**step) for step in saved["steps"]]
        write_report(arguments.report, output, steps, saved["results"], saved["meta"])
        print(f"Wrote {arguments.report}")
        return 0

    subjects = sorted(path for path in arguments.data.glob("oks*") if path.is_dir())
    if arguments.subjects:
        subjects = [path for path in subjects if path.name in arguments.subjects]
    if not subjects:
        parser.error(f"no subjects under {arguments.data}")
    missing = [f"{path.name}/{structure}.nii" for path in subjects for structure in STRUCTURES if not (path / f"{structure}.nii").exists()]
    if missing:
        parser.error(f"missing LabelMaps: {', '.join(missing)}")

    for data in subjects:
        write_configs(data, output / data.name, shapes)
    steps = plan(subjects, output, shapes)
    print(f"Running {len(steps)} steps for {', '.join(path.name for path in subjects)} with {arguments.jobs} jobs", flush=True)
    start = time.perf_counter()
    results = execute(steps, arguments.jobs, arguments.timeout)
    meta = {
        "generated": f"{datetime.now():%Y-%m-%d %H:%M}",
        "commit": _git_commit(),
        "shapes": list(shapes),
        "jobs": arguments.jobs,
        "timeout": arguments.timeout,
        "wall_seconds": time.perf_counter() - start,
    }
    _ = results_path.write_text(
        json.dumps({"meta": meta, "steps": [asdict(step) for step in steps], "results": results}, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    write_report(arguments.report, output, steps, results, meta)
    print(f"Wrote {arguments.report}")
    return 0 if all(result["status"] == "ok" for result in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
