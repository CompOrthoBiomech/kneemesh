import asyncio
from pathlib import Path

import numpy as np
import pytest
from vtkmodules.util.numpy_support import vtk_to_numpy

from kneemesh.brep import BRep
from kneemesh.disk import read_poly_data, write_brep, write_label_map
from kneemesh.gui.jobs import JobError, JobRunner
from kneemesh.gui.labels import LabelEntry
from kneemesh.gui.view import read_mesh_preview
from kneemesh.gui.worker import brep_surface, run_job


class _Reports(list[object]):
    def put(self, obj: object, /) -> None:
        self.append(obj)


def _outcome(reports: _Reports) -> object:
    return reports[-1]


def test_brep_surface_has_faces_and_curves(meniscus_solid: BRep):
    surface = brep_surface(meniscus_solid)
    assert surface.GetNumberOfPolys() > 100
    assert surface.GetNumberOfLines() > 10
    edge = vtk_to_numpy(surface.GetCellData().GetArray("edge"))
    assert edge.sum() == surface.GetNumberOfLines()
    assert len(edge) == surface.GetNumberOfCells()


def test_unknown_job_reports_an_error(tmp_path):
    reports = _Reports()
    run_job({"kind": "unknown", "preview_directory": str(tmp_path)}, reports)
    match _outcome(reports):
        case ("error", message):
            assert "Unknown job kind" in str(message)
        case other:
            pytest.fail(f"Expected an error, got {other!r}")


def test_geometry_mesh_and_preview_jobs(tmp_path, meniscus_label):
    label_path = str(tmp_path / "meniscus.nii")
    write_label_map(meniscus_label, label_path)
    entry = LabelEntry(id=1, path=label_path, name="meniscus.nii", file_value=1, structure="lateral_meniscus", value=3, include=True)
    out, previews = tmp_path / "out", tmp_path / "previews"

    reports = _Reports()
    run_job(
        {
            "kind": "geometry",
            "config": {"output_directory": str(out)},
            "entries": [entry],
            "structures": ["lateral_meniscus"],
            "interpolator": "label_gaussian",
            "padding": 4,
            "assembled_label_map": str(tmp_path / "segmentation.nrrd"),
            "preview_directory": str(previews),
        },
        reports,
    )
    kind, result = _outcome(reports)  # pyright: ignore[reportGeneralTypeIssues]
    assert kind == "done", result
    assert any(report[0] == "log" for report in reports)  # pyright: ignore[reportIndexIssue]
    brep = out / "lateral_meniscus.brep"
    assert result == {"lateral_meniscus": {"output": str(brep), "preview": str(previews / "lateral_meniscus_geometry.vtp")}}
    assert read_poly_data(previews / "lateral_meniscus_geometry.vtp").GetNumberOfPolys() > 0
    assert (tmp_path / "segmentation_labels.json").read_text(encoding="utf-8").strip().startswith("{")

    reports = _Reports()
    run_job(
        {
            "kind": "mesh",
            "config": {"breps": {"lateral_meniscus": str(brep)}, "output_directory": str(out), "element_type": "tet4"},
            "preview_directory": str(previews),
        },
        reports,
    )
    kind, result = _outcome(reports)  # pyright: ignore[reportGeneralTypeIssues]
    assert kind == "done", result
    assert Path(result["lateral_meniscus"]["output"]).is_file()
    grid = read_mesh_preview(result["lateral_meniscus"]["preview"])
    # The preview holds the tetrahedra and their boundary triangles, with the quality of each tetrahedron.
    types = vtk_to_numpy(grid.GetCellTypes())
    assert set(np.unique(types)) == {5, 10}
    scaled = vtk_to_numpy(grid.GetCellData().GetArray("scaled_jacobian"))
    assert np.all(scaled[types == 10] > 0)
    assert np.isnan(scaled[types == 5]).all()

    reports = _Reports()
    run_job({"kind": "preview", "breps": {"lateral_meniscus": str(brep)}, "preview_directory": str(previews)}, reports)
    kind, result = _outcome(reports)  # pyright: ignore[reportGeneralTypeIssues]
    assert kind == "done", result
    assert Path(result["lateral_meniscus"]["preview"]).is_file()


def test_job_runner_runs_in_a_process(tmp_path, meniscus_solid: BRep):
    brep = tmp_path / "solid.brep"
    write_brep(meniscus_solid, brep)
    logs: list[tuple[str, str]] = []
    runner = JobRunner()
    result = asyncio.run(
        runner.run({"kind": "preview", "breps": {"lateral_meniscus": str(brep)}, "preview_directory": str(tmp_path)}, lambda *log: logs.append(log))
    )
    assert Path(result["lateral_meniscus"]["preview"]).is_file()
    assert ("INFO", f"Previewing {brep}") in logs
    assert not runner.busy
    with pytest.raises(JobError, match="Unknown job kind"):
        asyncio.run(runner.run({"kind": "unknown", "preview_directory": str(tmp_path)}, lambda *_: None))


def test_job_runner_cancels(tmp_path):
    runner = JobRunner()

    async def run_and_cancel() -> None:
        # Starting the process takes seconds of imports, so it is cancelled before it can finish.
        task = asyncio.create_task(runner.run({"kind": "preview", "breps": {}, "preview_directory": str(tmp_path)}, lambda *_: None))
        await asyncio.sleep(0)
        runner.cancel()
        await task

    with pytest.raises(JobError, match="Cancelled"):
        asyncio.run(run_and_cancel())
    assert not runner.busy
