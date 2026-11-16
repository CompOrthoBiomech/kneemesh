import pytest

from kneemesh.gui.files import list_directory


def test_lists_directories_first_and_filters_files(tmp_path):
    for name in ("b.nii", "A.NRRD", "notes.txt", ".hidden.nii"):
        (tmp_path / name).write_text("", encoding="utf-8")
    (tmp_path / "zdir").mkdir()
    (tmp_path / "Adir").mkdir()
    (tmp_path / ".git").mkdir()
    entries = list_directory(tmp_path, [".nii", ".nrrd"])
    assert [(entry["name"], entry["directory"]) for entry in entries] == [
        ("Adir", True),
        ("zdir", True),
        ("A.NRRD", False),
        ("b.nii", False),
    ]
    assert entries[2]["path"] == str(tmp_path / "A.NRRD")
    assert [entry["name"] for entry in list_directory(tmp_path)] == ["Adir", "zdir", "A.NRRD", "b.nii", "notes.txt"]


def test_missing_directory_raises(tmp_path):
    with pytest.raises(OSError):
        list_directory(tmp_path / "missing")
