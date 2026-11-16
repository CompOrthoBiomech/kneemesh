"""Listing the server's directories for the GUI's file dialog, which picks paths on the machine running the pipeline."""

from collections.abc import Collection
from pathlib import Path
from typing import TypedDict


class DirectoryEntry(TypedDict):
    """A file or directory as the file dialog lists it."""

    name: str
    path: str
    directory: bool


def list_directory(directory: str | Path, suffixes: Collection[str] = ()) -> list[DirectoryEntry]:
    """The directories and files in a directory, directories first, each sorted by name, without hidden ones.

    Files are listed only if their name ends with one of suffixes, compared
    without case, or all files if suffixes is empty. Unreadable entries are skipped.

    Raises:
        OSError: If the directory cannot be read.
    """
    lowered = tuple(suffix.lower() for suffix in suffixes)
    directories: list[DirectoryEntry] = []
    files: list[DirectoryEntry] = []
    for path in Path(directory).iterdir():
        if path.name.startswith("."):
            continue
        try:
            is_directory = path.is_dir()
        except OSError:
            continue
        if is_directory:
            directories.append(DirectoryEntry(name=path.name, path=str(path), directory=True))
        elif not lowered or path.name.lower().endswith(lowered):
            files.append(DirectoryEntry(name=path.name, path=str(path), directory=False))
    return sorted(directories, key=lambda entry: entry["name"].lower()) + sorted(files, key=lambda entry: entry["name"].lower())
