"""The kneemesh GUI: a trame app that imports labels, configures each structure's geometry and mesh, runs them, and shows the results."""

import asyncio
import atexit
import io
import json
import os
import shutil
import sys
import tempfile
import zipfile
from collections.abc import Callable, Generator, Iterable, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast, get_args

import SimpleITK as sitk
from loguru import logger
from pydantic import ValidationError
from trame.app import asynchronous, get_server
from trame.app.file_upload import ClientFile
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import client, html
from trame.widgets import vtk as vtk_widgets
from trame.widgets import vuetify3 as v3
from trame_server import Server

from kneemesh.config import (
    BONE_STRUCTURES,
    BONES,
    OWN_SECTION_STRUCTURES,
    STRUCTURE_SECTIONS,
    STRUCTURES,
    BrepMesh,
    CartilageMesh,
    ElementType,
    LabelMapAssembly,
    LabelMapBrep,
    Structure,
)
from kneemesh.disk import SUPPORTED_SUFFIXES, read_label_map, read_poly_data, read_surface, segmentation_labels_path, write_segmentation
from kneemesh.gui import settings
from kneemesh.gui.files import list_directory
from kneemesh.gui.jobs import JobError, JobRunner
from kneemesh.gui.labels import (
    LabelEntry,
    brep_sources,
    compose_segmentation,
    config_entries,
    entry_problems,
    label_surface,
    label_values,
    missing_bones,
    new_entries,
)
from kneemesh.gui.view import METRICS, STRUCTURE_ITEMS, Scene, read_mesh_preview, structure_title

_LOG_LINES = 400
_DRAWER_WIDTH = 440

# Dragging the bar between the drawer's panels resizes the top one. Vue templates cannot reach window, so the handler is a global
# function the bar calls through its event's view.
_SPLITTER_SCRIPT = """
window.kneemeshSplit = function (event) {
  const top = event.currentTarget.previousElementSibling;
  const container = event.currentTarget.parentElement;
  const startY = event.clientY;
  const startHeight = top.getBoundingClientRect().height;
  const move = (e) => {
    const limit = container.getBoundingClientRect().height - 80;
    top.style.height = Math.min(limit, Math.max(60, startHeight + e.clientY - startY)) + "px";
  };
  const up = () => {
    window.removeEventListener("mousemove", move);
    window.removeEventListener("mouseup", up);
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
  };
  window.addEventListener("mousemove", move);
  window.addEventListener("mouseup", up);
  document.body.style.cursor = "row-resize";
  document.body.style.userSelect = "none";
  event.preventDefault();
};
"""

_STYLE = """
.kneemesh-splitter { flex: none; height: 7px; cursor: row-resize; background: rgba(var(--v-border-color), 0.12);
  border-top: 1px solid rgba(var(--v-border-color), 0.2); border-bottom: 1px solid rgba(var(--v-border-color), 0.2); }
.kneemesh-splitter:hover { background: rgba(var(--v-theme-primary), 0.5); }
.kneemesh-log { font-family: monospace; font-size: 12px; white-space: pre-wrap; }
"""

_LOG_COLORS = {"WARNING": "orange", "ERROR": "red", "CRITICAL": "red", "SUCCESS": "green"}


class KneemeshApp:
    """The GUI's state, layout, and handlers, bound to a trame server."""

    def __init__(self, server: Server | None = None) -> None:
        server = server if server is not None else get_server(client_type="vue3")
        assert server is not None
        self.server: Server = server
        # trame's state and controller are dynamic: any attribute is a state key or a controller function.
        self.state: Any = server.state
        self.ctrl: Any = server.controller
        self.scene: Scene = Scene()
        self.jobs: JobRunner = JobRunner()
        self.preview_directory: Path = Path(tempfile.mkdtemp(prefix="kneemesh-gui-"))
        atexit.register(shutil.rmtree, self.preview_directory, True)
        self._dialog_callback: Callable[[str], object] | None = None
        self._surfaces: dict[tuple[str, int], Any] = {}
        # The files this session wrote, in order, for downloading.
        self._written: dict[str, None] = {}
        # The structures whose own sections hold values of their own, which making them their own again keeps.
        self._edited_sections: dict[settings.Kind, set[str]] = {"geometry": set(), "mesh": set()}
        self._initialize_state()
        self._bind()
        self._build_ui()
        self._validate()

    # State

    def _initialize_state(self) -> None:
        state = self.state
        state.update(settings.state_values("geometry"))
        state.update(settings.state_values("mesh"))
        state.update(
            {
                "labels": [],
                "label_problems": [],
                "label_structures": [],
                "selected": "settings",
                "selected_kind": "settings",
                "selected_structure": None,
                "selected_section": None,
                "selected_shared": [],
                "element_types": {},
                "brep_paths": {},
                "expected_geometry": {},
                "status": {structure: {"geometry": False, "mesh": False} for structure in STRUCTURES},
                "own_sections": {"geometry": [], "mesh": []},
                "hidden": [],
                "solo": None,
                "clip": False,
                "clip_fixed": False,
                "clip_offset": 0.0,
                "dialog_upload": None,
                "dialog_selection_name": "",
                "written_count": 0,
                "display": "latest",
                "color_by": "",
                "color_by_items": [
                    {"title": "Structure colors", "value": ""},
                    *({"title": metric.title, "value": name} for name, metric in METRICS.items()),
                ],
                "form_errors": {},
                "busy": False,
                "busy_text": "",
                "log_lines": [],
                "show_log": True,
                "view_ratio": 1.0,
                "tree_open": ["labels", "structures"],
                "snackbar": False,
                "snackbar_text": "",
                "snackbar_color": "error",
                "dialog_open": False,
                "dialog_title": "",
                "dialog_mode": "open",
                "dialog_directory": str(Path.cwd()),
                "dialog_entries": [],
                "dialog_selection": "",
                "dialog_filename": "",
                "dialog_error": "",
                "dialog_suffixes": [],
                "structure_items": STRUCTURE_ITEMS,
                "structure_titles": {structure: structure_title(structure) for structure in STRUCTURES},
                "element_type_items": list(get_args(ElementType)),
                "display_items": [
                    {"title": "Latest stage", "value": "latest"},
                    {"title": "Labels", "value": "label"},
                    {"title": "Geometry", "value": "geometry"},
                    {"title": "Meshes", "value": "mesh"},
                ],
            }
        )

    def _bind(self) -> None:
        keys = [spec.key for kind in ("geometry", "mesh") for spec in settings.all_fields(kind)]
        self.state.change(*keys, settings.OUTPUT_DIRECTORY_KEY, "element_types", "brep_paths", "own_sections")(lambda **_: self._validate())
        self.state.change("display", "hidden", "solo", "color_by")(lambda **_: self._refresh_scene())
        self.state.change("clip", "clip_fixed", "clip_offset")(lambda **_: self._set_clipping())
        self.state.change("dialog_upload")(lambda dialog_upload, **_: self._upload(dialog_upload))
        self.server.trigger("kneemesh_download")(self.download_file)
        self.server.trigger("kneemesh_download_outputs")(self.download_outputs)
        self.server.cli.add_argument("--config", help="JSON configuration of kneemesh-brep, kneemesh-mesh, or kneemesh-assemble to import")
        self.server.cli.add_argument(
            "--view-ratio",
            type=float,
            default=1.0,
            help="Render the 3D view at this many pixels per screen pixel, such as 2 for sharp images on a high-density display; "
            + "server rendering costs grow with its square",
        )
        self.ctrl.on_server_ready.add(self._on_ready)
        for name in (
            "add_label",
            "update_label",
            "remove_label",
            "select",
            "set_element_type",
            "set_brep_path",
            "toggle_visibility",
            "toggle_solo",
            "set_own_section",
            "reset_section",
            "build",
            "resolve_overclosures",
            "mesh",
            "cancel",
            "load_existing",
            "import_config",
            "export_geometry",
            "export_mesh",
            "browse_output_directory",
            "dialog_open_directory",
            "dialog_up",
            "dialog_click",
            "dialog_confirm",
            "reset_camera",
        ):
            setattr(self.ctrl, name, getattr(self, name))

    def _on_ready(self, **_: object) -> None:
        args = self.server.cli.parse_known_args()[0]
        self.set_view_ratio(args.view_ratio)
        if args.config:
            asynchronous.create_task(self._import_config(args.config))

    def set_view_ratio(self, ratio: float) -> None:
        """Render the 3D view at ratio pixels per screen pixel, scaling its fonts and lines to match.

        Raises:
            ValueError: If ratio is not positive.
        """
        if not ratio > 0:
            raise ValueError(f"The view ratio must be positive, not {ratio}")
        self.scene.set_pixel_ratio(ratio)
        self.state.view_ratio = ratio

    # Layout

    def _build_ui(self) -> None:
        with SinglePageWithDrawerLayout(self.server, width=_DRAWER_WIDTH) as layout:
            client.Script(_SPLITTER_SCRIPT)
            client.Style(_STYLE)
            layout.title.set_text("kneemesh")
            self._toolbar(layout.toolbar)
            with layout.drawer, html.Div(style="display: flex; flex-direction: column; height: 100%;"):
                with html.Div(style="height: 45%; flex: none; overflow: auto;"):
                    self._tree()
                html.Div(classes="kneemesh-splitter", mousedown="$event.view.kneemeshSplit($event)", __events=["mousedown"])
                with html.Div(style="flex: 1 1 0; min-height: 0; overflow: auto;", classes="pa-3"):
                    self._settings_panel()
            with layout.content, html.Div(style="display: flex; flex-direction: column; height: 100%;"):
                with html.Div(style="flex: 1 1 0; min-height: 0; position: relative;"):
                    view = vtk_widgets.VtkRemoteView(self.scene.render_window, interactive_ratio=("view_ratio",), still_ratio=("view_ratio",))
                    self.ctrl.view_update = view.update
                    self.ctrl.view_reset_camera = view.reset_camera
                self._log_panel()
            layout.footer.hide()
            self._file_dialog()
            v3.VSnackbar(v_model=("snackbar",), text=("snackbar_text",), color=("snackbar_color",), timeout=8000, multi_line=True, location="bottom")

    def _toolbar(self, toolbar: Any) -> None:
        with toolbar:
            v3.VSpacer()
            _icon_button("mdi-file-import-outline", "Import a configuration (JSON)", click=self.ctrl.import_config, disabled=("busy",))
            _icon_button("mdi-file-export-outline", "Export the geometry configuration (JSON)", click=self.ctrl.export_geometry)
            _icon_button("mdi-file-export", "Export the mesh configuration (JSON)", click=self.ctrl.export_mesh)
            _icon_button(
                "mdi-folder-download-outline",
                "Download the files this session wrote, as a zip",
                click="utils.download('kneemesh_outputs.zip', trigger('kneemesh_download_outputs'), 'application/zip')",
                disabled=("!written_count",),
            )
            v3.VDivider(vertical=True, classes="mx-2")
            _icon_button(
                "mdi-folder-open-outline",
                "Show the geometry already written for the included structures",
                click=self.ctrl.load_existing,
                disabled=("busy",),
            )
            _icon_button("mdi-cube-outline", "Build the geometry of every included structure", click=(self.ctrl.build, "[null]"), disabled=("busy",))
            _icon_button(
                "mdi-vector-difference",
                "Resolve overclosures: move the structures with geometry apart where they overlap",
                click=self.ctrl.resolve_overclosures,
                disabled=("busy",),
            )
            _icon_button("mdi-cube-scan", "Mesh every structure with geometry", click=(self.ctrl.mesh, "[null]"), disabled=("busy",))
            _icon_button("mdi-stop-circle-outline", "Cancel the running job", click=self.ctrl.cancel, v_if="busy", color="error")
            v3.VDivider(vertical=True, classes="mx-2")
            v3.VSelect(
                v_model=("display",),
                items=("display_items",),
                density="compact",
                hide_details=True,
                variant="outlined",
                style="max-width: 170px;",
            )
            with _tooltip(
                "Color meshes by structure or by a quality metric of their elements: the scaled Jacobian is the smallest at "
                + "any corner or Gauss point; the dihedral angle measures tetrahedra only"
            ):
                v3.VSelect(
                    v_model=("color_by",),
                    items=("color_by_items",),
                    v_bind="props",
                    prepend_inner_icon="mdi-palette-outline",
                    density="compact",
                    hide_details=True,
                    variant="outlined",
                    classes="ml-2",
                    style="max-width: 250px;",
                )
            _icon_button(
                "mdi-box-cutter",
                "Clip by a plane, keeping whole cells (crinkle clip)",
                click="clip = !clip",
                color=("clip ? 'primary' : ''",),
                classes="ml-2",
            )
            with html.Div(v_if="clip", classes="d-flex align-center"):
                _icon_button(
                    ("clip_fixed ? 'mdi-pin' : 'mdi-pin-off-outline'",),
                    "Unpinned, the plane faces the viewer as the view turns; pinned, it stays where it is and turns with the geometry",
                    click="clip_fixed = !clip_fixed",
                    color=("clip_fixed ? 'primary' : ''",),
                )
                with _tooltip("Move the plane toward or away from the viewer"):
                    v3.VSlider(
                        v_model=("clip_offset",),
                        v_bind="props",
                        min=-1,
                        max=1,
                        step=0.01,
                        density="compact",
                        hide_details=True,
                        style="width: 140px;",
                    )
            _icon_button("mdi-crop-free", "Reset the camera", click=self.ctrl.reset_camera)
            _icon_button("mdi-console", "Show or hide the log", click="show_log = !show_log")
            v3.VProgressLinear(v_if="busy", indeterminate=True, absolute=True, location="bottom", color="primary")

    def _tree(self) -> None:
        with v3.VList(density="compact", opened=("tree_open",), update_opened="tree_open = $event", nav=True):
            v3.VListItem(
                title="Settings",
                prepend_icon="mdi-cog-outline",
                active=("selected == 'settings'",),
                click=(self.ctrl.select, "['settings']"),
            )
            with v3.VListGroup(value="labels"):
                with v3.Template(v_slot_activator="{ props }"):
                    v3.VListItem(v_bind="props", title="Labels", prepend_icon="mdi-layers-outline")
                v3.VListItem(title="Add Label", prepend_icon="mdi-plus", click=self.ctrl.add_label, disabled=("busy",))
                with v3.VListItem(v_if="label_problems.length"):
                    v3.VAlert(
                        v_for="(problem, i) in label_problems",
                        key="i",
                        text=("problem",),
                        type="warning",
                        density="compact",
                        variant="tonal",
                        classes="mb-1",
                    )
                with v3.VListGroup(v_for="entry in labels", key="entry.id", value=("'label-' + entry.id",)):
                    with (
                        v3.Template(v_slot_activator="{ props }"),
                        v3.VListItem(
                            v_bind="props",
                            title=("entry.structure ? structure_titles[entry.structure] : 'Unassigned'",),
                            subtitle=("entry.name + ' : ' + entry.file_value + ' → ' + entry.value",),
                        ),
                        v3.Template(v_slot_prepend=True),
                    ):
                        v3.VIcon(
                            icon=("entry.include ? 'mdi-checkbox-marked-outline' : 'mdi-checkbox-blank-outline'",),
                            color=("entry.include ? 'primary' : ''",),
                        )
                    with v3.VListItem():
                        v3.VSelect(
                            label="Structure Name",
                            items=("structure_items",),
                            model_value=("entry.structure",),
                            update_modelValue=(self.ctrl.update_label, "[entry.id, 'structure', $event]"),
                            density="compact",
                            hide_details=True,
                            clearable=True,
                            classes="my-1",
                        )
                    with v3.VListItem():
                        v3.VTextField(
                            label="Label Value",
                            type="number",
                            min=1,
                            step=1,
                            model_value=("entry.value",),
                            update_modelValue=(self.ctrl.update_label, "[entry.id, 'value', $event]"),
                            density="compact",
                            hide_details=True,
                            classes="my-1",
                        )
                    with v3.VListItem(), html.Div(style="display: flex; align-items: center;"):
                        v3.VCheckbox(
                            label="Include Label",
                            model_value=("entry.include",),
                            update_modelValue=(self.ctrl.update_label, "[entry.id, 'include', $event]"),
                            density="compact",
                            hide_details=True,
                        )
                        _icon_button("mdi-delete-outline", "Remove this label", click=(self.ctrl.remove_label, "[entry.id]"), disabled=("busy",))
            with v3.VListGroup(value="structures"):
                with v3.Template(v_slot_activator="{ props }"):
                    v3.VListItem(v_bind="props", title="Structures", prepend_icon="mdi-bone")
                for structure in STRUCTURES:
                    self._structure_branch(structure)

    def _structure_branch(self, structure: Structure) -> None:
        with v3.VListGroup(value=f"structure-{structure}"):
            with (
                v3.Template(v_slot_activator="{ props, isOpen }"),
                v3.VListItem(
                    v_bind="props",
                    title=structure_title(structure),
                    subtitle=(f"label_structures.includes('{structure}') ? '' : 'No included label'",),
                ),
                v3.Template(v_slot_append=True),
            ):
                # Clicks on these icons must not also fold the branch.
                with html.Div(click="$event.stopPropagation()", style="display: flex; align-items: center;"):
                    v3.VIcon(icon="mdi-cube-outline", size="small", v_if=f"status['{structure}'].geometry", color="primary", classes="mr-1")
                    v3.VIcon(icon="mdi-cube-scan", size="small", v_if=f"status['{structure}'].mesh", color="primary", classes="mr-1")
                    _icon_button(
                        (f"solo == '{structure}' ? 'mdi-target-account' : 'mdi-target'",),
                        "Show only this structure, or all again",
                        click=(self.ctrl.toggle_solo, f"['{structure}']"),
                        color=(f"solo == '{structure}' ? 'primary' : ''",),
                        size="small",
                    )
                    _icon_button(
                        (f"hidden.includes('{structure}') ? 'mdi-eye-off-outline' : 'mdi-eye-outline'",),
                        "Show or hide",
                        click=(self.ctrl.toggle_visibility, f"['{structure}']"),
                        size="small",
                    )
                # The append slot replaces the group's own fold icon.
                v3.VIcon(icon=("isOpen ? 'mdi-chevron-up' : 'mdi-chevron-down'",), classes="ml-1")
            for stage, icon in (("geometry", "mdi-cube-outline"), ("mesh", "mdi-cube-scan")):
                node = f"{stage}:{structure}"
                v3.VListItem(
                    title=stage.capitalize(),
                    prepend_icon=icon,
                    active=(f"selected == '{node}'",),
                    click=(self.ctrl.select, f"['{node}']"),
                )

    def _settings_panel(self) -> None:
        with html.Div(v_if="selected_kind == 'settings'"):
            html.Div("Settings", classes="text-h6 mb-2")
            with html.Div(style="display: flex; align-items: center;"):
                _field(
                    settings.FieldSpec(
                        key=settings.OUTPUT_DIRECTORY_KEY,
                        name="output_directory",
                        section=None,
                        label="Output directory",
                        description="Directory to write geometry and meshes in, created if missing",
                        widget="text",
                    )
                )
                _icon_button("mdi-folder-outline", "Choose the output directory", click=self.ctrl.browse_output_directory)
            html.Div("Geometry", classes="text-subtitle-2 mt-2")
            _alert("form_errors.geo")
            for spec in settings.top_fields("geometry"):
                _field(spec)
            html.Div("Overclosure", classes="text-subtitle-2 mt-2")
            html.Div(
                "Moves structures apart where their geometry overlaps: cartilage shares overlaps with cartilage by thickness, "
                + "and every meniscus and bone moves out of the structures before it in the priority. "
                + "Enabled, building every structure resolves them too.",
                classes="text-caption mb-2",
            )
            v3.VBtn(
                "Resolve overclosures",
                prepend_icon="mdi-vector-difference",
                color="primary",
                variant="tonal",
                click=self.ctrl.resolve_overclosures,
                disabled=("busy",),
                classes="mb-2",
            )
            _alert(f"form_errors['{settings.section_key('geometry', settings.OVERCLOSURE_SECTION)}']")
            for spec in settings.section_fields("geometry", settings.OVERCLOSURE_SECTION):
                _field(spec)
            html.Div("Mesh", classes="text-subtitle-2 mt-2")
            _alert("form_errors.mesh")
            for spec in settings.top_fields("mesh"):
                _field(spec)

        for kind in ("geometry", "mesh"):
            with html.Div(v_if=f"selected_kind == '{kind}'"):
                html.Div(f"{{{{ structure_titles[selected_structure] }}}} {kind}", classes="text-h6 mb-2")
                with html.Div(classes="d-flex flex-wrap ga-2 mb-2"):
                    if kind == "geometry":
                        v3.VBtn(
                            "Build", prepend_icon="mdi-play", color="primary", click=(self.ctrl.build, "[selected_structure]"), disabled=("busy",)
                        )
                    else:
                        v3.VBtn("Mesh", prepend_icon="mdi-play", color="primary", click=(self.ctrl.mesh, "[selected_structure]"), disabled=("busy",))
                    v3.VBtn("Defaults", prepend_icon="mdi-restore", variant="tonal", click=(self.ctrl.reset_section, f"['{kind}', selected_section]"))
                with (
                    html.Div(v_if=f"{_js_list(OWN_SECTION_STRUCTURES)}.includes(selected_structure)"),
                    _tooltip(
                        "Give this structure settings of its own, starting from the shared ones, or share them again. "
                        + "Its own settings are kept while it shares"
                    ),
                ):
                    v3.VSwitch(
                        v_bind="props",
                        label="Own settings",
                        model_value=(f"own_sections['{kind}'].includes(selected_structure)",),
                        update_modelValue=(self.ctrl.set_own_section, f"['{kind}', selected_structure, $event]"),
                        color="primary",
                        density="compact",
                        hide_details=True,
                        inset=True,
                    )
                v3.VAlert(
                    v_if="selected_shared.length",
                    text=("'These settings are shared with ' + selected_shared.map(s => structure_titles[s]).join(', ')",),
                    type="info",
                    density="compact",
                    variant="tonal",
                    classes="mb-2",
                )
                if kind == "mesh":
                    self._mesh_structure_fields()
                for section in settings.structure_section_names(kind):
                    with html.Div(v_if=f"selected_section == '{section}'"):
                        _alert(f"form_errors['{settings.section_key(kind, section)}']")
                        for spec in settings.section_fields(kind, section):
                            _field(spec)

    def _mesh_structure_fields(self) -> None:
        bones = _js_list(BONE_STRUCTURES)
        with (
            html.Div(v_if=f"!{bones}.includes(selected_structure)"),
            _tooltip("Element type of this structure; Default takes the mesh element type in Settings"),
        ):
            v3.VSelect(
                v_bind="props",
                label="Element type",
                items=("[{title: 'Default (' + mesh__element_type + ')', value: ''}, ...element_type_items]",),
                model_value=("element_types[selected_structure] || ''",),
                update_modelValue=(self.ctrl.set_element_type, "[selected_structure, $event]"),
                density="compact",
                classes="mb-1",
            )
        with _tooltip("B-rep or bone surface to mesh; empty takes the geometry this GUI writes"):
            v3.VTextField(
                v_bind="props",
                label="Geometry file",
                model_value=("brep_paths[selected_structure] || ''",),
                placeholder=("expected_geometry[selected_structure] || ''",),
                persistent_placeholder=True,
                update_modelValue=(self.ctrl.set_brep_path, "[selected_structure, $event]"),
                density="compact",
                clearable=True,
                classes="mb-1",
            )

    def _log_panel(self) -> None:
        with v3.VCard(v_show="show_log", style="flex: none; height: 200px; display: flex; flex-direction: column;", rounded=0, border=True):
            with html.Div(classes="d-flex align-center px-2", style="flex: none;"):
                html.Div("Log", classes="text-subtitle-2")
                v3.VSpacer()
                html.Div("{{ busy_text }}", classes="text-caption mr-2")
                _icon_button("mdi-delete-sweep-outline", "Clear the log", click="log_lines = []", size="small")
            # column-reverse keeps the scroll at the newest line.
            with (
                html.Div(style="flex: 1 1 0; min-height: 0; overflow: auto; display: flex; flex-direction: column-reverse;", classes="px-2"),
                html.Div(classes="kneemesh-log"),
            ):
                html.Div("{{ line.text }}", v_for="(line, i) in log_lines", key="i", style=("{color: line.color}",))

    def _file_dialog(self) -> None:
        with v3.VDialog(v_model=("dialog_open",), max_width=720, scrollable=True), v3.VCard(title=("dialog_title",)):
            with v3.VCardText(style="height: 480px; display: flex; flex-direction: column;"):
                with html.Div(classes="d-flex align-center", style="flex: none;"):
                    _icon_button("mdi-arrow-up", "Parent directory", click=self.ctrl.dialog_up)
                    v3.VTextField(
                        v_model=("dialog_directory",),
                        label="Directory",
                        density="compact",
                        hide_details=True,
                        append_inner_icon="mdi-arrow-right",
                        click_appendInner=(self.ctrl.dialog_open_directory, "[dialog_directory]"),
                        keyup_enter=(self.ctrl.dialog_open_directory, "[dialog_directory]"),
                        __events=[("keyup_enter", "keyup.enter")],
                    )
                with v3.VList(density="compact", style="flex: 1 1 0; min-height: 0; overflow: auto;", classes="my-2", border=True):
                    v3.VListItem(
                        v_for="item in dialog_entries",
                        key="item.path",
                        title=("item.name",),
                        prepend_icon=("item.directory ? 'mdi-folder' : 'mdi-file-outline'",),
                        active=("dialog_selection == item.path",),
                        click=(self.ctrl.dialog_click, "[item]"),
                    )
                v3.VTextField(
                    v_if="dialog_mode == 'save'",
                    v_model=("dialog_filename",),
                    label="File name",
                    density="compact",
                    hide_details=True,
                    style="flex: none;",
                )
                with _tooltip("Upload files from this computer into the directory shown, such as a .mhd with its .raw"):
                    v3.VFileInput(
                        v_model=("dialog_upload",),
                        v_bind="props",
                        label="Upload to this directory",
                        multiple=True,
                        density="compact",
                        hide_details=True,
                        prepend_icon="mdi-upload",
                        classes="mt-2",
                        style="flex: none;",
                    )
                v3.VAlert(v_if="dialog_error", text=("dialog_error",), type="error", density="compact", variant="tonal", classes="mt-2")
            with v3.VCardActions():
                v3.VBtn(
                    "Download",
                    prepend_icon="mdi-download",
                    v_if="dialog_selection",
                    click="utils.download(dialog_selection_name, trigger('kneemesh_download', [dialog_selection]), 'application/octet-stream')",
                )
                v3.VSpacer()
                v3.VBtn("Cancel", click="dialog_open = false")
                v3.VBtn(
                    text=("dialog_mode == 'save' ? 'Save' : dialog_mode == 'directory' ? 'Choose' : 'Open'",),
                    color="primary",
                    click=self.ctrl.dialog_confirm,
                )

    # Labels

    def add_label(self) -> None:
        """Pick a LabelMap to import."""
        self._open_dialog("Add Label", "open", sorted(SUPPORTED_SUFFIXES), self._load_label_file)

    async def _load_label_file(self, path: str) -> None:
        with self.state:
            self.state.busy_text = f"Reading {Path(path).name}"
        try:
            image = await asyncio.to_thread(read_label_map, path)
            values = await asyncio.to_thread(label_values, image)
        except (OSError, ValueError, RuntimeError) as error:
            self._notify(f"Could not read {path}: {error}")
            return
        finally:
            with self.state:
                self.state.busy_text = ""
        if not values:
            self._notify(f"{path} holds no labels")
            return
        entries = new_entries(path, values, self.state.labels)
        with self.state:
            self.state.labels = [*self.state.labels, *entries]
            self.state.tree_open = sorted({*self.state.tree_open, "labels"})
            self._labels_changed()
        self._log("INFO", f"Imported {path} with values {', '.join(map(str, values))}")
        await self._preview_labels(entries, image)

    async def _preview_labels(self, entries: Sequence[LabelEntry], image: sitk.Image | None = None) -> None:
        """Show the surfaces of entries' labels; image, if given, is the LabelMap all of them come from."""
        with self.state:
            self.state.busy_text = "Extracting label surfaces"
        images = {entries[0]["path"]: image} if image is not None and entries else {}
        try:
            for entry in entries:
                cache = (entry["path"], entry["file_value"])
                if cache not in self._surfaces:
                    if entry["path"] not in images:
                        images[entry["path"]] = await asyncio.to_thread(read_label_map, entry["path"])
                    self._surfaces[cache] = await asyncio.to_thread(label_surface, images[entry["path"]], entry["file_value"])
                self.scene.set_label(entry["id"], self._surfaces[cache])
                with self.state:
                    self._refresh_scene(reset=True)
        except (OSError, ValueError, RuntimeError) as error:
            self._notify(f"Could not preview a label: {error}")
        finally:
            with self.state:
                self.state.busy_text = ""

    def update_label(self, entry_id: int, field: str, value: object) -> None:
        """Change an imported label's structure, value, or whether it is included."""
        labels = [dict(entry) for entry in self.state.labels]
        entry = next((entry for entry in labels if entry["id"] == entry_id), None)
        if entry is None:
            return
        if field == "value":
            try:
                value = int(str(value))
            except ValueError:
                return
        elif field == "include":
            value = bool(value)
        elif field == "structure" and value not in STRUCTURES:
            value = None
        entry[field] = value
        self.state.labels = labels
        self._labels_changed()

    def remove_label(self, entry_id: int) -> None:
        """Remove an imported label."""
        self.state.labels = [entry for entry in self.state.labels if entry["id"] != entry_id]
        self.scene.set_label(entry_id, None)
        self._labels_changed()

    def _labels_changed(self) -> None:
        entries: list[LabelEntry] = self.state.labels
        self.state.label_problems = entry_problems(entries) if entries else []
        self.state.label_structures = sorted({entry["structure"] for entry in entries if entry["include"] and entry["structure"] is not None})
        self._refresh_scene()

    # Selection and per-structure settings

    def select(self, node: str) -> None:
        """Show the settings of a tree node: settings, or geometry:<structure> or mesh:<structure>."""
        kind, _, structure = node.partition(":")
        self.state.selected = node
        self.state.selected_kind = kind
        self.state.selected_structure = structure or None
        if structure in STRUCTURES and kind in ("geometry", "mesh"):
            own = self.state.own_sections[kind]
            self.state.selected_section = settings.structure_section(structure, own)
            self.state.selected_shared = settings.sharing_structures(structure, own)
        else:
            self.state.selected_section = None
            self.state.selected_shared = []

    def set_own_section(self, kind: settings.Kind, structure: Structure, own: bool) -> None:
        """Give a structure settings of its own, starting from those it shares the first time, or let it share them again."""
        if structure not in OWN_SECTION_STRUCTURES:
            return
        sections = {name: list(structures) for name, structures in self.state.own_sections.items()}
        if own and structure not in sections[kind]:
            # Its own section starts from the shared one's current values, unless it was edited before.
            if structure not in self._edited_sections[kind]:
                shared = STRUCTURE_SECTIONS[structure]
                self.state.update(
                    {spec.key: self.state[settings.state_key(kind, shared, spec.name)] for spec in settings.section_fields(kind, structure)}
                )
                self._edited_sections[kind].add(structure)
            sections[kind].append(structure)
        elif not own:
            sections[kind] = [name for name in sections[kind] if name != structure]
        self.state.own_sections = sections
        self.select(self.state.selected)

    def set_element_type(self, structure: Structure, element_type: str | None) -> None:
        """Set a structure's own element type, or with an empty one, let it take the mesh's."""
        element_types = {key: value for key, value in self.state.element_types.items() if key != structure}
        if element_type:
            element_types[structure] = element_type
        self.state.element_types = element_types

    def set_brep_path(self, structure: Structure, path: str | None) -> None:
        """Set the file a structure is meshed from, or with an empty one, the geometry this GUI writes."""
        paths = {key: value for key, value in self.state.brep_paths.items() if key != structure}
        if path:
            paths[structure] = path
        self.state.brep_paths = paths

    def toggle_visibility(self, structure: Structure) -> None:
        """Show or hide a structure in the view."""
        hidden = list(self.state.hidden)
        self.state.hidden = [name for name in hidden if name != structure] if structure in hidden else [*hidden, structure]

    def toggle_solo(self, structure: Structure) -> None:
        """Show only a structure in the view, or every structure again if it was the only one shown."""
        self.state.solo = None if self.state.solo == structure else structure
        if self.state.solo is not None:
            self.state.hidden = [name for name in self.state.hidden if name != structure]
        self._refresh_scene(reset=True)

    def reset_section(self, kind: settings.Kind, section: str) -> None:
        """Restore a section's defaults."""
        defaults = settings.state_values(kind)
        self.state.update({spec.key: defaults[spec.key] for spec in settings.section_fields(kind, section)})

    # Validation

    def _validate(self) -> None:
        errors: dict[str, str] = {}
        geometry = self._geometry_config(errors)
        mesh_values = settings.config_values("mesh", self._form_state(), self.state.own_sections["mesh"])
        mesh_values["element_types"] = self.state.element_types or None
        try:
            BrepMesh.model_validate(mesh_values)
        except ValidationError as error:
            errors.update(settings.validation_errors("mesh", error))
        if geometry is not None:
            self.state.expected_geometry = {structure: str(geometry.output_path(structure)) for structure in STRUCTURES}
        self.state.form_errors = errors

    def _form_state(self) -> dict[str, Any]:
        """The current value of every form field's state key, including updates not yet flushed, which state.to_dict leaves out."""
        keys = [settings.OUTPUT_DIRECTORY_KEY, *(spec.key for kind in ("geometry", "mesh") for spec in settings.all_fields(kind))]
        return {key: self.state[key] for key in keys}

    def _geometry_config(self, errors: dict[str, str] | None = None) -> LabelMapBrep | None:
        """The geometry settings as a LabelMapBrep without a segmentation, or None, with its errors put in errors, if they are invalid."""
        values = settings.config_values("geometry", self._form_state(), self.state.own_sections["geometry"])
        # The GUI composes the segmentation itself, so there are no label_maps for assembled_label_map to go with.
        values.pop("assembled_label_map", None)
        try:
            return LabelMapBrep.model_validate(values)
        except ValidationError as error:
            if errors is not None:
                errors.update(settings.validation_errors("geometry", error))
            return None

    def _mesh_config(self, breps: dict[Structure, str], assembled: bool) -> BrepMesh:
        values = settings.config_values("mesh", self._form_state(), self.state.own_sections["mesh"])
        values["breps"] = breps
        values["element_types"] = self.state.element_types or None
        if not assembled:
            values["assembled_mesh"] = None
        return BrepMesh.model_validate(values)

    def _geometry_path(self, structure: Structure) -> str | None:
        return self.state.brep_paths.get(structure) or self.state.expected_geometry.get(structure)

    # Jobs

    def build(self, structure: Structure | None) -> None:
        """Build the geometry of a structure, or of every included one."""
        asynchronous.create_task(self._build(structure))

    async def _build(self, structure: Structure | None) -> None:
        entries: list[LabelEntry] = self.state.labels
        included: list[Structure] = [entry["structure"] for entry in entries if entry["include"] and entry["structure"] is not None]
        structures: list[Structure] = [structure] if structure is not None else included
        problems = entry_problems(entries) + missing_bones(structures, entries)
        if structure is not None and structure not in included:
            problems.insert(0, f"Include a label for {structure_title(structure)} first")
        if not structures:
            problems.append("Include a label for a structure first")
        config = self._geometry_config()
        if config is None:
            problems.append("Fix the geometry settings first")
        if problems or config is None:
            self._notify("; ".join(dict.fromkeys(problems)))
            return
        geometry = config.model_dump(mode="json")
        assembled: str | None = self.state.geo__assembled_label_map or None
        job = {
            "kind": "geometry",
            "config": geometry,
            "entries": entries,
            "structures": structures,
            "interpolator": geometry["interpolator"],
            "padding": geometry["padding"],
            "assembled_label_map": assembled,
            "preview_directory": str(self.preview_directory),
        }
        result = await self._run(job, f"Building {', '.join(map(structure_title, structures))}")
        if result is None:
            return
        self._wrote(*(paths["output"] for paths in result.values()))
        if assembled:
            self._wrote(assembled, segmentation_labels_path(assembled))
        for name, paths in result.items():
            name = cast(Structure, name)
            self.scene.set_geometry(name, await asyncio.to_thread(_read_geometry, paths["preview"]))
            self.scene.clear("mesh", name)
            self._set_status(name, geometry=True, mesh=False)
        with self.state:
            self._refresh_scene(reset=True)
        self._notify(f"Built {', '.join(map(structure_title, result))}", "success")
        if structure is None and config.overclosure.enabled:
            await self._resolve_overclosures()

    def resolve_overclosures(self) -> None:
        """Move every structure with geometry apart from the others where they overlap, by the overclosure settings."""
        asynchronous.create_task(self._resolve_overclosures())

    async def _resolve_overclosures(self) -> None:
        geometry = {name: path for name in STRUCTURES if (path := self._geometry_path(name)) and Path(path).is_file()}
        if len(geometry) < 2:
            self._notify("Build the geometry of at least two structures first")
            return
        config = self._geometry_config()
        if config is None:
            self._notify("Fix the geometry settings first")
            return
        job = {
            "kind": "overclosure",
            "config": config.model_dump(mode="json"),
            "geometry": geometry,
            "preview_directory": str(self.preview_directory),
        }
        result = await self._run(job, "Resolving overclosures")
        if result is None:
            return
        self._wrote(*(paths["output"] for paths in result.values()))
        for name, paths in result.items():
            name = cast(Structure, name)
            self.scene.set_geometry(name, await asyncio.to_thread(_read_geometry, paths["preview"]))
            # Moved geometry leaves its mesh out of date.
            self.scene.clear("mesh", name)
            self._set_status(name, geometry=True, mesh=False)
        with self.state:
            self._refresh_scene()
        moved = ", ".join(map(structure_title, result))
        self._notify(f"Resolved overclosures, moving {moved}" if result else "No structures overlap", "success")

    def mesh(self, structure: Structure | None) -> None:
        """Mesh a structure's geometry, or that of every structure with geometry."""
        asynchronous.create_task(self._mesh(structure))

    async def _mesh(self, structure: Structure | None) -> None:
        candidates: list[Structure] = [structure] if structure is not None else list(STRUCTURES)
        breps: dict[Structure, str] = {}
        for name in candidates:
            path = self._geometry_path(name)
            if path and Path(path).is_file():
                breps[name] = path
        if not breps:
            self._notify(
                f"No geometry to mesh at {self._geometry_path(structure)}; build it first"
                if structure is not None
                else "No geometry to mesh; build some first"
            )
            return
        try:
            config = self._mesh_config(breps, assembled=structure is None)
        except ValidationError as error:
            self._notify(f"Fix the mesh settings first: {'; '.join(settings.validation_errors('mesh', error).values())}")
            return
        job = {"kind": "mesh", "config": config.model_dump(mode="json"), "preview_directory": str(self.preview_directory)}
        result = await self._run(job, f"Meshing {', '.join(map(structure_title, breps))}")
        if result is None:
            return
        self._wrote(*(paths["output"] for paths in result.values()), config.assembled_mesh)
        sheets: list[Structure] = [name for name in BONES if name in result]
        self._wrote(*(config.partition_path(name) for name in sheets if config.section(name, CartilageMesh).save_partition))
        for name, paths in result.items():
            name = cast(Structure, name)
            self.scene.set_mesh(name, await asyncio.to_thread(read_mesh_preview, paths["preview"]))
            self._set_status(name, mesh=True)
        with self.state:
            self._refresh_scene()
        self._notify(f"Meshed {', '.join(map(structure_title, result))}", "success")

    def load_existing(self) -> None:
        """Show the geometry files already written for the included structures, or given as their geometry files."""
        asynchronous.create_task(self._load_existing())

    async def _load_existing(self) -> None:
        structures = {*self.state.label_structures, *self.state.brep_paths}
        found = {structure: path for structure in structures if (path := self._geometry_path(structure)) and Path(path).is_file()}
        if not found:
            self._notify("None of the included structures' geometry files exist yet")
            return
        breps = {structure: path for structure, path in found.items() if Path(path).suffix.lower() == ".brep"}
        previews = {structure: path for structure, path in found.items() if structure not in breps}
        if breps:
            job = {"kind": "preview", "breps": breps, "preview_directory": str(self.preview_directory)}
            result = await self._run(job, "Previewing geometry")
            if result is None:
                return
            previews.update({structure: paths["preview"] for structure, paths in result.items()})
        for structure, path in previews.items():
            try:
                self.scene.set_geometry(structure, await asyncio.to_thread(_read_geometry, path))
            except (OSError, ValueError) as error:
                self._notify(f"Could not show {path}: {error}")
                continue
            self._set_status(structure, geometry=True)
        with self.state:
            self._refresh_scene(reset=True)

    async def _run(self, job: dict[str, Any], description: str) -> dict[str, Any] | None:
        if self.jobs.busy:
            self._notify("A job is already running")
            return None
        with self.state:
            self.state.busy = True
            self.state.busy_text = description
        self._log("INFO", description)
        try:
            return await self.jobs.run(job, self._log)
        except JobError as error:
            self._log("ERROR", str(error))
            self._notify(str(error))
            return None
        finally:
            with self.state:
                self.state.busy = False
                self.state.busy_text = ""

    def cancel(self) -> None:
        """Cancel the running job."""
        self.jobs.cancel()

    def _set_status(self, structure: str, **stages: bool) -> None:
        status = {name: dict(value) for name, value in self.state.status.items()}
        status[structure].update(stages)
        with self.state:
            self.state.status = status

    # Configuration files

    def import_config(self) -> None:
        """Pick a kneemesh-brep, kneemesh-mesh, or kneemesh-assemble JSON configuration to import."""
        self._open_dialog("Import configuration", "open", [".json"], self._import_config)

    async def _import_config(self, path: str) -> None:
        try:
            text = Path(path).read_text(encoding="utf-8")
            config = _parse_config(text)
        except (OSError, ValueError) as error:
            self._notify(f"Could not import {path}: {error}")
            return
        entries: list[LabelEntry] | None = None
        try:
            if isinstance(config, LabelMapBrep) and (config.label_map is not None or config.label_maps is not None):
                entries = await asyncio.to_thread(config_entries, config.label_map, config.labels, config.label_maps, _values_of)
            elif isinstance(config, LabelMapAssembly) and config.label_maps is not None:
                entries = await asyncio.to_thread(config_entries, None, None, config.label_maps, _values_of)
        except (OSError, ValueError, RuntimeError) as error:
            self._notify(f"Could not read the labels {path} names: {error}")
            return
        with self.state:
            if isinstance(config, LabelMapBrep | BrepMesh):
                kind: settings.Kind = "geometry" if isinstance(config, LabelMapBrep) else "mesh"
                self.state.own_sections = {**self.state.own_sections, kind: settings.own_sections(config)}
                self._edited_sections[kind] = set(settings.own_sections(config))
            if isinstance(config, LabelMapBrep):
                self.state.update(settings.state_values("geometry", config))
                self.state.geo__assembled_label_map = str(config.assembled_label_map) if config.assembled_label_map is not None else None
            elif isinstance(config, BrepMesh):
                self.state.update(settings.state_values("mesh", config))
                self.state.element_types = dict(config.element_types or {})
                self.state.brep_paths = {structure: str(brep) for structure, brep in (config.breps or {}).items()}
            else:
                self.state.geo__interpolator = config.interpolator
                self.state.geo__padding = config.padding
                self.state.geo__assembled_label_map = str(config.output)
            if entries is not None:
                for entry in self.state.labels:
                    self.scene.set_label(entry["id"], None)
                self.state.labels = entries
                self._labels_changed()
        self._log("INFO", f"Imported {type(config).__name__} configuration from {path}")
        self._notify(f"Imported {path}", "success")
        if entries:
            await self._preview_labels(entries)

    def export_geometry(self) -> None:
        """Pick where to write the geometry settings and labels as a kneemesh-brep configuration."""
        self._open_dialog("Export geometry configuration", "save", [".json"], self._export_geometry, "brep.json")

    async def _export_geometry(self, path: str) -> None:
        entries: list[LabelEntry] = self.state.labels
        if problems := entry_problems(entries):
            self._notify(f"Could not export: {'; '.join(problems)}")
            return
        values = settings.config_values("geometry", self._form_state(), self.state.own_sections["geometry"])
        assembled = values.pop("assembled_label_map", None)
        try:
            sources = await asyncio.to_thread(brep_sources, entries)
            if sources is None:
                # No configuration reads these labels as they are, so write the segmentation they compose for it to read.
                destination = Path(path)
                segmentation_path = destination.with_name(f"{destination.stem}_segmentation.nrrd")
                segmentation = await asyncio.to_thread(compose_segmentation, entries, values["interpolator"], values["padding"])
                await asyncio.to_thread(write_segmentation, segmentation, segmentation_path)
                sources = {"label_map": str(segmentation_path), "labels": segmentation.labels}
                self._log("INFO", f"Wrote the composed segmentation to {segmentation_path}")
                self._wrote(segmentation_path, segmentation_labels_path(segmentation_path))
            elif "label_maps" in sources:
                values["assembled_label_map"] = assembled
            values.update(sources)
            config = LabelMapBrep.model_validate(values)
            Path(path).write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")
            self._wrote(path)
        except ValidationError as error:
            self._notify(f"Could not export: {'; '.join(settings.validation_errors('geometry', error).values())}")
            return
        except (OSError, ValueError, RuntimeError) as error:
            self._notify(f"Could not export: {error}")
            return
        self._notify(f"Exported {path}", "success")

    def export_mesh(self) -> None:
        """Pick where to write the mesh settings as a kneemesh-mesh configuration."""
        self._open_dialog("Export mesh configuration", "save", [".json"], self._export_mesh, "mesh.json")

    def _export_mesh(self, path: str) -> None:
        structures: list[Structure] = [
            structure for structure in STRUCTURES if structure in self.state.label_structures or structure in self.state.brep_paths
        ]
        breps: dict[Structure, str] = {structure: geometry for structure in structures if (geometry := self._geometry_path(structure))}
        try:
            config = self._mesh_config(breps, assembled=True)
            Path(path).write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")
            self._wrote(path)
        except ValidationError as error:
            self._notify(f"Could not export: {'; '.join(settings.validation_errors('mesh', error).values())}")
            return
        except OSError as error:
            self._notify(f"Could not export: {error}")
            return
        self._notify(f"Exported {path}", "success")

    def browse_output_directory(self) -> None:
        """Pick the output directory."""
        self._open_dialog("Output directory", "directory", [], self._set_output_directory)

    def _set_output_directory(self, path: str) -> None:
        self.state[settings.OUTPUT_DIRECTORY_KEY] = path

    # File dialog

    def _open_dialog(self, title: str, mode: str, suffixes: Sequence[str], callback: Callable[[str], object], filename: str = "") -> None:
        self._dialog_callback = callback
        self.state.update(
            {
                "dialog_title": title,
                "dialog_mode": mode,
                "dialog_suffixes": list(suffixes),
                "dialog_selection": "",
                "dialog_filename": filename,
                "dialog_error": "",
                "dialog_open": True,
            }
        )
        self.dialog_open_directory(self.state.dialog_directory)

    def dialog_open_directory(self, directory: str) -> None:
        """List a directory in the file dialog."""
        path = Path(directory).expanduser()
        try:
            entries = list_directory(path, self.state.dialog_suffixes)
        except OSError as error:
            self.state.dialog_error = f"Could not list {path}: {error}"
            return
        self.state.update({"dialog_directory": str(path.resolve()), "dialog_entries": entries, "dialog_error": "", "dialog_selection": ""})

    def dialog_up(self) -> None:
        """List the file dialog's parent directory."""
        self.dialog_open_directory(str(Path(self.state.dialog_directory).parent))

    def dialog_click(self, item: dict[str, Any]) -> None:
        """Open a directory, or select a file, in the file dialog."""
        if item["directory"]:
            self.dialog_open_directory(item["path"])
        else:
            self.state.dialog_selection = item["path"]
            self.state.dialog_selection_name = item["name"]
            if self.state.dialog_mode == "save":
                self.state.dialog_filename = item["name"]

    def dialog_confirm(self) -> None:
        """Accept the file dialog's choice and pass it to the action that opened it."""
        mode, directory = self.state.dialog_mode, Path(self.state.dialog_directory)
        if mode == "directory":
            path = str(directory)
        elif mode == "save":
            name = str(self.state.dialog_filename).strip()
            if not name:
                self.state.dialog_error = "Give a file name"
                return
            if not Path(name).suffix and self.state.dialog_suffixes:
                name += self.state.dialog_suffixes[0]
            path = str(directory / name)
        else:
            if not self.state.dialog_selection:
                self.state.dialog_error = "Choose a file"
                return
            path = self.state.dialog_selection
        self.state.dialog_open = False
        callback, self._dialog_callback = self._dialog_callback, None
        if callback is not None:
            result = callback(path)
            if asyncio.iscoroutine(result):
                asynchronous.create_task(result)

    def _upload(self, files: object) -> None:
        """Write files uploaded in the file dialog into its directory, and select the one to open if there is one."""
        if not files:
            return
        uploads = [ClientFile(file) for file in (files if isinstance(files, list) else [files])]
        directory = Path(self.state.dialog_directory)
        written: list[Path] = []
        try:
            for upload in uploads:
                content = upload.content
                if content is None or not upload.name:
                    continue
                # Only the file's own name, so an upload cannot write outside the directory.
                path = directory / Path(upload.name).name
                path.write_bytes(content)
                written.append(path)
        except OSError as error:
            self.state.dialog_error = f"Could not upload: {error}"
            return
        finally:
            self.state.dialog_upload = None
        self._log("INFO", f"Uploaded {', '.join(str(path) for path in written)}")
        self.dialog_open_directory(str(directory))
        suffixes = {suffix.lower() for suffix in self.state.dialog_suffixes}
        openable = [path for path in written if not suffixes or path.suffix.lower() in suffixes]
        if len(openable) == 1:
            self.state.dialog_selection = str(openable[0])
            self.state.dialog_selection_name = openable[0].name
            if self.state.dialog_mode == "save":
                self.state.dialog_filename = openable[0].name

    def download_file(self, path: str) -> object:
        """A server file's contents, for the browser to save."""
        try:
            return self._attachment(Path(path).read_bytes())
        except OSError as error:
            self._notify(f"Could not download {path}: {error}")
            return None

    def download_outputs(self) -> object:
        """A zip of the files this session wrote, for the browser to save."""
        return self._attachment(zip_files(self._written))

    def _attachment(self, content: bytes) -> object:
        """Bytes sent to the browser as a binary attachment rather than encoded in a message."""
        protocol = self.server.protocol
        assert protocol is not None, "The server is not running"
        return protocol.addAttachment(content)

    def _wrote(self, *paths: str | Path | None) -> None:
        """Remember files this session wrote, for download_outputs."""
        for path in paths:
            if path is not None and Path(path).is_file():
                self._written[str(Path(path).resolve())] = None
        self.state.written_count = len(self._written)

    # View and messages

    def reset_camera(self) -> None:
        """Fit the view to what it shows."""
        self.scene.reset_camera()
        self.ctrl.view_update()

    def _set_clipping(self) -> None:
        self.scene.set_clipping(bool(self.state.clip), "fixed" if self.state.clip_fixed else "view", float(self.state.clip_offset))
        self.ctrl.view_update()

    def _refresh_scene(self, reset: bool = False) -> None:
        self.scene.refresh(self.state.labels, self.state.display, self.state.hidden, self.state.solo, self.state.color_by or None)
        if reset:
            self.scene.reset_camera()
        self.ctrl.view_update()

    def _log(self, level: str, message: str) -> None:
        logger.log(level, message)
        line = {"text": f"{level:<8} {message}", "color": _LOG_COLORS.get(level, "")}
        with self.state:
            self.state.log_lines = [*self.state.log_lines[-(_LOG_LINES - 1) :], line]

    def _notify(self, message: str, color: str = "error") -> None:
        if color == "error":
            self._log("ERROR", message)
        with self.state:
            self.state.update({"snackbar_text": message, "snackbar_color": color, "snackbar": True})


def _parse_config(text: str) -> LabelMapBrep | BrepMesh | LabelMapAssembly:
    """The kneemesh-brep, kneemesh-mesh, or kneemesh-assemble configuration in JSON text, whichever validates first.

    Raises:
        ValueError: If the text is not JSON or validates as none of them.
    """
    data = json.loads(text)
    errors: list[str] = []
    for model in (LabelMapBrep, BrepMesh, LabelMapAssembly):
        try:
            return model.model_validate(data)
        except ValidationError as error:
            errors.append(f"not a {model.__name__}: {error.error_count()} errors, first {error.errors()[0]['msg']}")
    raise ValueError("; ".join(errors))


def zip_files(paths: Iterable[str]) -> bytes:
    """A zip of files, each named by its path relative to the deepest directory holding them all."""
    existing = [Path(path) for path in paths if Path(path).is_file()]
    root = Path(os.path.commonpath([path.parent for path in existing])) if existing else Path()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in existing:
            archive.write(path, path.relative_to(root).as_posix())
    return buffer.getvalue()


def _values_of(path: str) -> list[int]:
    return label_values(read_label_map(path))


def _read_geometry(path: str) -> Any:
    """A geometry preview: a B-rep's .vtp or a bone's .stl."""
    return read_surface(path) if Path(path).suffix.lower() == ".stl" else read_poly_data(path)


def _icon_button(icon: str | tuple[str], tooltip: str, **kwargs: Any) -> None:
    with _tooltip(tooltip):
        v3.VBtn(icon=icon, v_bind="props", variant="text", density="comfortable", **kwargs)


@contextmanager
def _tooltip(text: str) -> Generator[None]:
    """A tooltip whose activator is the widget made inside the block, which must take v_bind="props".

    The text goes into a double-quoted template attribute unescaped, so its double quotes, such as a JSON example's, become entities.
    """
    with v3.VTooltip(text=text.replace('"', "&quot;"), location="end", max_width=420, open_delay=400), v3.Template(v_slot_activator="{ props }"):
        yield


def _js_list(values: Sequence[str]) -> str:
    """A JavaScript array of plain strings for a template expression, single-quoted to fit in a double-quoted attribute."""
    return "[" + ", ".join(f"'{value}'" for value in values) + "]"


def _alert(errors: str) -> None:
    v3.VAlert(v_if=errors, text=(errors,), type="error", density="compact", variant="tonal", classes="mb-2")


def _field(spec: settings.FieldSpec) -> None:
    """A widget editing a configuration field, its description as a tooltip and its errors from form_errors."""
    errors = f"form_errors['{spec.key}']"
    common: dict[str, Any] = {
        "v_model": (spec.key,),
        "label": spec.label,
        "density": "compact",
        "error_messages": (f"{errors} || []",),
        "v_bind": "props",
        "classes": "mb-1",
    }
    with _tooltip(spec.description or spec.label):
        if spec.widget == "switch":
            v3.VSwitch(color="primary", hide_details="auto", inset=True, **common)
        elif spec.widget == "select":
            v3.VSelect(items=(_js_list(spec.items),), **common)
        elif spec.widget == "number":
            v3.VTextField(
                type="number",
                step=1 if spec.integer else "any",
                min=spec.minimum,
                max=spec.maximum,
                clearable=spec.optional,
                placeholder="auto" if spec.optional else None,
                persistent_placeholder=spec.optional,
                **common,
            )
        elif spec.widget == "order":
            html.Div(spec.label, v_bind="props", classes="text-body-2")
        else:
            v3.VTextField(clearable=spec.optional, placeholder="none" if spec.optional else None, persistent_placeholder=spec.optional, **common)
    if spec.widget == "order":
        _order(spec)


def _order(spec: settings.FieldSpec) -> None:
    """A list of a field's items in its order, each with buttons moving it up or down, which reorder the state's list in place."""
    key = spec.key

    def swap(offset: int) -> str:
        # A new list with the item at index and its neighbor at index + offset exchanged, so trame sees the change.
        other = f"index + {offset}" if offset > 0 else f"index - {-offset}"
        return f"{key} = {key}.map((item, place, all) => place === index ? all[{other}] : place === {other} ? all[index] : item)"

    title = "(index + 1) + '. ' + (structure_titles[item] || item)"
    with (
        v3.VList(density="compact", classes="mb-1 py-0", border=True, rounded=True),
        v3.VListItem(v_for=f"(item, index) in {key}", key="item", title=(title,)),
        v3.Template(v_slot_append=True),
    ):
        v3.VBtn(icon="mdi-chevron-up", variant="text", density="compact", size="small", disabled=("index === 0",), click=swap(-1))
        v3.VBtn(icon="mdi-chevron-down", variant="text", density="compact", size="small", disabled=(f"index === {key}.length - 1",), click=swap(1))
    _alert(f"form_errors['{key}']")


def main() -> None:
    """Run the GUI's server and open it in a browser, logging to stderr."""
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<level>{level: <8}</level> {message}")
    logger.enable("kneemesh")
    app = KneemeshApp()
    app.server.start()
