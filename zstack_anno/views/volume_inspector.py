from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
except Exception as exc:  # pragma: no cover - exercised through lazy import in controller
    pv = None
    QtInteractor = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from ..utils.volume_utils import VolumeSource


class VolumeInspectorUnavailableError(RuntimeError):
    """Raised when the 3-D inspector backend is unavailable."""


@dataclass
class _NamedSource:
    label: str
    source: VolumeSource


class VolumeInspectorWindow(QMainWindow):
    """Interactive 3-D viewer for raw Z-stacks and segmentation masks."""

    def __init__(
        self,
        *,
        title: str,
        raw_sources: list[VolumeSource],
        mask_sources: list[VolumeSource] | None = None,
        default_raw_label: str | None = None,
        default_mask_label: str | None = None,
    ) -> None:
        if pv is None or QtInteractor is None:
            raise VolumeInspectorUnavailableError(str(_IMPORT_ERROR))

        super().__init__()
        self.setWindowTitle(title)
        self.resize(1320, 960)
        self._raw_sources = [_NamedSource(src.label, src) for src in raw_sources]
        self._mask_sources = [_NamedSource(src.label, src) for src in (mask_sources or [])]
        self._rendered_once = False
        self._build_ui()
        self._populate_sources(default_raw_label, default_mask_label)
        self._render_scene(reset_camera=True)

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)

        controls_row = QHBoxLayout()
        self.raw_combo = QComboBox()
        self.mask_combo = QComboBox()
        self.show_raw_chk = QCheckBox("Show Raw")
        self.show_raw_chk.setChecked(True)
        self.show_mask_chk = QCheckBox("Show Mask")
        self.show_mask_chk.setChecked(True)
        self.reload_btn = QPushButton("Reload")
        self.reset_btn = QPushButton("Reset Camera")
        self.xy_btn = QPushButton("XY")
        self.xz_btn = QPushButton("XZ")
        self.yz_btn = QPushButton("YZ")
        self.iso_btn = QPushButton("Oblique")
        self.screenshot_btn = QPushButton("Screenshot")
        controls_row.addWidget(QLabel("Raw"))
        controls_row.addWidget(self.raw_combo, 2)
        controls_row.addWidget(QLabel("Mask"))
        controls_row.addWidget(self.mask_combo, 2)
        controls_row.addWidget(self.show_raw_chk)
        controls_row.addWidget(self.show_mask_chk)
        controls_row.addWidget(self.reload_btn)
        controls_row.addWidget(self.reset_btn)
        controls_row.addWidget(self.xy_btn)
        controls_row.addWidget(self.xz_btn)
        controls_row.addWidget(self.yz_btn)
        controls_row.addWidget(self.iso_btn)
        controls_row.addWidget(self.screenshot_btn)
        layout.addLayout(controls_row)

        slider_row = QHBoxLayout()
        self.raw_opacity_slider = QSlider(Qt.Horizontal)
        self.raw_opacity_slider.setRange(0, 100)
        self.raw_opacity_slider.setValue(30)
        self.mask_opacity_slider = QSlider(Qt.Horizontal)
        self.mask_opacity_slider.setRange(0, 100)
        self.mask_opacity_slider.setValue(65)
        slider_row.addWidget(QLabel("Raw Opacity"))
        slider_row.addWidget(self.raw_opacity_slider, 2)
        slider_row.addWidget(QLabel("Mask Opacity"))
        slider_row.addWidget(self.mask_opacity_slider, 2)
        layout.addLayout(slider_row)

        self.plotter = QtInteractor(central)
        layout.addWidget(self.plotter.interactor, 1)

        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        self.setCentralWidget(central)

        self.raw_combo.currentTextChanged.connect(self._on_source_changed)
        self.mask_combo.currentTextChanged.connect(self._on_source_changed)
        self.show_raw_chk.toggled.connect(self._on_source_changed)
        self.show_mask_chk.toggled.connect(self._on_source_changed)
        self.raw_opacity_slider.valueChanged.connect(self._on_source_changed)
        self.mask_opacity_slider.valueChanged.connect(self._on_source_changed)
        self.reload_btn.clicked.connect(lambda: self._render_scene(reset_camera=False))
        self.reset_btn.clicked.connect(lambda: self._render_scene(reset_camera=True))
        self.xy_btn.clicked.connect(self._view_xy)
        self.xz_btn.clicked.connect(self._view_xz)
        self.yz_btn.clicked.connect(self._view_yz)
        self.iso_btn.clicked.connect(self._view_iso)
        self.screenshot_btn.clicked.connect(self._save_screenshot)

    def _populate_sources(
        self,
        default_raw_label: str | None,
        default_mask_label: str | None,
    ) -> None:
        self.raw_combo.clear()
        for item in self._raw_sources:
            self.raw_combo.addItem(item.label)

        self.mask_combo.clear()
        self.mask_combo.addItem("(none)")
        for item in self._mask_sources:
            self.mask_combo.addItem(item.label)
        self.mask_combo.setEnabled(bool(self._mask_sources))
        self.show_mask_chk.setEnabled(bool(self._mask_sources))

        if default_raw_label:
            idx = self.raw_combo.findText(default_raw_label)
            if idx >= 0:
                self.raw_combo.setCurrentIndex(idx)
        if default_mask_label:
            idx = self.mask_combo.findText(default_mask_label)
            if idx >= 0:
                self.mask_combo.setCurrentIndex(idx)
            elif self._mask_sources:
                self.mask_combo.setCurrentIndex(1)
        elif self._mask_sources:
            self.mask_combo.setCurrentIndex(1)

    def _named_source_from_combo(
        self,
        combo: QComboBox,
        named_sources: list[_NamedSource],
    ) -> VolumeSource | None:
        label = combo.currentText()
        if not label or label == "(none)":
            return None
        for item in named_sources:
            if item.label == label:
                return item.source
        return None

    def _on_source_changed(self, *_args) -> None:
        self._render_scene(reset_camera=False)

    def _render_scene(self, *, reset_camera: bool) -> None:
        camera_position = self.plotter.camera_position if self._rendered_once and not reset_camera else None
        self.plotter.clear()
        self.plotter.set_background("white")
        self.plotter.add_axes(line_width=2)

        raw_source = self._named_source_from_combo(self.raw_combo, self._raw_sources)
        mask_source = self._named_source_from_combo(self.mask_combo, self._mask_sources)

        try:
            if raw_source is not None and self.show_raw_chk.isChecked():
                self._add_raw_volume(raw_source)
            if mask_source is not None and self.show_mask_chk.isChecked():
                self._add_mask_surface(mask_source)
        except Exception as exc:
            QMessageBox.warning(self, "3D Inspector", f"Failed to render volume:\n{exc}")
            return

        if camera_position is not None:
            self.plotter.camera_position = camera_position
        else:
            self.plotter.view_isometric()
        self.plotter.reset_camera_clipping_range()
        self._rendered_once = True
        self._update_info_label(raw_source, mask_source)

    def _add_raw_volume(self, source: VolumeSource) -> None:
        _add_raw_volume_to_plotter(
            self.plotter,
            source,
            opacity_scale=self.raw_opacity_slider.value() / 100.0,
        )

    def _add_mask_surface(self, source: VolumeSource) -> None:
        _add_mask_surface_to_plotter(
            self.plotter,
            source,
            opacity=self.mask_opacity_slider.value() / 100.0,
        )

    def _view_xy(self) -> None:
        self.plotter.view_xy()

    def _view_xz(self) -> None:
        self.plotter.view_xz()

    def _view_yz(self) -> None:
        self.plotter.view_yz()

    def _view_iso(self) -> None:
        self.plotter.view_isometric()

    def _save_screenshot(self) -> None:
        default_name = _build_snapshot_name(self.raw_combo.currentText(), self.mask_combo.currentText())
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save 3D Screenshot",
            default_name,
            "PNG Images (*.png)",
        )
        if not out_path:
            return
        self.snapshot_current_view(out_path)
        QMessageBox.information(self, "3D Inspector", f"Screenshot saved:\n{out_path}")

    def snapshot_current_view(self, out_path: str) -> None:
        self.plotter.screenshot(out_path)

    def _update_info_label(
        self,
        raw_source: VolumeSource | None,
        mask_source: VolumeSource | None,
    ) -> None:
        lines: list[str] = []
        if raw_source is not None:
            lines.append(
                f"Raw: {raw_source.label} | shape ZYX={raw_source.shape_zyx} | "
                f"spacing XYZ={_format_spacing(raw_source.spacing_xyz)}"
            )
            if raw_source.path:
                lines.append(f"Raw path: {raw_source.path}")
            if raw_source.note:
                lines.append(f"Raw note: {raw_source.note}")
        if mask_source is not None:
            lines.append(
                f"Mask: {mask_source.label} | shape ZYX={mask_source.shape_zyx} | "
                f"spacing XYZ={_format_spacing(mask_source.spacing_xyz)}"
            )
            if mask_source.path:
                lines.append(f"Mask path: {mask_source.path}")
            if mask_source.note:
                lines.append(f"Mask note: {mask_source.note}")
        if not lines:
            lines.append("No volume selected.")
        self.info_label.setText("\n".join(lines))


def _normalize_volume_to_uint8(volume_zyx: np.ndarray) -> np.ndarray:
    arr = np.asarray(volume_zyx, dtype=np.float32)
    if arr.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    lo = float(np.percentile(arr, 1.0))
    hi = float(np.percentile(arr, 99.5))
    if hi <= lo:
        hi = float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return np.rint(arr * 255.0).astype(np.uint8)


def _build_image_data(volume_zyx: np.ndarray, spacing_xyz: tuple[float, float, float]) -> "pv.ImageData":
    volume_xyz = np.transpose(np.asarray(volume_zyx), (2, 1, 0))
    grid = pv.ImageData(dimensions=volume_xyz.shape, spacing=spacing_xyz, origin=(0.0, 0.0, 0.0))
    grid.point_data["values"] = np.ascontiguousarray(volume_xyz).ravel(order="F")
    return grid


def _add_raw_volume_to_plotter(
    plotter,
    source: VolumeSource,
    *,
    opacity_scale: float,
) -> None:
    values = _normalize_volume_to_uint8(source.volume_zyx)
    grid = _build_image_data(values, source.spacing_xyz)
    opacity = [0.0, 0.02 * opacity_scale, 0.10 * opacity_scale, 0.35 * opacity_scale, 0.90 * opacity_scale]
    plotter.add_volume(
        grid,
        scalars="values",
        opacity=opacity,
        cmap="gray",
        shade=False,
        blending="composite",
        name="raw_volume",
    )


def _add_mask_surface_to_plotter(
    plotter,
    source: VolumeSource,
    *,
    opacity: float,
) -> None:
    binary = np.asarray(source.volume_zyx > 0, dtype=np.uint8)
    if not binary.any():
        return
    grid = _build_image_data(binary, source.spacing_xyz)
    surface = grid.contour([0.5], scalars="values")
    plotter.add_mesh(
        surface,
        color="#d94841",
        opacity=opacity,
        smooth_shading=True,
        ambient=0.2,
        diffuse=0.7,
        specular=0.15,
        name="mask_surface",
    )


def render_volume_snapshot(
    *,
    raw_source: VolumeSource,
    mask_source: VolumeSource | None,
    out_path: str,
    camera: str = "iso",
    raw_opacity: float = 0.30,
    mask_opacity: float = 0.65,
) -> None:
    """Headless snapshot renderer for automated smoke tests and exports."""
    if pv is None:
        raise VolumeInspectorUnavailableError(str(_IMPORT_ERROR))
    plotter = pv.Plotter(off_screen=True, window_size=(1600, 1200))
    plotter.set_background("white")
    plotter.add_axes(line_width=2)
    _add_raw_volume_to_plotter(plotter, raw_source, opacity_scale=raw_opacity)
    if mask_source is not None:
        _add_mask_surface_to_plotter(plotter, mask_source, opacity=mask_opacity)

    camera_name = (camera or "iso").lower()
    if camera_name == "xy":
        plotter.view_xy()
    elif camera_name == "xz":
        plotter.view_xz()
    elif camera_name == "yz":
        plotter.view_yz()
    else:
        plotter.view_isometric()
    plotter.reset_camera_clipping_range()
    plotter.screenshot(out_path)
    plotter.close()


def _build_snapshot_name(raw_label: str, mask_label: str) -> str:
    raw = _sanitize_path_token(raw_label or "raw")
    mask = _sanitize_path_token(mask_label or "mask")
    return str(Path.cwd() / f"{raw}__{mask}__3d_inspector.png")


def _sanitize_path_token(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text).strip("_") or "volume"


def _format_spacing(spacing_xyz: tuple[float, float, float]) -> str:
    return "(" + ", ".join(f"{value:.4f}" for value in spacing_xyz) + ")"
