from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
from PyQt5.QtWidgets import QApplication, QMessageBox

from ..utils.volume_utils import (
    VolumeSource,
    build_resampled_three_d_raw_source,
    build_mask_source_with_reference_extent,
    find_matching_confocal_inference,
)

if TYPE_CHECKING:  # pragma: no cover
    from .main_controller import MainController


class VolumeMixin:
    """3-D volume viewing helpers for confocal raw/prediction stacks."""

    def _open_volume_inspector_current(self: "MainController") -> None:
        if self.model.data is None:
            QMessageBox.warning(self, "3D Inspector", "Please load an image stack first.")
            return
        raw_sources = self._current_raw_volume_sources()
        mask_sources = self._current_mask_volume_sources()
        self._launch_volume_inspector(
            title="3D Inspector - Current Stack",
            raw_sources=raw_sources,
            mask_sources=mask_sources,
            default_raw_label=raw_sources[0].label if raw_sources else None,
            default_mask_label=mask_sources[0].label if mask_sources else None,
        )

    def _open_volume_inspector_matching(self: "MainController") -> None:
        if self.model.path is None or self.model.data is None:
            QMessageBox.warning(
                self,
                "3D Inspector",
                "Load a raw stack first so matching inference assets can be discovered.",
            )
            return

        raw_sources = self._current_raw_volume_sources()
        mask_sources = self._current_mask_volume_sources()
        match = find_matching_confocal_inference(self.model.path)

        if match.get("three_d_resampled_raw_path"):
            try:
                raw_sources.append(
                    build_resampled_three_d_raw_source(
                        self.model.path,
                        match["three_d_resampled_raw_path"],
                        match.get("three_d_meta_json_path"),
                    )
                )
            except Exception as exc:
                self.statusBar().showMessage(f"3D aligned raw not available: {exc}")

        if match.get("two_p_five_d_mask_path"):
            try:
                mask_sources.append(
                    build_mask_source_with_reference_extent(
                        match["two_p_five_d_mask_path"],
                        reference_shape_zyx=raw_sources[0].shape_zyx,
                        reference_spacing_xyz=raw_sources[0].spacing_xyz,
                        label="2.5D nnUNet prediction",
                        note="Loaded from sibling inference folder.",
                    )
                )
            except Exception as exc:
                self.statusBar().showMessage(f"2.5D prediction not available: {exc}")

        if match.get("three_d_mask_path"):
            try:
                reference_raw = next(
                    (src for src in raw_sources if src.label == "3D aligned raw"),
                    raw_sources[0],
                )
                mask_sources.append(
                    build_mask_source_with_reference_extent(
                        match["three_d_mask_path"],
                        reference_shape_zyx=reference_raw.shape_zyx,
                        reference_spacing_xyz=reference_raw.spacing_xyz,
                        label="3D nnUNet prediction",
                        note="Loaded from 3D bundle output.",
                    )
                )
            except Exception as exc:
                self.statusBar().showMessage(f"3D prediction not available: {exc}")

        if len(raw_sources) == 1 and not mask_sources:
            QMessageBox.information(
                self,
                "3D Inspector",
                "No matching inference assets were found. Opening the current raw stack only.",
            )

        default_raw = "3D aligned raw" if any(src.label == "3D aligned raw" for src in raw_sources) else raw_sources[0].label
        if any(src.label == "2.5D nnUNet prediction" for src in mask_sources):
            default_mask = "2.5D nnUNet prediction"
        elif mask_sources:
            default_mask = mask_sources[0].label
        else:
            default_mask = None

        self._launch_volume_inspector(
            title=f"3D Inspector - {match.get('stack_id') or 'Matching Inference'}",
            raw_sources=raw_sources,
            mask_sources=mask_sources,
            default_raw_label=default_raw,
            default_mask_label=default_mask,
        )

    def _current_raw_volume_sources(self: "MainController") -> list[VolumeSource]:
        if self.model.data is None:
            return []
        spacing_xyz = self.model.get_pixel_sizes() or (1.0, 1.0, 1.0)
        source = VolumeSource(
            label="Current raw stack",
            volume_zyx=np.asarray(self.model.original_data if self.model.original_data is not None else self.model.data),
            spacing_xyz=spacing_xyz,
            path=self.model.path,
            kind="raw",
            note="Current loaded stack from Myelin_anno_tool.",
        )
        return [source]

    def _current_mask_volume_sources(self: "MainController") -> list[VolumeSource]:
        if self.model.masks is None:
            return []
        spacing_xyz = self.model.get_pixel_sizes() or (1.0, 1.0, 1.0)
        note = self.model.mask_alignment_note or "Current loaded mask stack."
        source = VolumeSource(
            label="Current loaded mask",
            volume_zyx=np.asarray(self.model.masks),
            spacing_xyz=spacing_xyz,
            path=self.model.mask_path,
            kind="mask",
            note=note,
        )
        return [source]

    def _launch_volume_inspector(
        self: "MainController",
        *,
        title: str,
        raw_sources: list[VolumeSource],
        mask_sources: list[VolumeSource],
        default_raw_label: str | None,
        default_mask_label: str | None,
    ) -> None:
        if not raw_sources:
            QMessageBox.warning(self, "3D Inspector", "No raw volume is available to render.")
            return
        app = QApplication.instance()
        platform_name = app.platformName().lower() if app is not None else ""
        if "wayland" in platform_name:
            QMessageBox.warning(
                self,
                "3D Inspector",
                "3D Inspector is unstable under the current Wayland session.\n"
                "Please restart Myelin_anno_tool in XCB mode and try again:\n\n"
                "QT_QPA_PLATFORM=xcb python3 -m zstack_anno",
            )
            self.statusBar().showMessage(
                f"3D Inspector blocked on platform={platform_name or os.environ.get('QT_QPA_PLATFORM', 'unknown')}"
            )
            return
        try:
            from ..views.volume_inspector import (
                VolumeInspectorUnavailableError,
                VolumeInspectorWindow,
            )
        except Exception as exc:
            QMessageBox.warning(self, "3D Inspector", f"Failed to load 3D viewer:\n{exc}")
            return

        try:
            window = VolumeInspectorWindow(
                title=title,
                raw_sources=raw_sources,
                mask_sources=mask_sources,
                default_raw_label=default_raw_label,
                default_mask_label=default_mask_label,
            )
        except VolumeInspectorUnavailableError as exc:
            QMessageBox.warning(self, "3D Inspector", str(exc))
            return
        except Exception as exc:
            QMessageBox.warning(self, "3D Inspector", f"Failed to open 3D viewer:\n{exc}")
            return

        if not hasattr(self, "_volume_windows"):
            self._volume_windows: list[object] = []
        self._volume_windows.append(window)
        try:
            window.destroyed.connect(lambda *_args, w=window: self._forget_volume_window(w))
        except Exception:
            pass
        window.show()

    def _forget_volume_window(self: "MainController", window: object) -> None:
        windows = getattr(self, "_volume_windows", None)
        if not windows:
            return
        try:
            windows.remove(window)
        except ValueError:
            pass

    def _close_volume_windows(self: "MainController") -> None:
        windows = list(getattr(self, "_volume_windows", []))
        for window in windows:
            try:
                window.close()
            except Exception:
                pass
        self._volume_windows = []
