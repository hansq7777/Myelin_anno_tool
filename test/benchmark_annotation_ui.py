from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

if sys.platform.startswith("win"):
    os.environ.setdefault("QT_QPA_PLATFORM", "minimal")
else:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import tifffile
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtWidgets import QApplication

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zstack_anno.controllers.main_controller import MainController
from zstack_anno.utils import morphology_tools
from zstack_anno.views.inline_volume_preview import (
    _build_preview_image,
    _prepare_preview_data,
    _render_prepared_preview_image,
)


def _sample_paths() -> tuple[Path, Path]:
    candidates = [
        (
            Path(
                "/mnt/d/Research/Image Analysis/Confocal Myelin data/Inference/"
                "2026-02-11_232719_3d_vanilla_5fold_665_detached_3d_vanilla_dataset007_665_chunked/"
                "original_zstacks/2504_42_L_M1_S08.ome_dz0p396.tif"
            ),
            Path(
                "/mnt/d/Research/Image Analysis/Confocal Myelin data/Inference/"
                "2026-02-11_232719_3d_vanilla_5fold_665_detached_3d_vanilla_dataset007_665_chunked/"
                "predictions/all_592stacks/2504_42_L_M1_S08.ome_pred.tif"
            ),
        ),
        (
            Path(
                "D:/Research/Image Analysis/Confocal Myelin data/Inference/"
                "2026-02-11_232719_3d_vanilla_5fold_665_detached_3d_vanilla_dataset007_665_chunked/"
                "original_zstacks/2504_42_L_M1_S08.ome_dz0p396.tif"
            ),
            Path(
                "D:/Research/Image Analysis/Confocal Myelin data/Inference/"
                "2026-02-11_232719_3d_vanilla_5fold_665_detached_3d_vanilla_dataset007_665_chunked/"
                "predictions/all_592stacks/2504_42_L_M1_S08.ome_pred.tif"
            ),
        ),
    ]
    for raw_path, pred_path in candidates:
        if raw_path.exists() and pred_path.exists():
            return raw_path, pred_path
    raise FileNotFoundError("Sample raw/prediction pair not found in known locations.")


def _load_sample_pair() -> tuple[np.ndarray, np.ndarray, tuple[float, float, float], dict[str, str]]:
    raw_path, pred_path = _sample_paths()
    raw = np.asarray(tifffile.imread(raw_path))
    pred = (np.asarray(tifffile.imread(pred_path)) > 0).astype(np.uint8)
    raw = np.squeeze(raw)
    pred = np.squeeze(pred)
    if raw.ndim != 3 or pred.ndim != 3:
        raise ValueError(f"Expected 3-D sample pair, got raw={raw.shape}, pred={pred.shape}")
    spacing_xyz = (0.455, 0.455, 0.396)
    return raw, pred, spacing_xyz, {"raw": str(raw_path), "pred": str(pred_path)}


def _wait_until(app: QApplication, predicate, timeout_sec: float = 5.0) -> bool:
    deadline = time.perf_counter() + timeout_sec
    while time.perf_counter() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    app.processEvents()
    return predicate()


def _make_controller(app: QApplication, raw: np.ndarray, masks: np.ndarray) -> MainController:
    controller = MainController()
    controller.model.data = raw.copy()
    controller.model.original_data = raw.copy()
    controller.model.replace_masks(masks.copy(), dirty=True)
    controller.model.index = min(raw.shape[0] // 2, raw.shape[0] - 1)
    controller.slider.setRange(0, raw.shape[0] - 1)
    controller.slider.setEnabled(True)
    controller.canvas.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    controller.canvas.resetTransform()
    controller._update_view(reset_view=True)
    app.processEvents()
    return controller


def _benchmark_paint_undo_delete(app: QApplication, raw: np.ndarray) -> dict[str, float | int]:
    masks = np.zeros(raw.shape, dtype=np.uint8)
    controller = _make_controller(app, raw, masks)
    try:
        z = controller.model.index
        t0 = time.perf_counter()
        controller._start_paint(QPoint(220, 220), 1)
        controller._continue_paint(QPoint(260, 250))
        controller._end_paint()
        paint_sec = time.perf_counter() - t0
        painted_pixels = int(controller.model.get_mask(z).sum())

        t1 = time.perf_counter()
        controller._undo()
        undo_sec = time.perf_counter() - t1
        undo_pixels = int(controller.model.get_mask(z).sum())

        t2 = time.perf_counter()
        controller._redo()
        redo_sec = time.perf_counter() - t2
        redo_pixels = int(controller.model.get_mask(z).sum())

        controller.model.masks[:] = 0
        controller.model.masks[z, 180:300, 180:300] = 1
        controller.model.update_components()
        controller._update_view(reset_view=True)
        t3 = time.perf_counter()
        controller._delete_rect(QPoint(200, 200), QPoint(240, 240))
        delete_sec = time.perf_counter() - t3
        delete_pixels = int(controller.model.get_mask(z).sum())
        return {
            "paint_sec": round(paint_sec, 6),
            "paint_pixels": painted_pixels,
            "undo_sec": round(undo_sec, 6),
            "undo_pixels": undo_pixels,
            "redo_sec": round(redo_sec, 6),
            "redo_pixels": redo_pixels,
            "delete_sec": round(delete_sec, 6),
            "delete_pixels": delete_pixels,
        }
    finally:
        controller.model.mask_dirty = False
        controller.close()
        controller.deleteLater()
        app.processEvents()


def _benchmark_rotation_widget(
    app: QApplication,
    raw: np.ndarray,
    pred: np.ndarray,
    spacing_xyz: tuple[float, float, float],
) -> dict[str, float | int]:
    controller = _make_controller(app, raw, pred)
    try:
        preview = controller.inline_volume_preview
        prepare_start = time.perf_counter()
        controller.inline_volume_chk.setChecked(True)
        controller.inline_view_combo.setCurrentText("Oblique")
        app.processEvents()
        controller._refresh_inline_volume_preview()
        ok = _wait_until(app, lambda: preview._render_thread is None and preview._projection is not None)
        if not ok:
            raise RuntimeError("Inline preview did not become ready")
        prepare_total_sec = time.perf_counter() - prepare_start

        t0 = time.perf_counter()
        preview.rotate_by_drag_delta(44, -18, interactive=True)
        interactive_call_sec = time.perf_counter() - t0
        ok = _wait_until(app, lambda: not preview._render_timer.isActive())
        if not ok:
            raise RuntimeError("Interactive rotation render did not finish")
        interactive_total_sec = time.perf_counter() - t0
        yaw_offset, pitch_offset = preview.rotation_offsets()

        t1 = time.perf_counter()
        preview._request_preview_render(interactive=False, delay_ms=0)
        full_refresh_sec = time.perf_counter() - t1

        t2 = time.perf_counter()
        preview.reset_view_rotation()
        reset_sec = time.perf_counter() - t2
        return {
            "prepare_total_sec": round(prepare_total_sec, 6),
            "interactive_call_sec": round(interactive_call_sec, 6),
            "interactive_total_sec": round(interactive_total_sec, 6),
            "full_refresh_sec": round(full_refresh_sec, 6),
            "reset_sec": round(reset_sec, 6),
            "yaw_offset_deg": round(float(yaw_offset), 3),
            "pitch_offset_deg": round(float(pitch_offset), 3),
        }
    finally:
        controller.model.mask_dirty = False
        controller.close()
        controller.deleteLater()
        app.processEvents()


def _run_quick_auto_baseline(controller: MainController, params: dict[str, object]) -> None:
    original_mask = controller.model.get_mask().copy()
    controller.script_seed(
        percentile=float(params["seed_percentile"]),
        pixel_percent=float(params["seed_pixel_percent"]),
    )
    controller.script_dilate(iterations=int(params["dilate_iterations"]))
    controller.script_bg_filter(
        percentile=float(params["bg1_percentile"]),
        bins=int(params["bg1_bins"]),
    )
    controller.script_int_grow(
        diff_pct=float(params["grow_diff_pct"]),
        hist_pct=(None if float(params["grow_hist_pct"]) < 0 else float(params["grow_hist_pct"])),
        force_pct=(None if float(params["grow_force_pct"]) < 0 else float(params["grow_force_pct"])),
        limit=(None if int(params["grow_limit"]) <= 0 else int(params["grow_limit"])),
    )
    controller.script_bg_filter(
        percentile=float(params["bg2_percentile"]),
        bins=int(params["bg2_bins"]),
    )
    for _ in range(max(0, int(params.get("final_bg_repeat", 5)))):
        controller.script_bg_filter(
            percentile=float(params.get("final_bg_percentile", 5.0)),
            bins=int(params.get("final_bg_bins", 5)),
        )
    final_small_threshold = int(params.get("final_small_threshold", 20))
    if final_small_threshold > 0:
        controller.script_filter_small(threshold=final_small_threshold)

    cur_mask = controller.model.get_mask().copy()
    addition_support_percentile = float(params.get("addition_support_percentile", 75.0))
    if addition_support_percentile >= 0:
        img = controller.model._extract_slice(controller.model.index).astype(float)
        support_thresh = float(np.percentile(img, addition_support_percentile))
        support = img >= support_thresh
        additions = (cur_mask > 0) & (original_mask == 0)
        cur_mask[additions & ~support] = 0

    if bool(params.get("protect_small_original", True)) and int(params.get("small_component_guard", 120)) > 0:
        labels = morphology_tools.label_components(original_mask)
        if labels.size > 0:
            counts = np.bincount(labels.ravel())
            if counts.size > 1:
                keep_ids = np.where(
                    (np.arange(counts.size) > 0)
                    & (counts <= int(params.get("small_component_guard", 120)))
                )[0]
                if keep_ids.size > 0:
                    protected = np.isin(labels, keep_ids)
                    cur_mask[protected] = 1

    controller.model.set_mask(cur_mask)
    controller._update_view()


def _benchmark_quick_auto(app: QApplication, raw: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    controller = _make_controller(app, raw, pred)
    params = controller._quick_auto_params_for_selected_preset()
    base_masks = pred.copy()
    slice_index = controller.model.index

    original_update_view = controller._update_view

    def _restore_masks() -> None:
        controller.model.replace_masks(base_masks.copy(), dirty=True)
        controller.model.index = slice_index
        controller.slider.setValue(slice_index)
        original_update_view(reset_view=True)
        controller.undo_stack.clear()
        controller.redo_stack.clear()
        controller.history.clear()
        app.processEvents()

    try:
        baseline_calls = {"count": 0}

        def baseline_update(*args, **kwargs):
            baseline_calls["count"] += 1
            return original_update_view(*args, **kwargs)

        controller._update_view = baseline_update
        _restore_masks()
        t0 = time.perf_counter()
        _run_quick_auto_baseline(controller, params)
        baseline_sec = time.perf_counter() - t0
        baseline_pixels = int(controller.model.get_mask().sum())

        optimized_calls = {"count": 0}

        def optimized_update(*args, **kwargs):
            optimized_calls["count"] += 1
            return original_update_view(*args, **kwargs)

        controller._update_view = optimized_update
        _restore_masks()
        t1 = time.perf_counter()
        controller.script_quick_seed_dilate_bg_int_bg(show_status=False, push_undo=False, **params)
        optimized_sec = time.perf_counter() - t1
        optimized_pixels = int(controller.model.get_mask().sum())
        return {
            "baseline_sec": round(baseline_sec, 6),
            "baseline_update_calls": baseline_calls["count"],
            "baseline_pixels": baseline_pixels,
            "optimized_sec": round(optimized_sec, 6),
            "optimized_update_calls": optimized_calls["count"],
            "optimized_pixels": optimized_pixels,
            "speedup": round(baseline_sec / max(optimized_sec, 1e-9), 3),
        }
    finally:
        controller._update_view = original_update_view
        controller.model.mask_dirty = False
        controller.close()
        controller.deleteLater()
        app.processEvents()


def _benchmark_rotation_pipeline(
    raw: np.ndarray,
    pred: np.ndarray,
    spacing_xyz: tuple[float, float, float],
) -> dict[str, float]:
    angles = [("oblique", 0.0, 0.0), ("oblique", 22.0, -14.0), ("xy", 28.0, -12.0)]
    baseline_runs = []
    for mode, yaw, pitch in angles:
        t0 = time.perf_counter()
        _build_preview_image(
            raw,
            pred,
            spacing_xyz,
            view_mode=mode,
            yaw_offset_deg=yaw,
            pitch_offset_deg=pitch,
            interactive=False,
        )
        baseline_runs.append(time.perf_counter() - t0)

    prepared = _prepare_preview_data(raw, pred, spacing_xyz)
    prepared_runs = []
    interactive_runs = []
    for mode, yaw, pitch in angles:
        t1 = time.perf_counter()
        _render_prepared_preview_image(
            prepared,
            view_mode=mode,
            yaw_offset_deg=yaw,
            pitch_offset_deg=pitch,
            interactive=False,
        )
        prepared_runs.append(time.perf_counter() - t1)
        t2 = time.perf_counter()
        _render_prepared_preview_image(
            prepared,
            view_mode=mode,
            yaw_offset_deg=yaw,
            pitch_offset_deg=pitch,
            interactive=True,
        )
        interactive_runs.append(time.perf_counter() - t2)

    baseline_mean = sum(baseline_runs) / len(baseline_runs)
    prepared_mean = sum(prepared_runs) / len(prepared_runs)
    interactive_mean = sum(interactive_runs) / len(interactive_runs)
    return {
        "baseline_full_rebuild_mean_sec": round(baseline_mean, 6),
        "prepared_full_mean_sec": round(prepared_mean, 6),
        "prepared_interactive_mean_sec": round(interactive_mean, 6),
        "prepared_full_speedup": round(baseline_mean / max(prepared_mean, 1e-9), 3),
        "prepared_interactive_speedup": round(baseline_mean / max(interactive_mean, 1e-9), 3),
    }


def main() -> None:
    raw, pred, spacing_xyz, paths = _load_sample_pair()
    app = QApplication.instance() or QApplication([])
    results = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "qt_platform": os.environ.get("QT_QPA_PLATFORM", ""),
        "sample_paths": paths,
        "shape": tuple(int(v) for v in raw.shape),
        "paint_undo_delete": _benchmark_paint_undo_delete(app, raw),
        "rotation_widget": _benchmark_rotation_widget(app, raw, pred, spacing_xyz),
        "rotation_pipeline": _benchmark_rotation_pipeline(raw, pred, spacing_xyz),
        "quick_auto": _benchmark_quick_auto(app, raw, pred),
    }
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
