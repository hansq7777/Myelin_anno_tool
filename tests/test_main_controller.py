import os
import sys
import time
from pathlib import Path
import importlib
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

if importlib.util.find_spec("numpy") is None:
    pytest.skip("numpy not available", allow_module_level=True)
if importlib.util.find_spec("PyQt5") is None:
    pytest.skip("PyQt5 not available", allow_module_level=True)

import numpy as np
from PyQt5.QtCore import QEvent, QPoint, QPointF, QRect, Qt
from PyQt5.QtGui import QKeyEvent, QMouseEvent
from PyQt5.QtWidgets import QApplication

from zstack_anno.controllers.main_controller import MainController, _window_geometry_for_available_rect
from zstack_anno.utils.volume_utils import VolumeSource
import zstack_anno.controllers.morphology_helper as morphology_helper
import zstack_anno.controllers.review_helper as review_helper
import zstack_anno.controllers.script_helper as script_helper


@pytest.fixture(scope="module")
def app():
    inst = QApplication.instance()
    if inst is None:
        inst = QApplication([])
    yield inst
    for widget in QApplication.topLevelWidgets():
        widget.close()
        widget.deleteLater()
    inst.processEvents()
    inst.quit()


@pytest.fixture
def controller(app):
    widget = MainController()
    data = np.zeros((1, 32, 32), dtype=np.uint8)
    widget.model.data = data
    widget.model.original_data = data.copy()
    widget.model.ensure_masks()
    widget.model.index = 0
    widget.slider.setRange(0, 0)
    widget.slider.setEnabled(True)
    widget.brush_size = 1
    widget._update_view(reset_view=True)
    widget.canvas.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    widget.canvas.resetTransform()
    try:
        yield widget
    finally:
        widget.model.mask_dirty = False
        widget.close()
        widget.deleteLater()
        app.processEvents()


def _mouse_event(event_type: QEvent.Type, pos: QPoint, button: Qt.MouseButton, buttons: Qt.MouseButtons) -> QMouseEvent:
    return QMouseEvent(event_type, QPointF(pos), button, buttons, Qt.NoModifier)


def _wait_until(app, predicate, timeout_sec: float = 1.5) -> bool:
    deadline = time.perf_counter() + timeout_sec
    while time.perf_counter() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    app.processEvents()
    return predicate()


def _configure_quick_auto_data(controller, *, n_slices: int = 3) -> None:
    data = np.zeros((n_slices, 32, 32), dtype=np.uint8)
    data[:, 10:22, 11:21] = 220
    controller.model.data = data
    controller.model.original_data = data.copy()
    controller.model.replace_masks(np.zeros_like(data, dtype=np.uint8), dirty=True)
    controller.model.index = min(n_slices // 2, n_slices - 1)
    controller.model.masks[controller.model.index, 16, 16] = 1
    controller.model.update_components()
    controller.slider.setRange(0, n_slices - 1)
    controller.slider.setValue(controller.model.index)
    controller.slider.setEnabled(True)
    controller.undo_stack.clear()
    controller.redo_stack.clear()
    controller.history.clear()


def test_p_brush_and_l_eraser_apply_expected_paint_values(controller, app):
    controller.eventFilter(controller, QKeyEvent(QEvent.KeyPress, Qt.Key_P, Qt.NoModifier))

    pos = QPoint(10, 10)
    controller.eventFilter(
        controller.canvas.viewport(),
        _mouse_event(QEvent.MouseButtonPress, pos, Qt.LeftButton, Qt.LeftButton),
    )
    controller.eventFilter(
        controller.canvas.viewport(),
        _mouse_event(QEvent.MouseButtonRelease, pos, Qt.LeftButton, Qt.NoButton),
    )
    app.processEvents()

    assert controller.model.get_mask(0)[10, 10] == 1

    controller.eventFilter(controller, QKeyEvent(QEvent.KeyPress, Qt.Key_L, Qt.NoModifier))
    controller.eventFilter(
        controller.canvas.viewport(),
        _mouse_event(QEvent.MouseButtonPress, pos, Qt.LeftButton, Qt.LeftButton),
    )
    controller.eventFilter(
        controller.canvas.viewport(),
        _mouse_event(QEvent.MouseButtonRelease, pos, Qt.LeftButton, Qt.NoButton),
    )
    app.processEvents()

    assert controller.model.get_mask(0)[10, 10] == 0


def test_window_geometry_for_available_rect_stays_within_screen():
    rect = _window_geometry_for_available_rect(QRect(0, 0, 1366, 768))

    assert rect.width() <= 1366
    assert rect.height() <= 768
    assert rect.left() >= 0
    assert rect.top() >= 0
    assert rect.right() <= 1365
    assert rect.bottom() <= 767


def test_main_controller_defaults_to_left_aligned_canvas():
    widget = MainController()
    try:
        assert widget.canvas.alignment() & Qt.AlignLeft
    finally:
        widget.close()
        widget.deleteLater()


def test_middle_drag_temporarily_pans_canvas_while_brush_active(controller, app):
    controller.resize(900, 700)
    controller.show()
    app.processEvents()
    controller.model.data = np.zeros((1, 256, 256), dtype=np.uint8)
    controller.model.original_data = controller.model.data.copy()
    controller.model.ensure_masks()
    controller._update_view(reset_view=True)
    controller.canvas.apply_zoom_factor(4.0, emit_signal=False)
    controller.eventFilter(controller, QKeyEvent(QEvent.KeyPress, Qt.Key_P, Qt.NoModifier))

    before_center, _before_visible = controller.canvas.normalized_view_window()
    before_h = controller.canvas.horizontalScrollBar().value()
    before_v = controller.canvas.verticalScrollBar().value()
    QApplication.sendEvent(
        controller.canvas.viewport(),
        _mouse_event(QEvent.MouseButtonPress, QPoint(140, 140), Qt.MiddleButton, Qt.MiddleButton),
    )
    QApplication.sendEvent(
        controller.canvas.viewport(),
        _mouse_event(QEvent.MouseMove, QPoint(220, 190), Qt.NoButton, Qt.MiddleButton),
    )
    QApplication.sendEvent(
        controller.canvas.viewport(),
        _mouse_event(QEvent.MouseButtonRelease, QPoint(220, 190), Qt.MiddleButton, Qt.NoButton),
    )
    app.processEvents()

    after_center, _after_visible = controller.canvas.normalized_view_window()
    after_h = controller.canvas.horizontalScrollBar().value()
    after_v = controller.canvas.verticalScrollBar().value()
    assert controller._paint_tool == controller._PAINT_TOOL_BRUSH
    assert controller.brush_enabled is True
    assert controller.model.get_mask(0).sum() == 0
    assert after_h != before_h or after_v != before_v
    assert after_center[0] != pytest.approx(before_center[0], abs=0.02) or after_center[1] != pytest.approx(before_center[1], abs=0.02)


def test_p_l_and_h_toggle_expected_tool_modes(controller):
    assert controller._paint_tool == controller._PAINT_TOOL_HAND
    assert controller.brush_enabled is False

    controller.eventFilter(controller, QKeyEvent(QEvent.KeyPress, Qt.Key_P, Qt.NoModifier))
    assert controller._paint_tool == controller._PAINT_TOOL_BRUSH
    assert controller.brush_enabled is True

    controller.eventFilter(controller, QKeyEvent(QEvent.KeyPress, Qt.Key_P, Qt.NoModifier))
    assert controller._paint_tool == controller._PAINT_TOOL_HAND
    assert controller.brush_enabled is False

    controller.eventFilter(controller, QKeyEvent(QEvent.KeyPress, Qt.Key_L, Qt.NoModifier))
    assert controller._paint_tool == controller._PAINT_TOOL_ERASER
    assert controller.brush_enabled is True

    controller.eventFilter(controller, QKeyEvent(QEvent.KeyPress, Qt.Key_P, Qt.NoModifier))
    assert controller._paint_tool == controller._PAINT_TOOL_BRUSH
    assert controller.brush_enabled is True

    controller.eventFilter(controller, QKeyEvent(QEvent.KeyPress, Qt.Key_L, Qt.NoModifier))
    assert controller._paint_tool == controller._PAINT_TOOL_ERASER
    assert controller.brush_enabled is True

    controller.eventFilter(controller, QKeyEvent(QEvent.KeyPress, Qt.Key_L, Qt.NoModifier))
    assert controller._paint_tool == controller._PAINT_TOOL_HAND
    assert controller.brush_enabled is False

    controller.eventFilter(controller, QKeyEvent(QEvent.KeyPress, Qt.Key_H, Qt.NoModifier))
    assert controller._paint_tool == controller._PAINT_TOOL_HAND
    assert controller.brush_enabled is False


def test_hold_d_drag_deletes_component_with_preview(controller, app):
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[2:6, 2:6] = 1
    mask[20:24, 20:24] = 1
    controller.model.set_mask(mask, slice_idx=0)
    controller._update_view()

    controller.eventFilter(controller, QKeyEvent(QEvent.KeyPress, Qt.Key_D, Qt.NoModifier))
    assert controller._delete_hold is True

    start = QPoint(2, 2)
    end = QPoint(5, 5)
    controller.eventFilter(
        controller.canvas.viewport(),
        _mouse_event(QEvent.MouseButtonPress, start, Qt.LeftButton, Qt.LeftButton),
    )
    controller.eventFilter(
        controller.canvas.viewport(),
        _mouse_event(QEvent.MouseMove, end, Qt.NoButton, Qt.LeftButton),
    )
    assert not controller._delete_band.isHidden()
    controller.eventFilter(
        controller.canvas.viewport(),
        _mouse_event(QEvent.MouseButtonRelease, end, Qt.LeftButton, Qt.NoButton),
    )
    controller.eventFilter(controller, QKeyEvent(QEvent.KeyRelease, Qt.Key_D, Qt.NoModifier))
    app.processEvents()

    result = controller.model.get_mask(0)
    assert result[2:6, 2:6].sum() == 0
    assert result[20:24, 20:24].sum() == 16
    assert controller._delete_hold is False
    assert not controller._delete_band.isVisible()


def test_paint_undo_and_redo_restore_mask(controller):
    controller.model.replace_masks(np.zeros((1, 32, 32), dtype=np.uint8), dirty=True)

    controller._start_paint(QPoint(12, 12), 1)
    controller._end_paint()
    assert controller.model.get_mask(0)[12, 12] == 1

    controller._undo()
    assert controller.model.get_mask(0)[12, 12] == 0

    controller._redo()
    assert controller.model.get_mask(0)[12, 12] == 1


def test_open_volume_inspector_current_collects_raw_and_mask_sources(controller):
    calls = {}

    def _capture(**kwargs):
        calls.update(kwargs)

    controller._launch_volume_inspector = _capture
    controller.model.path = "/tmp/current_stack.ome.tif"
    controller.model.mask_path = "/tmp/current_mask.ome.tif"
    controller.model.mask_alignment_note = "Mask depth aligned 2->1."

    controller._open_volume_inspector_current()

    assert calls["title"] == "3D Inspector - Current Stack"
    assert len(calls["raw_sources"]) == 1
    assert len(calls["mask_sources"]) == 1
    assert calls["raw_sources"][0].label == "Current raw stack"
    assert calls["mask_sources"][0].label == "Current loaded mask"


def test_open_volume_inspector_matching_discovers_sibling_sources(controller, monkeypatch):
    calls = {}

    def _capture(**kwargs):
        calls.update(kwargs)

    controller._launch_volume_inspector = _capture
    controller.model.path = "/tmp/2502_60_L_M1_S00.ome.tif"
    controller.model.mask_path = "/tmp/current_mask.ome.tif"
    controller.model.ome_metadata = (
        "<?xml version='1.0'?><OME xmlns='http://www.openmicroscopy.org/Schemas/OME/2016-06'>"
        "<Image><Pixels PhysicalSizeX='0.455' PhysicalSizeY='0.455' PhysicalSizeZ='0.2015' /></Image></OME>"
    )

    two_p_mask = VolumeSource(
        label="2.5D nnUNet prediction",
        volume_zyx=np.ones((2, 32, 32), dtype=np.uint8),
        spacing_xyz=(0.455, 0.455, 0.2015),
        path="/tmp/2502_60_L_M1_S00.pred.ome.tif",
        kind="mask",
    )
    three_d_raw = VolumeSource(
        label="3D aligned raw",
        volume_zyx=np.ones((2, 32, 32), dtype=np.uint8),
        spacing_xyz=(0.455, 0.455, 0.396),
        path="/tmp/2502_60_L_M1_S00.ome_dz0p396.tif",
        kind="raw",
    )

    monkeypatch.setattr(
        "zstack_anno.controllers.volume_helper.find_matching_confocal_inference",
        lambda _path: {
            "stack_id": "2502_60_L_M1_S00",
            "two_p_five_d_mask_path": "/tmp/2502_60_L_M1_S00.pred.ome.tif",
            "three_d_resampled_raw_path": "/tmp/2502_60_L_M1_S00.ome_dz0p396.tif",
            "three_d_meta_json_path": "/tmp/2502_60_L_M1_S00.ome.json",
            "three_d_mask_path": None,
        },
    )
    monkeypatch.setattr(
        "zstack_anno.controllers.volume_helper.build_mask_source_with_reference_extent",
        lambda path, **kwargs: two_p_mask if "pred" in path else None,
    )
    monkeypatch.setattr(
        "zstack_anno.controllers.volume_helper.build_resampled_three_d_raw_source",
        lambda *_args, **_kwargs: three_d_raw,
    )

    controller._open_volume_inspector_matching()

    assert calls["title"] == "3D Inspector - 2502_60_L_M1_S00"
    assert [src.label for src in calls["raw_sources"]] == ["Current raw stack", "3D aligned raw"]
    assert [src.label for src in calls["mask_sources"]] == ["Current loaded mask", "2.5D nnUNet prediction"]
    assert calls["default_raw_label"] == "3D aligned raw"
    assert calls["default_mask_label"] == "2.5D nnUNet prediction"


def test_filter_small_uses_3d_component_voxel_count(controller, app):
    controller.model.data = np.zeros((3, 32, 32), dtype=np.uint8)
    controller.model.original_data = controller.model.data.copy()
    controller.model.replace_masks(np.zeros((3, 32, 32), dtype=np.uint8), dirty=True)
    controller.model.masks[0, 3, 3] = 1
    controller.model.masks[1, 3, 3] = 1
    controller.model.masks[2, 3, 3] = 1
    controller.model.masks[0, 20, 20] = 1
    controller.model.masks[1, 20, 20] = 1
    controller.model.update_components()
    controller.filter_spin.setValue(3)

    controller._filter_small()
    assert controller.filter_thread is not None
    assert _wait_until(app, lambda: controller.filter_thread is None)

    assert controller.model.masks[:, 3, 3].tolist() == [1, 1, 1]
    assert controller.model.masks[:, 20, 20].tolist() == [0, 0, 0]
    message = controller.statusBar().currentMessage()
    assert "1/2 components" in message
    assert "50.0%" in message
    assert "2/5 voxels" in message


def test_filter_small_runs_in_background(controller, app, monkeypatch):
    controller.model.data = np.zeros((3, 32, 32), dtype=np.uint8)
    controller.model.original_data = controller.model.data.copy()
    controller.model.replace_masks(np.zeros((3, 32, 32), dtype=np.uint8), dirty=True)
    controller.model.masks[:, 10, 10] = 1
    controller.model.update_components()
    controller.filter_spin.setValue(2)

    original_fn = morphology_helper.morphology_tools.remove_small_components_with_stats

    def delayed_filter(*args, **kwargs):
        time.sleep(0.06)
        return original_fn(*args, **kwargs)

    monkeypatch.setattr(
        morphology_helper.morphology_tools,
        "remove_small_components_with_stats",
        delayed_filter,
    )

    start = time.perf_counter()
    controller._filter_small()
    elapsed = time.perf_counter() - start

    assert elapsed < 0.05
    assert controller.filter_thread is not None
    assert controller.filter_thread.isRunning()
    assert "background" in controller.statusBar().currentMessage().lower()
    assert _wait_until(app, lambda: controller.filter_thread is None)


def test_filter_small_discards_stale_background_result(controller, app, monkeypatch):
    controller.model.data = np.zeros((3, 32, 32), dtype=np.uint8)
    controller.model.original_data = controller.model.data.copy()
    controller.model.replace_masks(np.zeros((3, 32, 32), dtype=np.uint8), dirty=True)
    controller.model.masks[:, 8, 8] = 1
    controller.model.masks[0, 20, 20] = 1
    controller.model.update_components()
    controller.filter_spin.setValue(3)

    original_fn = morphology_helper.morphology_tools.remove_small_components_with_stats

    def delayed_filter(*args, **kwargs):
        time.sleep(0.06)
        return original_fn(*args, **kwargs)

    monkeypatch.setattr(
        morphology_helper.morphology_tools,
        "remove_small_components_with_stats",
        delayed_filter,
    )

    controller._filter_small()
    assert controller.filter_thread is not None
    controller.model.masks[2, 25, 25] = 1
    controller.model.update_components()
    controller.model._touch_mask_revision()

    assert _wait_until(app, lambda: controller.filter_thread is None)
    assert controller.model.masks[2, 25, 25] == 1
    assert "discarded" in controller.statusBar().currentMessage().lower()


def test_quick_auto_script_updates_once_and_pushes_single_undo(controller, monkeypatch):
    data = np.zeros((1, 32, 32), dtype=np.uint8)
    data[0, 10:22, 11:21] = 220
    controller.model.data = data
    controller.model.original_data = data.copy()
    controller.model.ensure_masks()
    controller.model.masks[:] = 0
    controller.model.masks[0, 16, 16] = 1
    controller.model.index = 0
    controller.undo_stack.clear()
    controller.history.clear()

    calls = {"count": 0}
    original_update_view = controller._update_view

    def wrapped_update_view(*args, **kwargs):
        calls["count"] += 1
        return original_update_view(*args, **kwargs)

    monkeypatch.setattr(controller, "_update_view", wrapped_update_view)

    metrics = controller.script_quick_seed_dilate_bg_int_bg(
        seed_percentile=90.0,
        seed_pixel_percent=1.0,
        dilate_iterations=2,
        bg1_percentile=0.0,
        bg1_bins=0,
        grow_diff_pct=200.0,
        grow_hist_pct=-1.0,
        grow_force_pct=-1.0,
        grow_limit=2000,
        bg2_percentile=0.0,
        bg2_bins=0,
        final_bg_repeat=0,
        final_small_threshold=0,
        addition_support_percentile=-1.0,
        protect_small_original=False,
        show_status=False,
        push_undo=True,
    )

    assert metrics is not None
    assert calls["count"] == 1
    assert len(controller.undo_stack) == 1
    assert controller.history == ["quick_auto"]


def test_quick_auto_script_runs_in_background(controller, app, monkeypatch):
    _configure_quick_auto_data(controller, n_slices=3)
    original_fn = script_helper._run_quick_auto_single
    gate_calls = {}

    def delayed_single(*args, **kwargs):
        time.sleep(0.06)
        return original_fn(*args, **kwargs)

    monkeypatch.setattr(script_helper, "_run_quick_auto_single", delayed_single)
    monkeypatch.setattr(
        controller,
        "_post_quick_auto_quality_gate",
        lambda metrics, context_label, elapsed_sec=None: gate_calls.update(
            {
                "metrics": metrics,
                "context_label": context_label,
                "elapsed_sec": elapsed_sec,
            }
        ),
    )

    start = time.perf_counter()
    controller._run_quick_auto_script()
    elapsed = time.perf_counter() - start

    assert elapsed < 0.05
    assert controller.quick_auto_thread is not None
    assert controller.quick_auto_thread.isRunning()
    assert controller.quick_auto_cancel_btn.isEnabled()
    assert "background" in controller.statusBar().currentMessage().lower()
    assert _wait_until(app, lambda: controller.quick_auto_thread is None, timeout_sec=2.0)
    assert gate_calls["context_label"] == f"slice {controller.model.index + 1}"
    assert gate_calls["elapsed_sec"] is not None
    assert len(controller.undo_stack) == 1
    assert controller.history == ["quick_auto"]


def test_quick_auto_script_cancelled_keeps_masks_unchanged(controller, app, monkeypatch):
    _configure_quick_auto_data(controller, n_slices=3)
    before_masks = controller.model.masks.copy()

    def cancellable_single(*args, **kwargs):
        cancel_event = kwargs.get("cancel_event")
        for _ in range(40):
            time.sleep(0.01)
            if cancel_event is not None and cancel_event.is_set():
                raise script_helper.QuickAutoCancelled("cancelled")
        raise AssertionError("Quick auto thread was expected to be cancelled before completion")

    monkeypatch.setattr(script_helper, "_run_quick_auto_single", cancellable_single)

    controller._run_quick_auto_script()
    assert controller.quick_auto_thread is not None
    assert controller.quick_auto_thread.isRunning()

    controller._cancel_quick_auto()

    assert _wait_until(app, lambda: controller.quick_auto_thread is None, timeout_sec=2.0)
    assert np.array_equal(controller.model.masks, before_masks)
    assert len(controller.undo_stack) == 0
    assert controller.history == []
    assert "cancelled" in controller.statusBar().currentMessage().lower()
    assert "no changes were applied" in controller.statusBar().currentMessage().lower()


def test_quick_auto_stack_runs_in_background(controller, app, monkeypatch):
    _configure_quick_auto_data(controller, n_slices=3)
    original_fn = script_helper._run_quick_auto_stack
    gate_calls = {}

    def delayed_stack(*args, **kwargs):
        time.sleep(0.06)
        return original_fn(*args, **kwargs)

    monkeypatch.setattr(script_helper, "_run_quick_auto_stack", delayed_stack)
    monkeypatch.setattr(
        controller,
        "_post_quick_auto_stack_quality_gate",
        lambda metrics_by_slice, indices, elapsed_sec=None: gate_calls.update(
            {
                "metrics_by_slice": metrics_by_slice,
                "indices": list(indices),
                "elapsed_sec": elapsed_sec,
            }
        ),
    )

    start = time.perf_counter()
    started = controller._start_quick_auto_job(
        mode="stack",
        indices=[0, 1, 2],
        label="stack quick auto",
        params=controller._quick_auto_params_for_selected_preset(),
    )
    elapsed = time.perf_counter() - start

    assert started is True
    assert elapsed < 0.05
    assert controller.quick_auto_thread is not None
    assert controller.quick_auto_thread.isRunning()
    assert controller.quick_auto_cancel_btn.isEnabled()
    assert _wait_until(app, lambda: controller.quick_auto_thread is None, timeout_sec=2.0)
    assert gate_calls["indices"] == [0, 1, 2]
    assert len(gate_calls["metrics_by_slice"]) == 3
    assert gate_calls["elapsed_sec"] is not None


def test_quick_auto_stack_cancelled_discards_partial_result(controller, app, monkeypatch):
    _configure_quick_auto_data(controller, n_slices=4)
    before_masks = controller.model.masks.copy()

    def cancellable_stack(data_stack, mask_stack, indices, *, cancel_event=None, progress_fn=None, **params):
        work_masks = np.asarray(mask_stack, dtype=np.uint8).copy()
        metrics = []
        ordered = [int(idx) for idx in indices]
        for processed, slice_index in enumerate(ordered, start=1):
            time.sleep(0.03)
            metric = {
                "slice_index": slice_index,
                "before_pixels": int(np.count_nonzero(work_masks[slice_index])),
                "after_pixels": int(np.count_nonzero(work_masks[slice_index])) + 4,
                "changed": True,
            }
            if progress_fn is not None:
                progress_fn(processed, len(ordered), slice_index, metric)
            if cancel_event is not None and cancel_event.is_set():
                raise script_helper.QuickAutoCancelled("cancelled")
            work_masks[slice_index, 5:7, 5:7] = 1
            metrics.append(metric)
        return {
            "masks": work_masks,
            "indices": ordered,
            "metrics_by_slice": metrics,
            "changed": True,
        }

    monkeypatch.setattr(script_helper, "_run_quick_auto_stack", cancellable_stack)

    started = controller._start_quick_auto_job(
        mode="stack",
        indices=[0, 1, 2, 3],
        label="stack quick auto",
        params=controller._quick_auto_params_for_selected_preset(),
    )
    assert started is True
    assert controller.quick_auto_thread is not None
    assert controller.quick_auto_thread.isRunning()
    assert _wait_until(
        app,
        lambda: "1/4" in controller.statusBar().currentMessage()
        or "2/4" in controller.statusBar().currentMessage(),
        timeout_sec=1.0,
    )

    controller._cancel_quick_auto()

    assert _wait_until(app, lambda: controller.quick_auto_thread is None, timeout_sec=2.0)
    assert np.array_equal(controller.model.masks, before_masks)
    message = controller.statusBar().currentMessage().lower()
    assert "cancelled" in message
    assert "no changes were applied" in message


def test_open_review_tracker_always_prompts_for_confirmation(controller, monkeypatch):
    chosen = {}
    monkeypatch.setattr(
        review_helper,
        "windows_to_local_path",
        lambda _path: "/tmp/default_tracker.xlsx",
    )

    def fake_open_file_name(parent, title, start_path, filter_text):
        chosen["title"] = title
        chosen["start_path"] = start_path
        chosen["filter"] = filter_text
        return ("/tmp/confirmed_tracker.xlsx", "Excel Files (*.xlsx)")

    monkeypatch.setattr(review_helper.QFileDialog, "getOpenFileName", fake_open_file_name)
    monkeypatch.setattr(
        controller,
        "_load_review_tracker",
        lambda path: chosen.setdefault("loaded_path", path),
    )

    controller._open_review_tracker()

    assert chosen["title"] == "Open Review Tracker"
    assert chosen["start_path"] == "/tmp/default_tracker.xlsx"
    assert chosen["loaded_path"] == "/tmp/confirmed_tracker.xlsx"


def test_review_maybe_reshuffle_loaded_tracker_prompts_and_saves(controller, monkeypatch):
    controller._review_table_rows = [{"zstack_id": "stack_a", "review_grade": "", "review_queue_rank": 1}]
    calls = {}
    monkeypatch.setattr(review_helper.QMessageBox, "question", lambda *args, **kwargs: review_helper.QMessageBox.Yes)
    monkeypatch.setattr(
        controller,
        "_review_reshuffle_loaded_tracker_queue",
        lambda: calls.setdefault("reshuffled", True),
    )
    monkeypatch.setattr(controller, "_review_save_tracker", lambda: calls.setdefault("saved", True))

    result = controller._review_maybe_reshuffle_loaded_tracker()

    assert result is True
    assert calls == {"reshuffled": True, "saved": True}


def test_review_reshuffle_loaded_tracker_queue_keeps_reviewed_items_first(controller):
    controller._review_table_rows = [
        {"zstack_id": "stack_c", "review_grade": "", "review_queue_rank": 3},
        {"zstack_id": "stack_a", "review_grade": "A", "review_queue_rank": 2},
        {"zstack_id": "stack_d", "review_grade": "", "review_queue_rank": 4},
        {"zstack_id": "stack_b", "review_grade": "B", "review_queue_rank": 1},
    ]

    controller._review_reshuffle_loaded_tracker_queue()

    rank_by_id = {
        row["zstack_id"]: int(row["review_queue_rank"])
        for row in controller._review_table_rows
    }
    assert rank_by_id["stack_b"] == 1
    assert rank_by_id["stack_a"] == 2
    assert {rank_by_id["stack_c"], rank_by_id["stack_d"]} == {3, 4}


def test_review_default_tracker_filename_uses_raw_folder_and_date(controller):
    filename = controller._review_default_tracker_filename(
        "/tmp/2026-02-11_run/original_zstacks",
        when=review_helper.datetime(2026, 4, 2),
    )

    assert filename == "2026-02-11_run_original_zstacks_review_tracker_2026-04-02.xlsx"


def test_build_tracker_uses_generated_default_filename(controller, monkeypatch):
    calls = {"existing_dirs": []}

    def fake_existing_directory(_parent, title):
        calls["existing_dirs"].append(title)
        if title == "Select Raw Stack Folder":
            return "/tmp/2026-02-11_run/original_zstacks"
        if title == "Select Prediction Mask Folder":
            return "/tmp/2026-02-11_run/predictions"
        return ""

    def fake_save_file_name(parent, title, start_path, filter_text):
        calls["save_title"] = title
        calls["save_start_path"] = start_path
        calls["save_filter"] = filter_text
        return ("/tmp/confirmed_tracker.xlsx", "Excel Files (*.xlsx)")

    monkeypatch.setattr(review_helper.QFileDialog, "getExistingDirectory", fake_existing_directory)
    monkeypatch.setattr(review_helper.QFileDialog, "getSaveFileName", fake_save_file_name)
    monkeypatch.setattr(controller, "_review_choose_import_tracker_path", lambda raw_dir, tracker_path: "")
    monkeypatch.setattr(
        controller,
        "_review_build_tracker_rows",
        lambda raw_dir, pred_dir, tracker_path, import_tracker_path="": (
            [],
            {
                "matched": 0,
                "raw_only": 0,
                "pred_only": 0,
                "raw_duplicates": 0,
                "pred_duplicates": 0,
            },
        ),
    )
    monkeypatch.setattr(controller, "_review_write_tracker", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        controller,
        "_load_review_tracker",
        lambda path: calls.setdefault("loaded_path", path),
    )
    monkeypatch.setattr(review_helper.QMessageBox, "information", lambda *args, **kwargs: None)

    controller._review_build_tracker_from_folders()

    assert calls["save_title"] == "Create or Refresh Review Tracker"
    assert calls["save_start_path"] == controller._review_default_tracker_path(
        "/tmp/2026-02-11_run/original_zstacks"
    )
    assert calls["loaded_path"] == "/tmp/confirmed_tracker.xlsx"


def test_build_tracker_passes_selected_import_tracker(controller, monkeypatch):
    captured = {}

    def fake_existing_directory(_parent, title):
        if title == "Select Raw Stack Folder":
            return "/tmp/run/original_zstacks"
        if title == "Select Prediction Mask Folder":
            return "/tmp/run/predictions"
        return ""

    def fake_save_file_name(_parent, _title, _start_path, _filter_text):
        return ("/tmp/new_tracker.xlsx", "Excel Files (*.xlsx)")

    monkeypatch.setattr(review_helper.QFileDialog, "getExistingDirectory", fake_existing_directory)
    monkeypatch.setattr(review_helper.QFileDialog, "getSaveFileName", fake_save_file_name)
    monkeypatch.setattr(
        controller,
        "_review_choose_import_tracker_path",
        lambda raw_dir, tracker_path: "/tmp/old_tracker.xlsx",
    )

    def fake_build_rows(raw_dir, pred_dir, tracker_path, import_tracker_path=""):
        captured["raw_dir"] = raw_dir
        captured["pred_dir"] = pred_dir
        captured["tracker_path"] = tracker_path
        captured["import_tracker_path"] = import_tracker_path
        return (
            [],
            {
                "matched": 0,
                "raw_only": 0,
                "pred_only": 0,
                "raw_duplicates": 0,
                "pred_duplicates": 0,
            },
        )

    monkeypatch.setattr(controller, "_review_build_tracker_rows", fake_build_rows)
    monkeypatch.setattr(controller, "_review_write_tracker", lambda *args, **kwargs: None)
    monkeypatch.setattr(controller, "_load_review_tracker", lambda _path: None)
    monkeypatch.setattr(review_helper.QMessageBox, "information", lambda *args, **kwargs: None)

    controller._review_build_tracker_from_folders()

    assert captured["raw_dir"] == "/tmp/run/original_zstacks"
    assert captured["pred_dir"] == "/tmp/run/predictions"
    assert captured["tracker_path"] == "/tmp/new_tracker.xlsx"
    assert captured["import_tracker_path"] == "/tmp/old_tracker.xlsx"


def test_review_build_tracker_assigns_fixed_weighted_queue(controller, monkeypatch):
    controller._review_rng.seed(7)
    monkeypatch.setattr(
        controller,
        "_review_collect_stack_files",
        lambda root_dir, allow_czi: (
            ["/tmp/raw_a.ome.tif", "/tmp/raw_b.ome.tif", "/tmp/raw_c.ome.tif"]
            if allow_czi
            else ["/tmp/raw_a.pred.ome.tif", "/tmp/raw_b.pred.ome.tif", "/tmp/raw_c.pred.ome.tif"]
        ),
    )
    monkeypatch.setattr(controller, "_review_load_existing_rows_by_id", lambda _path: {})
    monkeypatch.setattr(controller, "_review_guess_source_group", lambda *_args: "group")
    voxel_counts = {
        "/tmp/raw_a.pred.ome.tif": 18,
        "/tmp/raw_b.pred.ome.tif": 1800,
        "/tmp/raw_c.pred.ome.tif": 180000,
    }
    monkeypatch.setattr(
        controller,
        "_review_prediction_foreground_voxels",
        lambda path: voxel_counts[path],
    )

    rows, stats = controller._review_build_tracker_rows("/tmp/raw", "/tmp/pred", "/tmp/tracker.xlsx")

    assert stats["matched"] == 3
    assert [int(row["review_queue_rank"]) for row in rows] == [1, 2, 3]
    assert rows[0]["zstack_id"] == "raw_a"
    assert rows[0]["inference_foreground_voxels"] == 18
    assert float(rows[0]["review_priority_weight"]) > float(rows[-1]["review_priority_weight"])


def test_review_build_tracker_imports_review_fields_but_regenerates_queue(controller, monkeypatch):
    controller._review_rng.seed(11)
    monkeypatch.setattr(
        controller,
        "_review_collect_stack_files",
        lambda root_dir, allow_czi: ["/tmp/raw_a.ome.tif"] if allow_czi else ["/tmp/raw_a.pred.ome.tif"],
    )
    monkeypatch.setattr(
        controller,
        "_review_load_existing_rows_by_id",
        lambda path: {
            "raw_a": {
                "review_grade": "B",
                "review_completed": 1,
                "review_completed_at": "2026-01-01 08:00:00",
                "review_queue_rank": 99,
                "review_priority_weight": "0.000001",
            }
        },
    )
    monkeypatch.setattr(controller, "_review_guess_source_group", lambda *_args: "group")
    monkeypatch.setattr(controller, "_review_prediction_foreground_voxels", lambda _path: 128)

    rows, _stats = controller._review_build_tracker_rows(
        "/tmp/raw",
        "/tmp/pred",
        "/tmp/tracker.xlsx",
        import_tracker_path="/tmp/old_tracker.xlsx",
    )

    assert len(rows) == 1
    assert rows[0]["review_grade"] == "B"
    assert rows[0]["review_completed"] == 1
    assert rows[0]["review_completed_at"] == "2026-01-01 08:00:00"
    assert int(rows[0]["review_queue_rank"]) == 1
    assert rows[0]["review_priority_weight"] != "0.000001"


def test_review_refresh_items_uses_queue_rank_order(controller, monkeypatch):
    rows = [
        {
            "zstack_id": "stack_b",
            "raw_path": "/tmp/raw_b.ome.tif",
            "inference_path": "/tmp/pred_b.ome.tif",
            "review_queue_rank": 2,
            "review_grade": "",
            "review_completed": "",
            "review_completed_at": "",
            "review_corrected_mask_path": "",
            "source_group": "",
        },
        {
            "zstack_id": "stack_a",
            "raw_path": "/tmp/raw_a.ome.tif",
            "inference_path": "/tmp/pred_a.ome.tif",
            "review_queue_rank": 1,
            "review_grade": "A",
            "review_completed": 1,
            "review_completed_at": "2026-01-01 10:00:00",
            "review_corrected_mask_path": "",
            "source_group": "",
        },
    ]
    controller._review_header_names = list(rows[0].keys())
    controller._review_table_rows = rows
    controller._review_rebuild_header_index()
    monkeypatch.setattr(review_helper, "windows_to_local_path", lambda path: path)
    monkeypatch.setattr(review_helper.os.path, "exists", lambda _path: True)

    controller._review_refresh_items()

    assert [item["zstack_id"] for item in controller._review_items] == ["stack_a", "stack_b"]
    assert controller._review_items[0]["completed"] is True


def test_review_next_unfinished_stack_includes_incomplete_b_grade_items(controller, monkeypatch):
    controller._review_items = [
        {"zstack_id": "stack_a", "grade": "A", "completed": True, "raw_path": "/tmp/a.ome.tif"},
        {"zstack_id": "stack_b", "grade": "", "completed": False, "raw_path": "/tmp/b.ome.tif"},
        {"zstack_id": "stack_c", "grade": "B", "completed": False, "raw_path": "/tmp/c.ome.tif"},
        {"zstack_id": "stack_d", "grade": "", "completed": False, "raw_path": "/tmp/d.ome.tif"},
    ]
    controller._review_current_item_index = 1
    controller._review_filtered_indices = [0, 1, 2, 3]
    controller.review_filter_combo.setCurrentText("All")
    loaded = {}
    monkeypatch.setattr(controller, "_review_set_filter", lambda value: loaded.setdefault("filter", value))
    monkeypatch.setattr(
        controller,
        "_review_rebuild_filtered_indices",
        lambda load_item=False: controller._review_filtered_indices.__setitem__(slice(None), [0, 1, 2, 3]),
    )
    monkeypatch.setattr(controller, "_review_load_filtered_pos", lambda pos, force=False: loaded.setdefault("pos", pos))

    controller._review_next_unfinished_stack()

    assert loaded["filter"] == "All"
    assert loaded["pos"] == 2


def test_review_update_info_label_updates_canvas_and_preview_badges(controller):
    controller.model.path = "/tmp/review_stack.ome.tif"
    controller._review_items = [
        {
            "zstack_id": "review_stack",
            "grade": "B",
            "completed": True,
            "raw_path": "/tmp/review_stack.ome.tif",
        }
    ]
    controller._review_filtered_indices = [0]
    controller._review_current_item_index = 0
    controller._review_current_filtered_pos = 0

    controller._review_update_info_label()

    assert controller.canvas._review_badge_text == "B"
    assert controller.canvas._review_badge_done is True
    assert controller.inline_volume_preview._review_badge_text == "B"
