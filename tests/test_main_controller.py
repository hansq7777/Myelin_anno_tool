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
from PyQt5.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt5.QtGui import QKeyEvent, QMouseEvent
from PyQt5.QtWidgets import QApplication

from zstack_anno.controllers.main_controller import MainController
from zstack_anno.utils.volume_utils import VolumeSource
import zstack_anno.controllers.morphology_helper as morphology_helper


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


def test_p_brush_uses_left_add_and_right_erase(controller, app):
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

    controller.eventFilter(
        controller.canvas.viewport(),
        _mouse_event(QEvent.MouseButtonPress, pos, Qt.RightButton, Qt.RightButton),
    )
    controller.eventFilter(
        controller.canvas.viewport(),
        _mouse_event(QEvent.MouseButtonRelease, pos, Qt.RightButton, Qt.NoButton),
    )
    app.processEvents()

    assert controller.model.get_mask(0)[10, 10] == 0


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
