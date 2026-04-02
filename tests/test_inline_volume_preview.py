import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from PyQt5.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt5.QtGui import QKeyEvent, QMouseEvent, QWheelEvent
from PyQt5.QtWidgets import QApplication

from zstack_anno.controllers.main_controller import MainController
import zstack_anno.views.inline_volume_preview as inline_volume_preview
from zstack_anno.views.canvas import SliceCanvas
from zstack_anno.views.inline_volume_preview import (
    InlineVolumePreview,
    _MAX_VOXELS,
    _build_preview_image,
    _choose_downsample_steps,
)


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


def _wait_until(app, predicate, timeout_sec: float = 1.5) -> bool:
    deadline = time.perf_counter() + timeout_sec
    while time.perf_counter() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    app.processEvents()
    return predicate()


def _wheel_event_for(widget, delta_y: int = 120) -> QWheelEvent:
    center = widget.rect().center()
    return QWheelEvent(
        QPointF(center),
        QPointF(center),
        QPoint(0, 0),
        QPoint(0, delta_y),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.ScrollUpdate,
        False,
    )


def test_choose_downsample_steps_keeps_preview_volume_bounded():
    steps = _choose_downsample_steps((96, 512, 512))
    sampled = tuple(int(np.ceil(dim / step)) for dim, step in zip((96, 512, 512), steps))
    assert sampled[0] * sampled[1] * sampled[2] <= _MAX_VOXELS
    assert all(step >= 1 for step in steps)


def test_build_preview_image_renders_raw_and_mask_projection(app):
    raw = np.zeros((18, 96, 96), dtype=np.uint8)
    raw[4:15, 20:70, 22:74] = 180
    raw[7:12, 35:60, 38:66] = 255
    mask = np.zeros_like(raw, dtype=np.uint8)
    mask[8:11, 40:58, 42:63] = 1

    image, projection = _build_preview_image(raw, mask, (0.45, 0.45, 0.39))

    assert image.width() > 0
    assert image.height() > 0
    assert projection.extent_xyz.shape == (3,)
    assert projection.box_norm_xy.shape == (8, 2)


def test_build_preview_image_supports_multiple_view_modes(app):
    raw = np.zeros((10, 40, 40), dtype=np.uint8)
    raw[2:8, 10:30, 12:32] = 200
    mask = np.zeros_like(raw, dtype=np.uint8)
    mask[4:6, 15:24, 16:28] = 1

    for mode in ("oblique", "xy", "xz", "yz"):
        image, projection = _build_preview_image(raw, mask, (0.5, 0.5, 0.4), view_mode=mode)
        assert image.width() > 0
        assert image.height() > 0
        assert projection.box_norm_xy.shape == (8, 2)


def test_slice_canvas_wheel_zoom_uses_larger_step():
    canvas = SliceCanvas()
    canvas.resize(320, 320)
    canvas.set_image(np.zeros((64, 64), dtype=np.uint8), reset_view=True)
    before = canvas.zoom_factor()

    QApplication.sendEvent(canvas.viewport(), _wheel_event_for(canvas.viewport(), 120))

    assert canvas.zoom_factor() > before * 1.1


def test_main_controller_inline_preview_toggle_updates_projection(app):
    widget = MainController()
    try:
        data = np.random.randint(0, 255, (14, 64, 64), dtype=np.uint8)
        widget.model.data = data
        widget.model.original_data = data.copy()
        widget.model.ensure_masks()
        widget.model.masks[5:9, 18:40, 18:40] = 1
        widget.slider.setRange(0, data.shape[0] - 1)
        widget.slider.setEnabled(True)
        widget._update_view(reset_view=True)

        widget.inline_volume_chk.setChecked(True)
        app.processEvents()
        widget._refresh_inline_volume_preview()
        widget._last_canvas_cursor_voxel = (24, 28)
        widget._refresh_inline_volume_locator()
        assert _wait_until(app, lambda: widget.inline_volume_preview._base_image is not None)

        assert widget._inline_volume_enabled is True
        assert widget.inline_view_combo.currentData() == "xy"
        assert widget.inline_volume_preview.view_mode() == "xy"
        assert widget.inline_volume_preview._base_image is not None
        assert widget.inline_volume_preview._projection is not None
        assert widget.inline_volume_preview._locator_xyz is not None

        widget.inline_volume_chk.setChecked(False)
        app.processEvents()
        assert widget._inline_volume_enabled is False
        assert widget.inline_volume_preview._projection is None
    finally:
        widget.model.mask_dirty = False
        widget.close()
        widget.deleteLater()
        app.processEvents()


def test_main_controller_inline_preview_skips_rebuild_for_slice_only_changes(app):
    widget = MainController()
    try:
        data = np.random.randint(0, 255, (12, 48, 48), dtype=np.uint8)
        widget.model.data = data
        widget.model.original_data = data.copy()
        widget.model.ensure_masks()
        widget.slider.setRange(0, data.shape[0] - 1)
        widget.slider.setEnabled(True)
        widget._update_view(reset_view=True)
        widget.inline_volume_chk.setChecked(True)
        app.processEvents()

        calls = {"count": 0}
        original_set_volume = widget.inline_volume_preview.set_volume

        def wrapped_set_volume(*args, **kwargs):
            calls["count"] += 1
            return original_set_volume(*args, **kwargs)

        widget.inline_volume_preview.set_volume = wrapped_set_volume

        widget._refresh_inline_volume_preview()
        assert calls["count"] == 1

        widget.model.index = 3
        widget._refresh_inline_volume_preview()
        assert calls["count"] == 1

        widget.inline_view_combo.setCurrentText("XY")
        widget._refresh_inline_volume_preview()
        assert calls["count"] == 1
        assert widget.inline_volume_preview.view_mode() == "xy"
        assert _wait_until(
            app,
            lambda: widget.inline_volume_preview._render_thread is None
            and widget.inline_volume_preview._projection is not None,
        )
    finally:
        widget.model.mask_dirty = False
        widget.close()
        widget.deleteLater()
        app.processEvents()


def test_inline_volume_preview_renders_new_view_in_background(app, monkeypatch):
    preview = InlineVolumePreview()
    raw = np.random.randint(0, 255, (16, 64, 64), dtype=np.uint8)
    mask = np.zeros_like(raw, dtype=np.uint8)
    mask[4:12, 20:40, 18:42] = 1

    original_build = inline_volume_preview._build_preview_image

    def delayed_build(*args, **kwargs):
        time.sleep(0.06)
        return original_build(*args, **kwargs)

    monkeypatch.setattr(inline_volume_preview, "_build_preview_image", delayed_build)

    preview.set_volume(raw, mask, (0.4, 0.4, 0.5), cache_key=("stack-a",))

    assert preview._render_thread is not None
    assert "background" in preview._status_text.lower()
    assert _wait_until(app, lambda: preview._render_thread is None and preview._base_image is not None)


def test_inline_volume_preview_drag_rotates_and_g_resets(app):
    preview = InlineVolumePreview()
    preview.resize(360, 360)
    raw = np.random.randint(0, 255, (18, 72, 72), dtype=np.uint8)
    mask = np.zeros_like(raw, dtype=np.uint8)
    mask[5:13, 24:48, 22:50] = 1

    preview.set_view_mode("xy")
    preview.set_volume(raw, mask, (0.4, 0.4, 0.5), cache_key=("stack-b",))
    assert _wait_until(app, lambda: preview._render_thread is None and preview._projection is not None)

    base_rotation = preview._projection.rotation.copy()
    press = QMouseEvent(
        QEvent.MouseButtonPress,
        QPointF(140, 140),
        Qt.LeftButton,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    move = QMouseEvent(
        QEvent.MouseMove,
        QPointF(188, 118),
        Qt.NoButton,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    release = QMouseEvent(
        QEvent.MouseButtonRelease,
        QPointF(188, 118),
        Qt.LeftButton,
        Qt.NoButton,
        Qt.NoModifier,
    )
    QApplication.sendEvent(preview, press)
    QApplication.sendEvent(preview, move)
    assert _wait_until(app, lambda: not preview._render_timer.isActive())
    yaw_offset, pitch_offset = preview.rotation_offsets()
    assert abs(yaw_offset) > 0.1
    assert abs(pitch_offset) > 0.1
    assert not np.allclose(preview._projection.rotation, base_rotation)

    QApplication.sendEvent(preview, release)
    QApplication.sendEvent(preview, QKeyEvent(QEvent.KeyPress, Qt.Key_G, Qt.NoModifier))
    assert _wait_until(app, lambda: not preview._render_timer.isActive())
    yaw_offset, pitch_offset = preview.rotation_offsets()
    assert yaw_offset == pytest.approx(0.0)
    assert pitch_offset == pytest.approx(0.0)
    assert np.allclose(preview._projection.rotation, base_rotation)


def test_inline_volume_preview_wheel_zoom_scales_without_rebuild(app):
    preview = InlineVolumePreview()
    preview.resize(360, 360)
    raw = np.random.randint(0, 255, (16, 64, 64), dtype=np.uint8)
    mask = np.zeros_like(raw, dtype=np.uint8)
    mask[4:11, 22:46, 20:48] = 1

    preview.set_view_mode("xy")
    preview.set_volume(raw, mask, (0.4, 0.4, 0.5), cache_key=("stack-c",))
    assert _wait_until(app, lambda: preview._render_thread is None and preview._projection is not None)

    before_zoom = preview.zoom_factor()
    before_width = preview._image_target_rect(preview._content_rect()).width()
    center = preview.rect().center()
    wheel_in = QWheelEvent(
        QPointF(center),
        QPointF(center),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.ScrollUpdate,
        False,
    )
    QApplication.sendEvent(preview, wheel_in)
    app.processEvents()

    assert preview.zoom_factor() > before_zoom
    assert preview._image_target_rect(preview._content_rect()).width() > before_width
    assert preview._render_thread is None


def test_main_controller_canvas_and_inline_zoom_stay_in_sync(app):
    widget = MainController()
    try:
        data = np.random.randint(0, 255, (12, 64, 64), dtype=np.uint8)
        widget.model.data = data
        widget.model.original_data = data.copy()
        widget.model.ensure_masks()
        widget.slider.setRange(0, data.shape[0] - 1)
        widget.slider.setEnabled(True)
        widget._update_view(reset_view=True)
        widget.inline_volume_chk.setChecked(True)
        app.processEvents()
        widget._refresh_inline_volume_preview()
        assert _wait_until(app, lambda: widget.inline_volume_preview._base_image is not None)

        canvas_before = widget.canvas.zoom_factor()
        preview_before = widget.inline_volume_preview.zoom_factor()
        QApplication.sendEvent(
            widget.canvas.viewport(),
            _wheel_event_for(widget.canvas.viewport(), 120),
        )
        app.processEvents()

        assert widget.canvas.zoom_factor() > canvas_before
        assert widget.inline_volume_preview.zoom_factor() > preview_before

        canvas_mid = widget.canvas.zoom_factor()
        preview_mid = widget.inline_volume_preview.zoom_factor()
        QApplication.sendEvent(
            widget.inline_volume_preview,
            _wheel_event_for(widget.inline_volume_preview, 120),
        )
        app.processEvents()

        assert widget.inline_volume_preview.zoom_factor() > preview_mid
        assert widget.canvas.zoom_factor() > canvas_mid
    finally:
        widget.model.mask_dirty = False
        widget.close()
        widget.deleteLater()
        app.processEvents()
