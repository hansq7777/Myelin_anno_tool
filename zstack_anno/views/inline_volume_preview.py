from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from PyQt5.QtCore import QPoint, QPointF, QRectF, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
from PyQt5.QtWidgets import QWidget


_MAX_IMAGE_SIZE = 960
_MAX_RAW_POINTS = 52000
_MAX_MASK_POINTS = 30000
_MAX_INTERACTIVE_RAW_POINTS = 14000
_MAX_INTERACTIVE_MASK_POINTS = 9000
_MAX_VOXELS = 2_200_000
_TARGET_SHAPE_ZYX = (96, 224, 224)
_ROTATION_SENSITIVITY_DEG = 0.55
_MAX_PITCH_OFFSET_DEG = 82.0
_MIN_VIEW_ZOOM = 1.0
_MAX_VIEW_ZOOM = 12.0
_WHEEL_ZOOM_FACTOR = 1.14
VIEW_MODE_LABELS = (
    ("oblique", "Oblique"),
    ("xy", "XY"),
    ("xz", "XZ"),
    ("yz", "YZ"),
)


@dataclass(frozen=True)
class _ProjectionState:
    rotation: np.ndarray
    center_xyz: np.ndarray
    min_xy: np.ndarray
    span: float
    extent_xyz: np.ndarray
    box_norm_xy: np.ndarray


@dataclass(frozen=True)
class _PreparedPreviewData:
    extent_xyz: np.ndarray
    raw_points_xyz: np.ndarray
    raw_intensity: np.ndarray
    raw_points_xyz_interactive: np.ndarray
    raw_intensity_interactive: np.ndarray
    mask_points_xyz: np.ndarray
    mask_points_xyz_interactive: np.ndarray


class _PreviewPrepareThread(QThread):
    rendered = pyqtSignal(object, object, object)
    failed = pyqtSignal(object, object, str)

    def __init__(
        self,
        *,
        request_id: int,
        render_key: object,
        raw_zyx: np.ndarray,
        mask_zyx: np.ndarray | None,
        spacing_xyz: tuple[float, float, float],
    ) -> None:
        super().__init__(None)
        self.request_id = request_id
        self.render_key = render_key
        self.raw_zyx = raw_zyx
        self.mask_zyx = mask_zyx
        self.spacing_xyz = spacing_xyz

    def run(self) -> None:
        try:
            prepared = _prepare_preview_data(self.raw_zyx, self.mask_zyx, self.spacing_xyz)
        except Exception as exc:
            self.failed.emit(self.request_id, self.render_key, str(exc))
            return
        self.rendered.emit(self.request_id, self.render_key, prepared)


class InlineVolumePreview(QWidget):
    """Lightweight interactive 3-D preview for the current stack."""
    zoomAdjusted = pyqtSignal(float)
    viewWindowChanged = pyqtSignal(float, float, float, float)
    rotationChanged = pyqtSignal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(260)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self._base_image: QImage | None = None
        self._projection: _ProjectionState | None = None
        self._prepared_data: _PreparedPreviewData | None = None
        self._locator_xyz: tuple[float, float, float] | None = None
        self._current_slice_z: float | None = None
        self._view_mode = "xy"
        self._status_text = "Enable 3D visualization to render a lightweight preview."
        self._volume_key: object | None = None
        self._raw_zyx: np.ndarray | None = None
        self._mask_zyx: np.ndarray | None = None
        self._spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0)
        self._prepared_cache: dict[object, _PreparedPreviewData] = {}
        self._prepared_cache_order: list[object] = []
        self._render_thread: _PreviewPrepareThread | None = None
        self._active_render_request_id: int | None = None
        self._active_render_key: object | None = None
        self._queued_render_requested = False
        self._render_request_seq = 0
        self._yaw_offset_deg = 0.0
        self._pitch_offset_deg = 0.0
        self._pan_active = False
        self._rotate_active = False
        self._drag_last_pos: QPoint | None = None
        self._pending_render_interactive = False
        self._view_zoom = 1.0
        self._view_center_norm = np.array([0.5, 0.5], dtype=np.float32)
        self._display_volume_identity: object | None = None
        self._review_badge_text = ""
        self._review_badge_done = False
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._render_pending_preview)

    def clear_preview(self) -> None:
        self._base_image = None
        self._projection = None
        self._prepared_data = None
        self._locator_xyz = None
        self._current_slice_z = None
        self._volume_key = None
        self._raw_zyx = None
        self._mask_zyx = None
        self._prepared_cache.clear()
        self._prepared_cache_order.clear()
        self._queued_render_requested = False
        self._active_render_key = None
        self._active_render_request_id = None
        self._yaw_offset_deg = 0.0
        self._pitch_offset_deg = 0.0
        self._pan_active = False
        self._rotate_active = False
        self._drag_last_pos = None
        self._view_zoom = 1.0
        self._view_center_norm = np.array([0.5, 0.5], dtype=np.float32)
        self._display_volume_identity = None
        self._review_badge_text = ""
        self._review_badge_done = False
        self._render_timer.stop()
        self.unsetCursor()
        self._status_text = "Enable 3D visualization to render a lightweight preview."
        self.update()

    def clear_locator(self) -> None:
        self._locator_xyz = None
        self.update()

    def set_review_badge(self, text: str, *, done: bool = False) -> None:
        self._review_badge_text = (text or "").strip().upper()
        self._review_badge_done = bool(done)
        self.update()

    def set_view_mode(self, mode: str, *, render_if_ready: bool = True) -> None:
        normalized = _normalize_view_mode(mode)
        if normalized == self._view_mode:
            return
        self._view_mode = normalized
        self._yaw_offset_deg = 0.0
        self._pitch_offset_deg = 0.0
        self.rotationChanged.emit(0.0)
        if render_if_ready and self._volume_key is not None and self._raw_zyx is not None:
            self._apply_render_for_current_view()
        else:
            self.update()

    def view_mode(self) -> str:
        return self._view_mode

    def rotation_offsets(self) -> tuple[float, float]:
        return float(self._yaw_offset_deg), float(self._pitch_offset_deg)

    def zoom_factor(self) -> float:
        return float(self._view_zoom)

    def apply_zoom_factor(
        self,
        factor: float,
        *,
        anchor_norm: tuple[float, float] | None = None,
        emit_signal: bool = False,
    ) -> float:
        factor = float(factor)
        if factor <= 0.0:
            return float(self._view_zoom)
        old_zoom = float(self._view_zoom)
        new_zoom = float(np.clip(old_zoom * factor, _MIN_VIEW_ZOOM, _MAX_VIEW_ZOOM))
        if abs(new_zoom - old_zoom) < 1e-6:
            return float(self._view_zoom)
        if anchor_norm is None:
            anchor = np.array([0.5, 0.5], dtype=np.float32)
        else:
            anchor = np.clip(np.asarray(anchor_norm, dtype=np.float32), 0.0, 1.0)
        center = np.asarray(self._view_center_norm, dtype=np.float32)
        offset = (anchor - center) / max(old_zoom, 1e-6)
        self._view_zoom = new_zoom
        self._view_center_norm = anchor - offset * new_zoom
        self._clamp_view_center()
        self._status_text = (
            f"{_view_mode_label(self._view_mode)} view | wheel zoom {self._view_zoom:.2f}x | "
            "left-drag pan | right-drag rotate | G reset."
        )
        self.update()
        if emit_signal:
            self.zoomAdjusted.emit(new_zoom / max(old_zoom, 1e-9))
            self._emit_view_window_changed()
        return float(self._view_zoom)

    def set_planar_view_window(
        self,
        *,
        center_xy_norm: tuple[float, float],
        visible_fraction_xy: tuple[float, float],
    ) -> None:
        center = np.clip(np.asarray(center_xy_norm, dtype=np.float32), 0.0, 1.0)
        visible = np.clip(np.asarray(visible_fraction_xy, dtype=np.float32), 0.05, 1.0)
        target_zoom = float(
            np.clip(1.0 / max(float(visible[0]), float(visible[1]), 1e-6), _MIN_VIEW_ZOOM, _MAX_VIEW_ZOOM)
        )
        changed = (
            abs(float(self._view_zoom) - target_zoom) > 1e-4
            or not np.allclose(self._view_center_norm, center, atol=1e-4)
        )
        self._view_zoom = target_zoom
        self._view_center_norm = center
        self._clamp_view_center()
        if changed:
            self.update()

    def reset_view_rotation(self) -> None:
        self._yaw_offset_deg = 0.0
        self._pitch_offset_deg = 0.0
        self.rotationChanged.emit(0.0)
        self._request_preview_render(interactive=False, delay_ms=0)

    def set_locator(self, x: int, y: int, z: int, spacing_xyz: tuple[float, float, float]) -> None:
        sx, sy, sz = (float(v) for v in spacing_xyz)
        self._locator_xyz = (float(x) * sx, float(y) * sy, float(z) * sz)
        self._current_slice_z = float(z) * sz
        self.update()

    def set_current_slice(self, z: int, spacing_xyz: tuple[float, float, float]) -> None:
        self._current_slice_z = float(z) * float(spacing_xyz[2])
        self.update()

    def set_volume(
        self,
        raw_zyx: np.ndarray | None,
        mask_zyx: np.ndarray | None,
        spacing_xyz: tuple[float, float, float],
        *,
        cache_key: object | None = None,
    ) -> None:
        if raw_zyx is None:
            self.clear_preview()
            return
        display_identity = (
            int(id(raw_zyx)),
            tuple(int(v) for v in np.asarray(raw_zyx).shape),
            tuple(float(v) for v in spacing_xyz),
        )
        if display_identity != self._display_volume_identity:
            self._view_zoom = 1.0
            self._view_center_norm = np.array([0.5, 0.5], dtype=np.float32)
            self._display_volume_identity = display_identity
        self._raw_zyx = np.asarray(raw_zyx)
        self._mask_zyx = None if mask_zyx is None else np.asarray(mask_zyx)
        self._spacing_xyz = tuple(float(v) for v in spacing_xyz)
        self._volume_key = cache_key if cache_key is not None else (
            self._raw_zyx.shape,
            None if self._mask_zyx is None else self._mask_zyx.shape,
            self._spacing_xyz,
        )
        self._prepared_data = self._prepared_cache.get(self._volume_key)
        self._apply_render_for_current_view()

    def rotate_by_drag_delta(self, dx: float, dy: float, *, interactive: bool = True) -> None:
        self._yaw_offset_deg += float(dx) * _ROTATION_SENSITIVITY_DEG
        self._pitch_offset_deg = float(
            np.clip(
                self._pitch_offset_deg + float(dy) * _ROTATION_SENSITIVITY_DEG,
                -_MAX_PITCH_OFFSET_DEG,
                _MAX_PITCH_OFFSET_DEG,
            )
        )
        self.rotationChanged.emit(float(self._yaw_offset_deg))
        self._request_preview_render(interactive=interactive, delay_ms=0 if not interactive else 12)

    def pan_by_drag_delta(self, dx: float, dy: float) -> None:
        target = self._image_target_rect(self._content_rect())
        width = max(target.width(), 1e-6)
        height = max(target.height(), 1e-6)
        center = np.asarray(self._view_center_norm, dtype=np.float32).copy()
        center[0] -= float(dx) / width
        center[1] -= float(dy) / height
        self._view_center_norm = center
        self._clamp_view_center()
        self._status_text = (
            f"{_view_mode_label(self._view_mode)} view | wheel zoom {self._view_zoom:.2f}x | "
            "left-drag pan | right-drag rotate | G reset."
        )
        self.update()
        self._emit_view_window_changed()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(248, 247, 243))

        viewport = self._content_rect()
        target = self._image_target_rect(viewport)
        if self._base_image is not None and self._projection is not None:
            painter.save()
            painter.setClipRect(viewport)
            painter.drawImage(target, self._base_image)
            self._draw_slice_plane(painter, target)
            self._draw_locator(painter, target)
            painter.restore()
        else:
            painter.setPen(QColor(90, 90, 90))
            painter.drawText(viewport, Qt.AlignCenter | Qt.TextWordWrap, self._status_text)

        painter.setPen(QColor(95, 95, 95))
        painter.drawText(
            QRectF(12.0, float(self.height() - 34), float(max(0, self.width() - 24)), 22.0),
            Qt.AlignLeft | Qt.AlignVCenter,
            self._status_text,
        )
        self._draw_review_badge(painter)
        painter.end()

    def mousePressEvent(self, event) -> None:
        if self._prepared_data is None:
            super().mousePressEvent(event)
            return
        if self._is_secondary_button_event(event):
            self.setFocus(Qt.MouseFocusReason)
            self._rotate_active = True
            self._drag_last_pos = event.pos()
            self.setCursor(Qt.SizeAllCursor)
            event.accept()
            return
        if event.button() == Qt.LeftButton:
            self.setFocus(Qt.MouseFocusReason)
            self._pan_active = True
            self._drag_last_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._pan_active and self._drag_last_pos is not None and self._prepared_data is not None:
            dx = event.pos().x() - self._drag_last_pos.x()
            dy = event.pos().y() - self._drag_last_pos.y()
            self._drag_last_pos = event.pos()
            self.pan_by_drag_delta(dx, dy)
            event.accept()
            return
        if self._rotate_active and self._drag_last_pos is not None and self._prepared_data is not None:
            dx = event.pos().x() - self._drag_last_pos.x()
            dy = event.pos().y() - self._drag_last_pos.y()
            self._drag_last_pos = event.pos()
            self.rotate_by_drag_delta(dx, dy, interactive=True)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._pan_active:
            self._pan_active = False
            self._drag_last_pos = None
            if self._prepared_data is not None:
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.unsetCursor()
            event.accept()
            return
        if (event.button() == Qt.RightButton or (event.button() == Qt.LeftButton and self._rotate_active)) and self._rotate_active:
            self._rotate_active = False
            self._drag_last_pos = None
            if self._prepared_data is not None:
                self.setCursor(Qt.OpenHandCursor)
                self._request_preview_render(interactive=False, delay_ms=0)
            else:
                self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_G and self._prepared_data is not None:
            self.reset_view_rotation()
            event.accept()
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event) -> None:
        if self._base_image is None or self._projection is None:
            super().wheelEvent(event)
            return
        viewport = self._content_rect()
        pos = QPointF(event.position()) if hasattr(event, "position") else QPointF(event.pos())
        if not viewport.contains(pos):
            super().wheelEvent(event)
            return
        wheel_steps = _wheel_steps(event)
        if abs(wheel_steps) < 1e-6:
            event.accept()
            return
        factor = _WHEEL_ZOOM_FACTOR ** wheel_steps
        old_target = self._image_target_rect(viewport)
        anchor_norm = (
            (pos.x() - old_target.left()) / max(old_target.width(), 1e-6),
            (pos.y() - old_target.top()) / max(old_target.height(), 1e-6),
        )
        self.apply_zoom_factor(factor, anchor_norm=anchor_norm, emit_signal=True)
        event.accept()

    def focusOutEvent(self, event) -> None:
        self._pan_active = False
        self._rotate_active = False
        self._drag_last_pos = None
        if self._prepared_data is not None:
            self.setCursor(Qt.OpenHandCursor)
        else:
            self.unsetCursor()
        super().focusOutEvent(event)

    def _apply_render_for_current_view(self) -> None:
        if self._volume_key is None or self._raw_zyx is None:
            self._base_image = None
            self._projection = None
            self._prepared_data = None
            self.update()
            return
        cached = self._prepared_cache.get(self._volume_key)
        if cached is None:
            self._prepared_data = None
            self._request_background_prepare(self._volume_key)
            return
        self._prepared_data = cached
        self._request_preview_render(interactive=False, delay_ms=0)
        if not self._has_active_pointer_interaction():
            self.setCursor(Qt.OpenHandCursor)

    def _request_background_prepare(self, render_key: object) -> None:
        if self._raw_zyx is None:
            return
        if self._render_thread is not None and self._render_thread.isRunning():
            if self._active_render_key != render_key:
                self._queued_render_requested = True
                self._status_text = "Preparing 3D preview data in background (queued)..."
                self.update()
            return
        self._queued_render_requested = False
        self._render_request_seq += 1
        request_id = self._render_request_seq
        self._active_render_request_id = request_id
        self._active_render_key = render_key
        raw_copy = np.array(self._raw_zyx, copy=True, order="C")
        mask_copy = None if self._mask_zyx is None else np.array(self._mask_zyx, copy=True, order="C")
        thread = _PreviewPrepareThread(
            request_id=request_id,
            render_key=render_key,
            raw_zyx=raw_copy,
            mask_zyx=mask_copy,
            spacing_xyz=self._spacing_xyz,
        )
        thread.rendered.connect(self._on_background_prepared)
        thread.failed.connect(self._on_background_prepare_failed)
        thread.finished.connect(self._on_background_prepare_finished)
        thread.finished.connect(thread.deleteLater)
        self._render_thread = thread
        self._status_text = "Preparing 3D preview data in background..."
        self.update()
        thread.start()

    def _on_background_prepared(
        self,
        request_id: object,
        render_key: object,
        prepared: object,
    ) -> None:
        if self._active_render_request_id is not None and request_id != self._active_render_request_id:
            return
        if not isinstance(prepared, _PreparedPreviewData):
            return
        self._remember_prepared_cache(render_key, prepared)
        if self._volume_key is not None and render_key == self._volume_key:
            self._prepared_data = prepared
            if not self._has_active_pointer_interaction():
                self.setCursor(Qt.OpenHandCursor)
            self._request_preview_render(interactive=False, delay_ms=0)

    def _on_background_prepare_failed(
        self,
        request_id: object,
        render_key: object,
        error_text: str,
    ) -> None:
        if self._active_render_request_id is not None and request_id != self._active_render_request_id:
            return
        if self._volume_key is not None and render_key == self._volume_key:
            self._base_image = None
            self._projection = None
            self._prepared_data = None
            self._status_text = f"3D preview unavailable: {error_text}"
            self.unsetCursor()
            self.update()

    def _on_background_prepare_finished(self) -> None:
        self._render_thread = None
        self._active_render_key = None
        self._active_render_request_id = None
        if self._queued_render_requested:
            self._queued_render_requested = False
            self._apply_render_for_current_view()

    def _remember_prepared_cache(self, render_key: object, prepared: _PreparedPreviewData) -> None:
        self._prepared_cache[render_key] = prepared
        if render_key in self._prepared_cache_order:
            self._prepared_cache_order.remove(render_key)
        self._prepared_cache_order.append(render_key)
        while len(self._prepared_cache_order) > 4:
            stale = self._prepared_cache_order.pop(0)
            self._prepared_cache.pop(stale, None)

    def _request_preview_render(self, *, interactive: bool, delay_ms: int) -> None:
        if self._prepared_data is None:
            return
        if delay_ms <= 0:
            self._render_timer.stop()
            self._pending_render_interactive = interactive
            self._render_pending_preview()
            return
        if not self._render_timer.isActive():
            self._pending_render_interactive = interactive
        else:
            self._pending_render_interactive = self._pending_render_interactive and interactive
        self._render_timer.start(max(0, int(delay_ms)))

    def _render_pending_preview(self) -> None:
        interactive = bool(self._pending_render_interactive)
        self._pending_render_interactive = False
        self._render_preview_now(interactive=interactive)

    def _render_preview_now(self, *, interactive: bool) -> None:
        if self._prepared_data is None:
            return
        image, projection = _render_prepared_preview_image(
            self._prepared_data,
            view_mode=self._view_mode,
            yaw_offset_deg=self._yaw_offset_deg,
            pitch_offset_deg=self._pitch_offset_deg,
            interactive=interactive,
        )
        self._base_image = image
        self._projection = projection
        hint = "interactive" if interactive else "full"
        self._status_text = (
            f"{_view_mode_label(self._view_mode)} view | wheel zoom {self._view_zoom:.2f}x | "
            f"left-drag pan | right-drag rotate | G reset | {hint} preview."
        )
        self.update()

    def normalized_view_window(self) -> tuple[tuple[float, float], tuple[float, float]]:
        zoom = max(_MIN_VIEW_ZOOM, float(self._view_zoom))
        frac = float(np.clip(1.0 / zoom, 0.05, 1.0))
        center = np.clip(np.asarray(self._view_center_norm, dtype=np.float32), 0.0, 1.0)
        return (float(center[0]), float(center[1])), (frac, frac)

    def _emit_view_window_changed(self) -> None:
        center, frac = self.normalized_view_window()
        self.viewWindowChanged.emit(center[0], center[1], frac[0], frac[1])

    def _has_active_pointer_interaction(self) -> bool:
        return bool(self._pan_active or self._rotate_active)

    @staticmethod
    def _is_secondary_button_event(event) -> bool:
        if event.button() == Qt.RightButton:
            return True
        if event.button() != Qt.LeftButton:
            return False
        return bool(event.modifiers() & Qt.ControlModifier)

    def _content_rect(self) -> QRectF:
        margin = 12.0
        footer = 42.0
        width = max(40.0, float(self.width()) - 2.0 * margin)
        height = max(40.0, float(self.height()) - margin - footer)
        size = min(width, height)
        left = margin + (width - size)
        top = margin + (height - size) / 2.0
        return QRectF(left, top, size, size)

    def _image_target_rect(self, viewport: QRectF | None = None) -> QRectF:
        view = self._content_rect() if viewport is None else viewport
        zoom = max(_MIN_VIEW_ZOOM, float(self._view_zoom))
        width = view.width() * zoom
        height = view.height() * zoom
        center = np.asarray(self._view_center_norm, dtype=np.float32)
        left = view.center().x() - float(center[0]) * width
        top = view.center().y() - float(center[1]) * height
        return QRectF(left, top, width, height)

    def _clamp_view_center(self) -> None:
        zoom = max(_MIN_VIEW_ZOOM, float(self._view_zoom))
        min_center = 0.5 / zoom
        max_center = 1.0 - min_center
        if max_center < min_center:
            min_center = max_center = 0.5
        self._view_center_norm = np.clip(
            np.asarray(self._view_center_norm, dtype=np.float32),
            min_center,
            max_center,
        )

    def _draw_slice_plane(self, painter: QPainter, target: QRectF) -> None:
        if self._projection is None or self._current_slice_z is None:
            return
        z = float(np.clip(self._current_slice_z, 0.0, self._projection.extent_xyz[2]))
        extent_x, extent_y, _extent_z = self._projection.extent_xyz
        plane_xyz = np.array(
            [
                [0.0, 0.0, z],
                [extent_x, 0.0, z],
                [extent_x, extent_y, z],
                [0.0, extent_y, z],
            ],
            dtype=np.float32,
        )
        plane = _project_world_points(plane_xyz, self._projection)
        polygon = QPolygonF(_map_norm_points_to_rect(plane, target))
        painter.setPen(QPen(QColor(50, 140, 190, 165), 1.6))
        painter.setBrush(QColor(80, 170, 210, 32))
        painter.drawPolygon(polygon)

    def _draw_locator(self, painter: QPainter, target: QRectF) -> None:
        if self._projection is None or self._locator_xyz is None:
            return
        point = _project_world_points(np.array([self._locator_xyz], dtype=np.float32), self._projection)[0]
        mapped = _map_norm_point_to_rect(point, target)
        painter.setPen(QPen(QColor(18, 120, 184), 2.2))
        painter.setBrush(QColor(255, 255, 255, 220))
        radius = 5.0
        painter.drawEllipse(mapped, radius, radius)
        painter.drawLine(QPointF(mapped.x() - 8.0, mapped.y()), QPointF(mapped.x() + 8.0, mapped.y()))
        painter.drawLine(QPointF(mapped.x(), mapped.y() - 8.0), QPointF(mapped.x(), mapped.y() + 8.0))

    def _draw_review_badge(self, painter: QPainter) -> None:
        if not self._review_badge_text:
            return
        text = self._review_badge_text
        bg, fg = _review_badge_palette(text, self._review_badge_done)
        rect = QRectF(14.0, 14.0, 74.0, 30.0)
        painter.setPen(Qt.NoPen)
        painter.setBrush(bg)
        painter.drawRoundedRect(rect, 9.0, 9.0)
        painter.setPen(fg)
        painter.drawText(rect, Qt.AlignCenter, text)


def _build_preview_image(
    raw_zyx: np.ndarray,
    mask_zyx: np.ndarray | None,
    spacing_xyz: tuple[float, float, float],
    *,
    view_mode: str = "oblique",
    yaw_offset_deg: float = 0.0,
    pitch_offset_deg: float = 0.0,
    interactive: bool = False,
) -> tuple[QImage, _ProjectionState]:
    prepared = _prepare_preview_data(raw_zyx, mask_zyx, spacing_xyz)
    return _render_prepared_preview_image(
        prepared,
        view_mode=view_mode,
        yaw_offset_deg=yaw_offset_deg,
        pitch_offset_deg=pitch_offset_deg,
        interactive=interactive,
    )


def _prepare_preview_data(
    raw_zyx: np.ndarray,
    mask_zyx: np.ndarray | None,
    spacing_xyz: tuple[float, float, float],
) -> _PreparedPreviewData:
    raw = np.asarray(raw_zyx)
    if raw.ndim != 3:
        raise ValueError(f"Expected 3-D raw stack, got shape={raw.shape}")
    mask = None if mask_zyx is None else np.asarray(mask_zyx)
    if mask is not None and mask.shape != raw.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match raw shape {raw.shape}")

    steps_zyx = _choose_downsample_steps(raw.shape)
    raw_small = raw[:: steps_zyx[0], :: steps_zyx[1], :: steps_zyx[2]]
    mask_small = None if mask is None else mask[:: steps_zyx[0], :: steps_zyx[1], :: steps_zyx[2]]

    spacing_sampled_xyz = (
        float(spacing_xyz[0]) * steps_zyx[2],
        float(spacing_xyz[1]) * steps_zyx[1],
        float(spacing_xyz[2]) * steps_zyx[0],
    )
    extent_xyz = _extent_xyz_for_shape(raw_small.shape, spacing_sampled_xyz)
    raw_points_xyz, raw_intensity = _select_raw_points(raw_small, spacing_sampled_xyz)
    mask_points_xyz = _select_mask_points(mask_small, spacing_sampled_xyz)
    raw_points_xyz_interactive, raw_intensity_interactive = _limit_point_cloud(
        raw_points_xyz,
        raw_intensity,
        _MAX_INTERACTIVE_RAW_POINTS,
    )
    mask_points_xyz_interactive, _ = _limit_point_cloud(
        mask_points_xyz,
        None,
        _MAX_INTERACTIVE_MASK_POINTS,
    )
    return _PreparedPreviewData(
        extent_xyz=extent_xyz,
        raw_points_xyz=raw_points_xyz,
        raw_intensity=raw_intensity,
        raw_points_xyz_interactive=raw_points_xyz_interactive,
        raw_intensity_interactive=raw_intensity_interactive,
        mask_points_xyz=mask_points_xyz,
        mask_points_xyz_interactive=mask_points_xyz_interactive,
    )


def _render_prepared_preview_image(
    prepared: _PreparedPreviewData,
    *,
    view_mode: str,
    yaw_offset_deg: float = 0.0,
    pitch_offset_deg: float = 0.0,
    interactive: bool = False,
) -> tuple[QImage, _ProjectionState]:
    projection = _build_projection_state(
        prepared.extent_xyz,
        view_mode=view_mode,
        yaw_offset_deg=yaw_offset_deg,
        pitch_offset_deg=pitch_offset_deg,
    )
    if interactive:
        raw_points_xyz = prepared.raw_points_xyz_interactive
        raw_intensity = prepared.raw_intensity_interactive
        mask_points_xyz = prepared.mask_points_xyz_interactive
    else:
        raw_points_xyz = prepared.raw_points_xyz
        raw_intensity = prepared.raw_intensity
        mask_points_xyz = prepared.mask_points_xyz

    raw_norm = _project_world_points(raw_points_xyz, projection) if raw_points_xyz.size else np.empty((0, 2), dtype=np.float32)
    mask_norm = _project_world_points(mask_points_xyz, projection) if mask_points_xyz.size else np.empty((0, 2), dtype=np.float32)

    image = QImage(_MAX_IMAGE_SIZE, _MAX_IMAGE_SIZE, QImage.Format_ARGB32_Premultiplied)
    image.fill(QColor(252, 252, 250))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    _draw_box(painter, projection.box_norm_xy, image.width(), image.height())
    _draw_raw_points(painter, raw_norm, raw_intensity, image.width(), image.height())
    _draw_mask_points(painter, mask_norm, image.width(), image.height())
    painter.end()
    return image, projection


def _choose_downsample_steps(shape_zyx: tuple[int, int, int]) -> tuple[int, int, int]:
    dims = np.asarray(shape_zyx, dtype=np.int32)
    targets = np.asarray(_TARGET_SHAPE_ZYX, dtype=np.int32)
    steps = np.maximum(1, np.ceil(dims / targets).astype(np.int32))
    while np.prod(np.ceil(dims / steps).astype(np.int32)) > _MAX_VOXELS:
        ratios = dims / steps.astype(np.float32)
        axis = int(np.argmax(ratios))
        steps[axis] += 1
    return int(steps[0]), int(steps[1]), int(steps[2])


def _extent_xyz_for_shape(
    shape_zyx: tuple[int, int, int],
    spacing_xyz: tuple[float, float, float],
) -> np.ndarray:
    sx, sy, sz = (float(v) for v in spacing_xyz)
    return np.array(
        [
            max(1.0, (shape_zyx[2] - 1) * sx),
            max(1.0, (shape_zyx[1] - 1) * sy),
            max(1.0, (shape_zyx[0] - 1) * sz),
        ],
        dtype=np.float32,
    )


def _build_projection_state(
    extent_xyz: np.ndarray,
    *,
    view_mode: str,
    yaw_offset_deg: float = 0.0,
    pitch_offset_deg: float = 0.0,
) -> _ProjectionState:
    extent = np.asarray(extent_xyz, dtype=np.float32)
    center_xyz = extent / 2.0
    rotation = _build_rotation_matrix(
        view_mode,
        yaw_offset_deg=yaw_offset_deg,
        pitch_offset_deg=pitch_offset_deg,
    )
    corners_xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [extent[0], 0.0, 0.0],
            [extent[0], extent[1], 0.0],
            [0.0, extent[1], 0.0],
            [0.0, 0.0, extent[2]],
            [extent[0], 0.0, extent[2]],
            [extent[0], extent[1], extent[2]],
            [0.0, extent[1], extent[2]],
        ],
        dtype=np.float32,
    )
    rotated = (corners_xyz - center_xyz) @ rotation.T
    min_xy = rotated[:, :2].min(axis=0)
    max_xy = rotated[:, :2].max(axis=0)
    span = float(max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1], 1.0))
    padding = 0.08 * span
    min_xy = min_xy - padding
    span = span + 2.0 * padding
    box_norm_xy = (rotated[:, :2] - min_xy) / span
    return _ProjectionState(
        rotation=rotation,
        center_xyz=center_xyz,
        min_xy=min_xy,
        span=float(span),
        extent_xyz=extent,
        box_norm_xy=box_norm_xy.astype(np.float32),
    )


def _build_rotation_matrix(
    view_mode: str,
    *,
    yaw_offset_deg: float = 0.0,
    pitch_offset_deg: float = 0.0,
) -> np.ndarray:
    base = _base_rotation_matrix(view_mode)
    if abs(float(yaw_offset_deg)) < 1e-6 and abs(float(pitch_offset_deg)) < 1e-6:
        return base
    yaw = math.radians(float(yaw_offset_deg))
    pitch = math.radians(float(pitch_offset_deg))
    rz = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(pitch), -math.sin(pitch)],
            [0.0, math.sin(pitch), math.cos(pitch)],
        ],
        dtype=np.float32,
    )
    return rx @ rz @ base


def _base_rotation_matrix(view_mode: str) -> np.ndarray:
    normalized = _normalize_view_mode(view_mode)
    if normalized == "xy":
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    if normalized == "xz":
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
    if normalized == "yz":
        return np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    yaw = math.radians(-45.0)
    pitch = math.radians(35.0)
    rz = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(pitch), -math.sin(pitch)],
            [0.0, math.sin(pitch), math.cos(pitch)],
        ],
        dtype=np.float32,
    )
    return rx @ rz


def _select_raw_points(
    raw_small: np.ndarray,
    spacing_xyz: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(raw_small, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)
    lo = float(np.percentile(arr, 55.0))
    hi = float(np.percentile(arr, 99.8))
    if hi <= lo:
        hi = float(arr.max())
    if hi <= lo:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    structural = _normalized_gradient_score(arr)
    local_contrast = _normalized_local_contrast_score(arr)
    score = np.maximum(norm, 0.62 * norm + 0.28 * structural + 0.26 * local_contrast)
    candidates = np.flatnonzero(score >= 0.085)
    if candidates.size == 0:
        candidates = np.flatnonzero(norm >= 0.07)
    if candidates.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)
    flat_score = score.ravel()
    coords_zyx = np.column_stack(np.unravel_index(candidates, norm.shape)).astype(np.int32)
    values = flat_score[candidates].astype(np.float32)
    keep_idx = _select_indices_with_spatial_coverage(
        coords_zyx,
        values,
        norm.shape,
        _MAX_RAW_POINTS,
        preferred_bins=(6, 10, 10),
        per_cell_quota=3,
    )
    coords_zyx = coords_zyx[keep_idx].astype(np.float32)
    values = values[keep_idx]
    points = _coords_zyx_to_xyz(coords_zyx, spacing_xyz)
    return points, values


def _select_mask_points(
    mask_small: np.ndarray | None,
    spacing_xyz: tuple[float, float, float],
) -> np.ndarray:
    if mask_small is None:
        return np.empty((0, 3), dtype=np.float32)
    mask = np.asarray(mask_small) > 0
    if not mask.any():
        return np.empty((0, 3), dtype=np.float32)
    surface = _surface_mask_zyx(mask)
    coords_zyx = np.argwhere(surface if surface.any() else mask).astype(np.int32)
    keep_idx = _select_indices_with_spatial_coverage(
        coords_zyx,
        np.ones((coords_zyx.shape[0],), dtype=np.float32),
        mask.shape,
        _MAX_MASK_POINTS,
        preferred_bins=(8, 14, 14),
        per_cell_quota=4,
    )
    coords_zyx = coords_zyx[keep_idx].astype(np.float32)
    return _coords_zyx_to_xyz(coords_zyx.astype(np.float32), spacing_xyz)


def _limit_point_cloud(
    points_xyz: np.ndarray,
    intensity: np.ndarray | None,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if points_xyz.size == 0:
        empty_intensity = None if intensity is None else np.empty((0,), dtype=np.float32)
        return np.empty((0, 3), dtype=np.float32), empty_intensity
    if points_xyz.shape[0] <= max_points:
        return points_xyz, intensity
    indices = np.linspace(0, points_xyz.shape[0] - 1, num=max_points, dtype=np.int32)
    limited_points = points_xyz[indices]
    limited_intensity = None if intensity is None else intensity[indices]
    return limited_points, limited_intensity


def _wheel_steps(event) -> float:
    angle_delta = event.angleDelta().y()
    if angle_delta:
        return float(angle_delta) / 120.0
    pixel_delta = event.pixelDelta().y()
    if pixel_delta:
        return float(pixel_delta) / 120.0
    return 0.0


def _normalized_gradient_score(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0 or min(arr.shape) <= 1:
        return np.zeros_like(arr, dtype=np.float32)
    grad_z, grad_y, grad_x = np.gradient(arr, edge_order=1)
    grad = np.sqrt(0.35 * grad_z * grad_z + grad_y * grad_y + grad_x * grad_x, dtype=np.float32)
    lo = float(np.percentile(grad, 55.0))
    hi = float(np.percentile(grad, 99.7))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((grad - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _normalized_local_contrast_score(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    padded = np.pad(arr.astype(np.float32), 1, mode="edge")
    local_mean = (
        padded[1:-1, 1:-1, 1:-1]
        + padded[:-2, 1:-1, 1:-1]
        + padded[2:, 1:-1, 1:-1]
        + padded[1:-1, :-2, 1:-1]
        + padded[1:-1, 2:, 1:-1]
        + padded[1:-1, 1:-1, :-2]
        + padded[1:-1, 1:-1, 2:]
    ) / 7.0
    contrast = np.abs(arr.astype(np.float32) - local_mean).astype(np.float32)
    lo = float(np.percentile(contrast, 55.0))
    hi = float(np.percentile(contrast, 99.6))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((contrast - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _surface_mask_zyx(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 3 or not mask.any():
        return np.asarray(mask, dtype=bool)
    interior = mask.copy()
    interior[1:-1, 1:-1, 1:-1] = (
        mask[1:-1, 1:-1, 1:-1]
        & mask[:-2, 1:-1, 1:-1]
        & mask[2:, 1:-1, 1:-1]
        & mask[1:-1, :-2, 1:-1]
        & mask[1:-1, 2:, 1:-1]
        & mask[1:-1, 1:-1, :-2]
        & mask[1:-1, 1:-1, 2:]
    )
    return mask & ~interior


def _select_indices_with_spatial_coverage(
    coords_zyx: np.ndarray,
    scores: np.ndarray,
    shape_zyx: tuple[int, int, int],
    max_points: int,
    *,
    preferred_bins: tuple[int, int, int],
    per_cell_quota: int,
) -> np.ndarray:
    if coords_zyx.shape[0] <= max_points:
        return np.arange(coords_zyx.shape[0], dtype=np.int32)
    shape = np.maximum(1, np.asarray(shape_zyx, dtype=np.int32))
    bins = np.maximum(1, np.minimum(np.asarray(preferred_bins, dtype=np.int32), shape))
    order = np.argsort(scores.astype(np.float32), kind="stable")[::-1]
    coords = coords_zyx[order]
    cell_z = np.minimum((coords[:, 0] * bins[0]) // shape[0], bins[0] - 1)
    cell_y = np.minimum((coords[:, 1] * bins[1]) // shape[1], bins[1] - 1)
    cell_x = np.minimum((coords[:, 2] * bins[2]) // shape[2], bins[2] - 1)
    num_cells = int(bins[0] * bins[1] * bins[2])
    cell_ids = (cell_z * bins[1] * bins[2] + cell_y * bins[2] + cell_x).astype(np.int32)
    counts = np.zeros((num_cells,), dtype=np.int16)
    kept_positions: list[int] = []
    used = np.zeros((order.shape[0],), dtype=bool)

    coarse_bins = np.maximum(1, np.minimum(np.maximum(1, bins // 2), shape))
    coarse_cell_z = np.minimum((coords[:, 0] * coarse_bins[0]) // shape[0], coarse_bins[0] - 1)
    coarse_cell_y = np.minimum((coords[:, 1] * coarse_bins[1]) // shape[1], coarse_bins[1] - 1)
    coarse_cell_x = np.minimum((coords[:, 2] * coarse_bins[2]) // shape[2], coarse_bins[2] - 1)
    coarse_num_cells = int(coarse_bins[0] * coarse_bins[1] * coarse_bins[2])
    coarse_cell_ids = (
        coarse_cell_z * coarse_bins[1] * coarse_bins[2] + coarse_cell_y * coarse_bins[2] + coarse_cell_x
    ).astype(np.int32)
    coarse_counts = np.zeros((coarse_num_cells,), dtype=np.int16)

    # First pass: make sure distant structures keep at least one representative point.
    for pos, cell_id in enumerate(coarse_cell_ids):
        if used[pos] or coarse_counts[cell_id] >= 1:
            continue
        kept_positions.append(pos)
        used[pos] = True
        coarse_counts[cell_id] += 1
        if len(kept_positions) >= max_points:
            return order[np.asarray(kept_positions, dtype=np.int32)]

    for quota in range(1, max(1, int(per_cell_quota)) + 1):
        for pos, cell_id in enumerate(cell_ids):
            if used[pos] or counts[cell_id] >= quota:
                continue
            kept_positions.append(pos)
            used[pos] = True
            counts[cell_id] += 1
            if len(kept_positions) >= max_points:
                return order[np.asarray(kept_positions, dtype=np.int32)]

    if len(kept_positions) < max_points:
        remaining = np.flatnonzero(~used)
        if remaining.size:
            fill = remaining[: max_points - len(kept_positions)]
            kept_positions.extend(fill.tolist())
    return order[np.asarray(kept_positions[:max_points], dtype=np.int32)]


def _coords_zyx_to_xyz(coords_zyx: np.ndarray, spacing_xyz: tuple[float, float, float]) -> np.ndarray:
    sx, sy, sz = (float(v) for v in spacing_xyz)
    points = np.empty((coords_zyx.shape[0], 3), dtype=np.float32)
    points[:, 0] = coords_zyx[:, 2] * sx
    points[:, 1] = coords_zyx[:, 1] * sy
    points[:, 2] = coords_zyx[:, 0] * sz
    return points


def _project_world_points(points_xyz: np.ndarray, projection: _ProjectionState) -> np.ndarray:
    if points_xyz.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    rotated = (points_xyz - projection.center_xyz) @ projection.rotation.T
    return ((rotated[:, :2] - projection.min_xy) / projection.span).astype(np.float32)


def _draw_box(painter: QPainter, box_norm_xy: np.ndarray, width: int, height: int) -> None:
    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    painter.setPen(QPen(QColor(188, 188, 182), 1.2))
    for start, end in edges:
        p0 = _map_norm_point(box_norm_xy[start], width, height)
        p1 = _map_norm_point(box_norm_xy[end], width, height)
        painter.drawLine(p0, p1)


def _draw_raw_points(
    painter: QPainter,
    raw_norm_xy: np.ndarray,
    intensity: np.ndarray,
    width: int,
    height: int,
) -> None:
    if raw_norm_xy.size == 0 or intensity.size == 0:
        return
    bins = np.clip(np.floor(intensity * 6.0).astype(np.int32), 0, 5)
    for bucket in range(6):
        idx = np.flatnonzero(bins == bucket)
        if idx.size == 0:
            continue
        alpha = 42 + bucket * 32
        shade = 66 + bucket * 30
        pen_width = 1.8 if bucket < 3 else 2.2
        painter.setPen(QPen(QColor(shade, shade + 4, shade + 8, min(alpha, 228)), pen_width))
        painter.drawPoints(QPolygonF([_map_norm_point(raw_norm_xy[i], width, height) for i in idx]))
    hot_idx = np.flatnonzero(intensity >= 0.82)
    if hot_idx.size:
        painter.setPen(QPen(QColor(244, 242, 236, 210), 1.4))
        painter.drawPoints(QPolygonF([_map_norm_point(raw_norm_xy[i], width, height) for i in hot_idx]))


def _draw_mask_points(painter: QPainter, mask_norm_xy: np.ndarray, width: int, height: int) -> None:
    if mask_norm_xy.size == 0:
        return
    painter.setPen(QPen(QColor(240, 198, 78, 55), 3.6))
    painter.drawPoints(QPolygonF([_map_norm_point(point, width, height) for point in mask_norm_xy]))
    painter.setPen(QPen(QColor(214, 74, 62, 178), 2.2))
    painter.drawPoints(QPolygonF([_map_norm_point(point, width, height) for point in mask_norm_xy]))


def _map_norm_point(point_xy: np.ndarray, width: int, height: int) -> QPointF:
    return QPointF(float(point_xy[0]) * width, float(point_xy[1]) * height)


def _map_norm_point_to_rect(point_xy: np.ndarray, rect: QRectF) -> QPointF:
    return QPointF(rect.left() + float(point_xy[0]) * rect.width(), rect.top() + float(point_xy[1]) * rect.height())


def _map_norm_points_to_rect(points_xy: np.ndarray, rect: QRectF) -> list[QPointF]:
    return [_map_norm_point_to_rect(point, rect) for point in points_xy]


def _normalize_view_mode(mode: str | None) -> str:
    text = (mode or "oblique").strip().lower()
    if text in {"xy", "xz", "yz", "oblique"}:
        return text
    return "oblique"


def _view_mode_label(mode: str) -> str:
    normalized = _normalize_view_mode(mode)
    for key, label in VIEW_MODE_LABELS:
        if key == normalized:
            return label
    return "Oblique"


def _review_badge_palette(text: str, done: bool) -> tuple[QColor, QColor]:
    key = (text or "").strip().upper()
    if key == "A":
        bg = QColor(44, 128, 92, 220)
    elif key == "B":
        bg = QColor(198, 130, 42, 220)
    elif key == "C":
        bg = QColor(184, 72, 58, 220)
    else:
        bg = QColor(86, 96, 108, 205)
    if done:
        bg = QColor(
            min(bg.red() + 8, 255),
            min(bg.green() + 8, 255),
            min(bg.blue() + 8, 255),
            230,
        )
    return bg, QColor(250, 248, 244)
