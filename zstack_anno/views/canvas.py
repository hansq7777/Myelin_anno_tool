from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene
from PyQt5.QtGui import QPixmap, QImage, QTransform
from PyQt5.QtCore import Qt, pyqtSignal
import numpy as np

_ZOOM_BASE = 1.2
_MIN_ZOOM = 0.2
_MAX_ZOOM = 24.0


class SliceCanvas(QGraphicsView):
    """负责把 2-D ndarray 显示为灰度图，可拖拽缩放，并支持掩膜叠加。"""
    zoomAdjusted = pyqtSignal(float)

    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene())
        # 启用拖拽平移
        self.setDragMode(self.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        # 启用鼠标移动追踪以显示像素信息
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

        self._image_item = None
        self._mask_item = None
        self._zoom = 1.0
        self._mask_opacity = 0.5  # default 50%

    def set_image(self, arr: np.ndarray, reset_view: bool = False) -> None:
        """显示图像，支持灰度或 RGB."""
        if arr.ndim == 2:
            h, w = arr.shape
            arr = np.ascontiguousarray(arr)
            img = QImage(arr.data, w, h, arr.strides[0], QImage.Format_Grayscale8)
        elif arr.ndim == 3 and arr.shape[2] == 3:
            h, w, _ = arr.shape
            arr = np.ascontiguousarray(arr)
            img = QImage(arr.data, w, h, arr.strides[0], QImage.Format_RGB888)
        else:
            raise ValueError("只支持 2-D 灰度图或 3-D RGB 图像")
        pix = QPixmap.fromImage(img)

        if self._image_item is None:
            self._image_item = self.scene().addPixmap(pix)
        else:
            self._image_item.setPixmap(pix)

        self.setSceneRect(0, 0, w, h)

        if self._mask_item:
            self._mask_item.setZValue(1)

        if reset_view:
            self.resetTransform()
            self._zoom = 1.0
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def zoom_factor(self) -> float:
        return float(self._zoom)

    def apply_zoom_factor(self, factor: float, *, emit_signal: bool = False) -> float:
        factor = float(factor)
        if factor <= 0.0:
            return float(self._zoom)
        new_zoom = float(np.clip(self._zoom * factor, _MIN_ZOOM, _MAX_ZOOM))
        applied = new_zoom / max(self._zoom, 1e-9)
        if abs(applied - 1.0) < 1e-6:
            return float(self._zoom)
        self._zoom = new_zoom
        self.scale(applied, applied)
        if emit_signal:
            self.zoomAdjusted.emit(applied)
        return float(self._zoom)

    def wheelEvent(self, event):
        angle = event.angleDelta().y()
        if angle == 0:
            return
        steps = float(angle) / 120.0
        factor = _ZOOM_BASE ** steps
        self.apply_zoom_factor(factor, emit_signal=True)
        event.accept()

    def set_mask_opacity(self, opacity: float) -> None:
        """Set mask opacity as a fraction from 0 to 1."""
        opacity = max(0.0, min(1.0, opacity))
        self._mask_opacity = opacity
        if self._mask_item is not None:
            # Reapply current mask to update opacity
            img = self._mask_item.pixmap().toImage()
            if img is not None:
                w = img.width()
                h = img.height()
                buf = img.bits()
                buf.setsize(img.byteCount())
                arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
                arr[..., 3] = (arr[..., 3] > 0).astype(np.uint8) * int(255 * opacity)
                new_img = QImage(arr.data, w, h, QImage.Format_RGBA8888)
                self._mask_item.setPixmap(QPixmap.fromImage(new_img))

    def set_mask(self, mask: np.ndarray | None) -> None:
        """设置掩膜叠加，None 表示移除。"""
        if mask is None:
            if self._mask_item:
                self.scene().removeItem(self._mask_item)
                self._mask_item = None
            return

        if mask.ndim != 2:
            raise ValueError("掩膜必须是 2-D 数组")

        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., 0] = 255  # 红色
        alpha = int(255 * self._mask_opacity)
        rgba[..., 3] = (mask > 0).astype(np.uint8) * alpha
        rgba = np.ascontiguousarray(rgba)
        img = QImage(rgba.data, w, h, rgba.strides[0], QImage.Format_RGBA8888)
        pix = QPixmap.fromImage(img)

        if self._mask_item is None:
            self._mask_item = self.scene().addPixmap(pix)
            self._mask_item.setZValue(1)
        else:
            self._mask_item.setPixmap(pix)


class SyncCanvas(SliceCanvas):
    """Canvas that syncs panning and zooming with peers."""

    viewChanged = pyqtSignal(QTransform, int, int)

    def __init__(self) -> None:
        super().__init__()
        self._ignore = False
        self.horizontalScrollBar().valueChanged.connect(self._emit_transform)
        self.verticalScrollBar().valueChanged.connect(self._emit_transform)

    def wheelEvent(self, event):
        super().wheelEvent(event)
        self._emit_transform()

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if event.button() == Qt.LeftButton:
            self._emit_transform()

    def _emit_transform(self):
        if not self._ignore:
            self.viewChanged.emit(
                self.transform(),
                self.horizontalScrollBar().value(),
                self.verticalScrollBar().value(),
            )

    def apply_transform(self, tr: QTransform, h: int, v: int) -> None:
        self._ignore = True
        self.setTransform(tr)
        self.horizontalScrollBar().setValue(h)
        self.verticalScrollBar().setValue(v)
        self._ignore = False
