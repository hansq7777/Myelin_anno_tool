from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import numpy as np


class SliceCanvas(QGraphicsView):
    """负责把 2-D ndarray 显示为灰度图，可拖拽缩放，并支持掩膜叠加。"""

    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene())
        # 启用拖拽平移
        self.setDragMode(self.ScrollHandDrag)

        self._image_item = None
        self._mask_item = None
        self._zoom = 1.0
        self._mask_opacity = 0.5  # default 50%

    def set_image(self, arr: np.ndarray, reset_view: bool = False) -> None:
        """显示灰度图像。"""
        if arr.ndim != 2:
            raise ValueError("只支持 2-D 灰度图")

        h, w = arr.shape
        img = QImage(arr.data, w, h, QImage.Format_Grayscale8)
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

    def wheelEvent(self, event):
        angle = event.angleDelta().y()
        if angle == 0:
            return
        # Use a smaller scaling factor for smoother zooming
        factor = 1.02 if angle > 0 else 0.98
        self._zoom *= factor
        self.scale(factor, factor)

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
                arr = np.frombuffer(img.bits(), dtype=np.uint8).reshape(h, w, 4)
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
        img = QImage(rgba.data, w, h, QImage.Format_RGBA8888)
        pix = QPixmap.fromImage(img)

        if self._mask_item is None:
            self._mask_item = self.scene().addPixmap(pix)
            self._mask_item.setZValue(1)
        else:
            self._mask_item.setPixmap(pix)

