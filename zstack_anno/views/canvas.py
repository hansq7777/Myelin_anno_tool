from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import numpy as np


class SliceCanvas(QGraphicsView):
    """负责把 2-D ndarray 显示为灰度图，可拖拽缩放。"""

    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene())
        # 启用拖拽平移
        self.setDragMode(self.ScrollHandDrag)

    def set_image(self, arr: np.ndarray) -> None:
        if arr.ndim != 2:
            raise ValueError("只支持 2-D 灰度图")
        h, w = arr.shape
        # ndarray → Qt 图像对象（8-bit 灰度）
        img = QImage(arr.data, w, h, QImage.Format_Grayscale8)
        pix = QPixmap.fromImage(img)
        self.scene().clear()
        self.scene().addPixmap(pix)
        self.setSceneRect(0, 0, w, h)
        # 自适应窗口大小
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
