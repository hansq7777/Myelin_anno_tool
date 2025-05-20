from PyQt5.QtWidgets import QMainWindow, QFileDialog, QSlider, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt
from ..models.zstack_model import ZStackModel
from ..views.canvas import SliceCanvas

class MainController(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Z-Stack Annotation (alpha)")
        self.model = ZStackModel()
        self.canvas = SliceCanvas()
        self._build_layout()
        self._create_menu()
        self.statusBar().showMessage("Ready")

    def _build_layout(self):
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        # -------- Slider --------
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slice_changed)
        layout.addWidget(self.slider)
        self.setCentralWidget(central)

    def _create_menu(self):
        file_menu = self.menuBar().addMenu("File")
        open_act = file_menu.addAction("Open…")
        open_act.triggered.connect(self._open_file)

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open TIFF", "", "TIFF Images (*.tif *.tiff *.ome.tif)")
        if path:
            self.model.load(path)
            self.slider.setRange(0, self.model.n_slices - 1)
            self.slider.setEnabled(True)
            self._update_view()

    def _on_slice_changed(self, idx: int):
        if self.model.data is None:
            return
        self.model.index = idx
        self._update_view()

    def _update_view(self):
        self.canvas.set_image(self.model.get_current())
        self.statusBar().showMessage(
            f"Slice {self.model.index + 1} / {self.model.n_slices}")

    # 键盘快捷
    def keyPressEvent(self, event):
        if not self.slider.isEnabled():
            return
        if event.key() in (Qt.Key_Up, Qt.Key_W):
            self.slider.setValue(max(0, self.slider.value() - 1))
        elif event.key() in (Qt.Key_Down, Qt.Key_S):
            self.slider.setValue(min(self.model.n_slices - 1, self.slider.value() + 1))