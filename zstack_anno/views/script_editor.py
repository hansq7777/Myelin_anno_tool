import json
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QVBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QFileDialog,
    QInputDialog,
    QWidget,
    QLabel,
    QLineEdit,
    QMessageBox,
)
from PyQt5.QtCore import Qt


class StepListWidget(QListWidget):
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        # allow both internal move and external drops
        self.setDragDropMode(QListWidget.DragDrop)

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.source() is not self and event.mimeData().hasText():
            action = event.mimeData().text()
            self.parent().add_step(action)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


class StepWidget(QWidget):
    """Widget used to edit parameters for a script step."""

    def __init__(self, item: QListWidgetItem, action: str, params: dict):
        super().__init__()
        self._item = item
        self._action = action
        self._defaults = params.copy()
        self._inputs: dict[str, QLineEdit] = {}

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(QLabel(action))
        for key, value in params.items():
            edit = QLineEdit()
            if value is not None:
                edit.setText(str(value))
            edit.editingFinished.connect(self._update_item)
            layout.addWidget(edit)
            self._inputs[key] = edit
        layout.addStretch(1)
        self._update_item()

    def _update_item(self) -> None:
        params: dict[str, object] = {}
        for key, edit in self._inputs.items():
            text = edit.text().strip()
            default = self._defaults.get(key)
            if text == "":
                params[key] = default
                continue
            try:
                if isinstance(default, int):
                    params[key] = int(text)
                else:
                    params[key] = float(text)
            except ValueError:
                params[key] = default
        data = {"action": self._action, "params": params}
        self._item.setData(Qt.UserRole, data)

    def set_params(self, params: dict) -> None:
        for key, value in params.items():
            if key in self._inputs:
                self._inputs[key].setText("" if value is None else str(value))
        self._update_item()


class ScriptEditor(QDialog):
    ACTIONS = {
        "Dilate": {"method": "script_dilate", "params": {"iterations": 1}},
        "Erode": {"method": "script_erode", "params": {"iterations": 1}},
        "Filter Small": {"method": "script_filter_small", "params": {"threshold": 100}},
        "Seed": {"method": "script_seed", "params": {"percentile": 90.0}},
        "Intensity Grow": {
            "method": "script_int_grow",
            "params": {"diff_pct": 20.0, "hist_pct": None},
        },
        "Background Filter": {
            "method": "script_bg_filter",
            "params": {"percentile": 0.0},
        },
        "Gaussian Blur": {"method": "script_blur", "params": {"sigma": 1.0}},
        "Clear Blur": {"method": "script_clear_blur", "params": {}},
    }

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setWindowTitle("Script Editor")

        layout = QHBoxLayout(self)
        self.step_list = StepListWidget()
        self.action_list = QListWidget()
        self.action_list.setDragEnabled(True)

        for act in self.ACTIONS.keys():
            item = QListWidgetItem(act)
            self.action_list.addItem(item)

        layout.addWidget(self.step_list)
        layout.addWidget(self.action_list)

        btn_layout = QVBoxLayout()
        self.add_btn = QPushButton("Add")
        self.remove_btn = QPushButton("Remove")
        self.run_btn = QPushButton("Run")
        self.save_btn = QPushButton("Save")
        self.load_btn = QPushButton("Load")
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.remove_btn)
        btn_layout.addWidget(self.run_btn)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.load_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.run_btn.clicked.connect(self.run_script)
        self.add_btn.clicked.connect(self.add_selected_action)
        self.remove_btn.clicked.connect(self.remove_selected_step)
        self.save_btn.clicked.connect(self.save_script)
        self.load_btn.clicked.connect(self.load_script)
        self.step_list.itemDoubleClicked.connect(self.edit_step)

    def add_step(self, action: str, params: dict | None = None) -> QListWidgetItem:
        data = self.get_default_step(action)
        if params:
            data["params"].update(params)
        item = QListWidgetItem()
        widget = StepWidget(item, action, data.get("params", {}))
        if params:
            widget.set_params(data["params"])
        item.setText("")
        item.setSizeHint(widget.sizeHint())
        self.step_list.addItem(item)
        self.step_list.setItemWidget(item, widget)
        return item

    def get_default_step(self, action: str) -> dict:
        info = self.ACTIONS.get(action, {})
        return {"action": action, "params": info.get("params", {}).copy()}

    def format_step(self, step: dict) -> str:
        params = ", ".join(f"{k}={v}" for k, v in step.get("params", {}).items())
        return f"{step['action']} ({params})" if params else step["action"]

    def edit_step(self, item: QListWidgetItem) -> None:
        data = item.data(Qt.UserRole)
        if data is None:
            QMessageBox.warning(self, "Error", "No step data")
            return
        params = data.get("params", {})
        for key, value in params.items():
            text, ok = QInputDialog.getText(self, data["action"], key, text=str(value))
            if ok:
                try:
                    if isinstance(value, int):
                        params[key] = int(text)
                    else:
                        params[key] = float(text) if text else None
                except ValueError:
                    pass
        data["params"] = params
        item.setData(Qt.UserRole, data)
        item.setText(self.format_step(data))

    def run_script(self) -> None:
        if self.controller.model.data is None:
            QMessageBox.warning(self, "No Image", "Please load an image first")
            return
        for idx in range(self.step_list.count()):
            data = self.step_list.item(idx).data(Qt.UserRole)
            action = data["action"]
            info = self.ACTIONS.get(action)
            if not info:
                continue
            method = getattr(self.controller, info["method"], None)
            if method:
                method(**data.get("params", {}))

    def save_script(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Script", "", "JSON Files (*.json)")
        if not path:
            return
        steps = [self.step_list.item(i).data(Qt.UserRole) for i in range(self.step_list.count())]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(steps, f, indent=2)

    def load_script(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Script", "", "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                steps = json.load(f)
        except Exception:
            return
        self.step_list.clear()
        for step in steps:
            self.add_step(step["action"], step.get("params", {}))

    def add_selected_action(self) -> None:
        """Add the currently selected action from the action list."""
        item = self.action_list.currentItem()
        if not item:
            return
        action = item.text()
        self.add_step(action)

    def remove_selected_step(self) -> None:
        """Remove the currently selected step from the sequence."""
        row = self.step_list.currentRow()
        if row >= 0:
            self.step_list.takeItem(row)

