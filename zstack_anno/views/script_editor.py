import json
import os
import time
from ..utils import config
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
    QApplication,
)
from PyQt5.QtGui import QDrag
from PyQt5.QtCore import Qt, QMimeData


class StepListWidget(QListWidget):
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        # allow both internal move and external drops
        self.setDragDropMode(QListWidget.DragDrop)

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.setDropAction(Qt.CopyAction)
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasText():
            event.setDropAction(Qt.CopyAction)
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.source() is not self and event.mimeData().hasText():
            action = event.mimeData().text()
            self.parent().add_step(action, prompt=True)
            event.setDropAction(Qt.CopyAction)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


class ActionListWidget(QListWidget):
    """List widget that starts drags with plain text."""

    def startDrag(self, supported_actions):
        item = self.currentItem()
        if item is None:
            return
        drag = QDrag(self)
        mime = QMimeData()
        mime.setText(item.text())
        drag.setMimeData(mime)
        drag.exec_(Qt.CopyAction)


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
            layout.addWidget(QLabel(f"{key}:"))
            edit = QLineEdit()
            if value is not None:
                edit.setText(str(value))
            else:
                edit.setPlaceholderText(str(key))
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
        "Next Slice": {"method": "script_next_slice", "params": {}},
        "Previous Slice": {"method": "script_prev_slice", "params": {}},
        "Dilate": {"method": "script_dilate", "params": {"iterations": 1}},
        "Erode": {"method": "script_erode", "params": {"iterations": 1}},
        "Filter Small": {"method": "script_filter_small", "params": {"threshold": 100}},
        "Seed": {"method": "script_seed", "params": {"percentile": 85.0}},
        "Intensity Grow": {
            "method": "script_int_grow",
            "params": {"diff_pct": 50.0, "hist_pct": 20.0, "force_pct": None, "limit": 30000},
        },
        "Flood Grow": {
            "method": "script_flood_grow",
            "params": {"connectivity": 1, "tolerance": 5.0},
        },
        "Background Filter": {
            "method": "script_bg_filter",
            "params": {"percentile": 10.0, "bins": 0},
        },
        "Gaussian Blur": {"method": "script_blur", "params": {"sigma": 1.0}},
        "Clear Blur": {"method": "script_clear_blur", "params": {}},
        "Save": {"method": "script_save", "params": {}},
    }

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setWindowTitle("Script Editor")
        self._paused = False
        self._stopped = False

        layout = QHBoxLayout(self)
        self.step_list = StepListWidget()
        self.action_list = ActionListWidget()
        self.action_list.setDragEnabled(True)

        for act in self.ACTIONS.keys():
            item = QListWidgetItem(act)
            self.action_list.addItem(item)

        layout.addWidget(self.step_list)
        layout.addWidget(self.action_list)

        btn_layout = QVBoxLayout()
        self.run_btn = QPushButton("Run")
        self.run_stack_btn = QPushButton("Run Stack(s)")
        self.stop_btn = QPushButton("Stop")
        self.pause_btn = QPushButton("Pause")
        self.resume_btn = QPushButton("Resume")
        self.add_btn = QPushButton("Add")
        self.remove_btn = QPushButton("Remove")
        self.save_btn = QPushButton("Save")
        self.load_btn = QPushButton("Load")
        self.run_btn.setStyleSheet("background-color: #007bff; color: white")
        btn_layout.addWidget(self.run_btn)
        btn_layout.addWidget(self.run_stack_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.pause_btn)
        btn_layout.addWidget(self.resume_btn)
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.remove_btn)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.load_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.run_btn.clicked.connect(self.run_script)
        self.run_stack_btn.clicked.connect(self.run_stack)
        self.stop_btn.clicked.connect(self.stop_script)
        self.pause_btn.clicked.connect(self.pause_script)
        self.resume_btn.clicked.connect(self.resume_script)
        self.add_btn.clicked.connect(self.add_selected_action)
        self.remove_btn.clicked.connect(self.remove_selected_step)
        self.save_btn.clicked.connect(self.save_script)
        self.load_btn.clicked.connect(self.load_script)
        self.step_list.itemDoubleClicked.connect(self.edit_step)
        self.action_list.itemDoubleClicked.connect(self.add_action_item)

        stored = config.get("script", [])
        if isinstance(stored, list):
            self.set_script(stored)

    def add_step(
        self, action: str, params: dict | None = None, prompt: bool = False
    ) -> QListWidgetItem:
        data = self.get_default_step(action)
        if params:
            data["params"].update(params)
        if prompt:
            for key, value in data["params"].items():
                text, ok = QInputDialog.getText(self, action, key, text=str(value))
                if ok:
                    try:
                        if isinstance(value, int):
                            data["params"][key] = int(text)
                        else:
                            data["params"][key] = float(text)
                    except ValueError:
                        pass
        item = QListWidgetItem()
        widget = StepWidget(item, action, data.get("params", {}))
        item.setText("")
        item.setSizeHint(widget.sizeHint())
        self.step_list.addItem(item)
        self.step_list.setItemWidget(item, widget)
        self._save_to_config()
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
        self._save_to_config()

    def run_script(self) -> float:
        """Run the configured steps once and return the elapsed time."""
        if self.controller.model.data is None:
            QMessageBox.warning(self, "No Image", "Please load an image first")
            return 0.0
        start_time = time.monotonic()
        self._paused = False
        self._stopped = False
        for idx in range(self.step_list.count()):
            if self._stopped:
                break
            item = self.step_list.item(idx)
            data = item.data(Qt.UserRole)
            if data is None:
                QMessageBox.warning(self, "Error", f"Step {idx + 1} has no data")
                return
            action = data.get("action")
            if not action:
                QMessageBox.warning(self, "Error", f"Step {idx + 1} missing action")
                return
            info = self.ACTIONS.get(action)
            if not info:
                continue
            params = data.get("params", {})
            for key, value in params.items():
                if value is None:
                    QMessageBox.warning(
                        self,
                        "Error",
                        f"Parameter '{key}' for action '{action}' is missing",
                    )
                    return
            method = getattr(self.controller, info["method"], None)
            if method:
                prev_index = self.controller.model.index
                step_start = time.monotonic()
                method(**params)
                step_time = time.monotonic() - step_start
                self.controller.report_action(action, params)
                print(f"Time for {action}: {step_time:.3f}s")
                QApplication.processEvents()
                if (
                    action == "Next Slice"
                    and self.controller.model.index == prev_index
                ):
                    self._stopped = True
                while self._paused and not self._stopped:
                    QApplication.processEvents()
                    time.sleep(0.1)
                if self._stopped:
                    break
        return time.monotonic() - start_time

    def save_script(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Script", "", "JSON Files (*.json)"
        )
        if not path:
            return
        steps = [
            self.step_list.item(i).data(Qt.UserRole)
            for i in range(self.step_list.count())
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(steps, f, indent=2)
        self._save_to_config()

    def load_script(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Script", "", "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                steps = json.load(f)
        except Exception:
            return
        self.step_list.clear()
        for step in steps:
            if not isinstance(step, dict):
                continue
            action = step.get("action")
            if not action:
                continue
            self.add_step(action, step.get("params", {}))
        self._save_to_config()

    def add_selected_action(self) -> None:
        """Add the currently selected action from the action list."""
        item = self.action_list.currentItem()
        if not item:
            return
        action = item.text()
        self.add_step(action, prompt=True)

    def add_action_item(self, item: QListWidgetItem) -> None:
        """Add action by double-clicking in the action list."""
        if not item:
            return
        action = item.text()
        self.add_step(action, prompt=True)

    def remove_selected_step(self) -> None:
        """Remove the currently selected step from the sequence."""
        row = self.step_list.currentRow()
        if row >= 0:
            self.step_list.takeItem(row)
            self._save_to_config()

    def stop_script(self) -> None:
        """Stop execution after the current action."""
        self._stopped = True
        self._paused = False

    def pause_script(self) -> None:
        """Pause execution after the current action."""
        self._paused = True

    def resume_script(self) -> None:
        """Resume execution if paused."""
        self._paused = False

    def run_stack(self) -> None:
        """Entry point for running the script on one or more stacks."""
        mode = QMessageBox(self)
        mode.setWindowTitle("Run Stack(s)")
        mode.setText("Run on current stack or multiple stacks?")
        current_btn = mode.addButton("Current Stack", QMessageBox.AcceptRole)
        multi_btn = mode.addButton("Multiple Stacks…", QMessageBox.AcceptRole)
        cancel_btn = mode.addButton(QMessageBox.Cancel)
        mode.exec()
        clicked = mode.clickedButton()
        if clicked is cancel_btn:
            return
        if clicked is multi_btn:
            self._run_multiple_stacks()
        else:
            self._run_single_stack()

    def _run_stack_range(self, start_idx: int, end_idx: int) -> None:
        """Run the script from ``start_idx`` to ``end_idx`` inclusive."""
        self.controller.slider.setValue(start_idx)
        while True:
            slice_start = self.controller.model.index
            elapsed = self.run_script()
            self.controller.script_save()
            print(
                f"Slice {slice_start + 1}/{self.controller.model.n_slices} finished in {elapsed:.3f}s"
            )
            if self.controller.model.index >= end_idx or self._stopped:
                break
            self.controller.script_next_slice()
            QApplication.processEvents()

    def _run_single_stack(self) -> None:
        """Run the script on the currently loaded stack."""
        if self.controller.model.data is None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Stack",
                "",
                "TIFF Images (*.tif *.tiff *.ome.tif)",
            )
            if not path:
                return
            self.controller.model.load(path)
            self.controller.slider.setRange(
                0, self.controller.model.n_slices - 1
            )
            # enable navigation after loading via the script editor
            self.controller.slider.setEnabled(True)
            self.controller._update_view(reset_view=True)

        msg = QMessageBox(self)
        msg.setWindowTitle("Run Stack")
        msg.setText("Select run mode")
        current_btn = msg.addButton("Current to End", QMessageBox.AcceptRole)
        range_btn = msg.addButton("Slice Range…", QMessageBox.AcceptRole)
        cancel_btn = msg.addButton(QMessageBox.Cancel)
        msg.exec()
        clicked = msg.clickedButton()
        if clicked is cancel_btn:
            return
        if clicked is range_btn:
            total = self.controller.model.n_slices
            start, ok = QInputDialog.getInt(
                self,
                "Start Slice",
                "Start slice (1-{}):".format(total),
                self.controller.model.index + 1,
                1,
                total,
            )
            if not ok:
                return
            end, ok = QInputDialog.getInt(
                self,
                "End Slice",
                "End slice ({}-{}):".format(start, total),
                total,
                start,
                total,
            )
            if not ok:
                return
            start_idx = start - 1
            end_idx = end - 1
        else:
            start_idx = self.controller.model.index
            end_idx = self.controller.model.n_slices - 1

        self._run_stack_range(start_idx, end_idx)
        QMessageBox.information(self, "Run Stack", "Stack segmentation complete")

    def _run_multiple_stacks(self) -> None:
        """Run the script on a batch of stacks selected by the user."""
        choice = QMessageBox(self)
        choice.setWindowTitle("Select Inputs")
        choice.setText("Choose stacks from a folder or select files")
        folder_btn = choice.addButton("Folder…", QMessageBox.AcceptRole)
        files_btn = choice.addButton("Files…", QMessageBox.AcceptRole)
        cancel_btn = choice.addButton(QMessageBox.Cancel)
        choice.exec()
        clicked = choice.clickedButton()
        if clicked is cancel_btn:
            return
        if clicked is folder_btn:
            folder = QFileDialog.getExistingDirectory(self, "Select Folder")
            if not folder:
                return
            file_list = sorted(
                [
                    os.path.join(folder, f)
                    for f in os.listdir(folder)
                    if f.lower().endswith((".tif", ".tiff", ".ome.tif"))
                ]
            )
        else:
            file_list, _ = QFileDialog.getOpenFileNames(
                self,
                "Select Stack Files",
                "",
                "TIFF Images (*.tif *.tiff *.ome.tif)",
            )
            if not file_list:
                return

        start_slice, ok = QInputDialog.getInt(
            self,
            "Start Slice",
            "Start processing from which slice?",
            1,
            1,
        )
        if not ok:
            return
        start_idx = start_slice - 1

        save_folder = QFileDialog.getExistingDirectory(self, "Select Save Folder")
        if not save_folder:
            return

        for path in file_list:
            self.controller.model.load(path)
            self.controller.slider.setRange(
                0, self.controller.model.n_slices - 1
            )
            # ensure navigation works when processing multiple stacks
            self.controller.slider.setEnabled(True)
            self.controller._update_view(reset_view=True)
            mask_name = os.path.splitext(os.path.basename(path))[0] + "_mask.tif"
            mask_path = os.path.join(save_folder, mask_name)
            self.controller.model.create_blank_masks(mask_path)
            start = min(start_idx, self.controller.model.n_slices - 1)
            self._run_stack_range(start, self.controller.model.n_slices - 1)
            if self._stopped:
                break
        QMessageBox.information(self, "Run Stacks", "Batch segmentation complete")

    # -------- persistence helpers ---------
    def get_script(self) -> list[dict]:
        steps = []
        for i in range(self.step_list.count()):
            data = self.step_list.item(i).data(Qt.UserRole)
            if isinstance(data, dict):
                steps.append(data)
        return steps

    def set_script(self, steps: list[dict]) -> None:
        self.step_list.clear()
        for step in steps:
            action = step.get("action")
            if not action:
                continue
            self.add_step(action, step.get("params", {}))

    def _save_to_config(self) -> None:
        config.set("script", self.get_script())

    def closeEvent(self, event) -> None:
        self._save_to_config()
        super().closeEvent(event)
