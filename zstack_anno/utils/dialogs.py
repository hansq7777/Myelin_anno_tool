from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence


def question_with_shortcuts(parent, title: str, text: str) -> QMessageBox.StandardButton:
    """Show a yes/no/cancel question dialog with Y/N shortcuts."""
    box = QMessageBox(QMessageBox.Question, title, text, parent=parent)
    yes_btn = box.addButton(QMessageBox.Yes)
    no_btn = box.addButton(QMessageBox.No)
    cancel_btn = box.addButton(QMessageBox.Cancel)
    yes_btn.setShortcut(QKeySequence(Qt.Key_Y))
    no_btn.setShortcut(QKeySequence(Qt.Key_N))
    box.setDefaultButton(yes_btn)
    ret = box.exec()
    return ret

