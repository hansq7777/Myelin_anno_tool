import os
import sys
from pathlib import Path


def _configure_qt_runtime() -> None:
    """Prefer PyQt's plugin directory and XCB on Linux for VTK child windows."""
    try:
        import PyQt5  # type: ignore
    except Exception:
        return

    plugin_dir = Path(PyQt5.__file__).resolve().parent / "Qt5" / "plugins"
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(plugin_dir))
    os.environ.setdefault("QT_PLUGIN_PATH", str(plugin_dir))

    if sys.platform.startswith("linux") and os.environ.get("DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")


def main() -> None:
    _configure_qt_runtime()
    from PyQt5.QtWidgets import QApplication
    from zstack_anno.controllers.main_controller import MainController

    app = QApplication(sys.argv)
    window = MainController()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
