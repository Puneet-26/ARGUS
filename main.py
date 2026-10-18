"""Native desktop entry point for ARGUS."""

import sys
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DESKTOP_APP = PROJECT_ROOT / "argus_app"
os.chdir(PROJECT_ROOT)
if str(DESKTOP_APP) not in sys.path:
    sys.path.insert(0, str(DESKTOP_APP))


def main():
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication
    from ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("ARGUS: The Last Chai")
    app.setFont(QFont("Segoe UI", 10))
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
