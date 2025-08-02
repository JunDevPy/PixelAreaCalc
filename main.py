"""
Входная точка приложения
"""

import sys
from PySide6.QtWidgets import QApplication

from src.gui_components import MainWindow


def main():
    app = QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
