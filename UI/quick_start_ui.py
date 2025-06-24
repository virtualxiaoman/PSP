import sys
from PyQt6.QtWidgets import QApplication
from UI.ui import PSP_UI

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = PSP_UI()
    ui.show()
    sys.exit(app.exec())
