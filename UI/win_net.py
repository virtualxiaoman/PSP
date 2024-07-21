from PyQt6.QtWidgets import QWidget, QLabel


class Win_Net(QWidget):
    def __init__(self):
        super().__init__()
        QLabel("未来预计对接各大网络搜索网站\n", self)
        self.setStyleSheet("background-color:lightblue;")
