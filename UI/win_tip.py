from PyQt6.QtWidgets import QWidget, QLabel


class Win_Tip(QWidget):
    def __init__(self):
        super().__init__()
        QLabel("Future Work", self)  # PS.当前不要修改默认路径，我没写这个功能
        self.setStyleSheet("background-color:lightblue;")
