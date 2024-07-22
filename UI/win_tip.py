from PyQt6.QtWidgets import QWidget, QLabel


class Win_Tip(QWidget):
    def __init__(self):
        super().__init__()
        QLabel("懒得写\n", self)
        self.setStyleSheet("background-color:lightblue;")
