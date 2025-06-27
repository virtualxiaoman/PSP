import sys
from PyQt6.QtWidgets import QApplication, QWidget, QMainWindow, QHBoxLayout, QVBoxLayout, QPushButton, QStackedLayout

from UI.win_local import Win_Local
from UI.win_net import Win_Net
from UI.win_tip import Win_Tip

from UI.config import Text as Text_config
from UI.config import Background_css, Button_css
Text_config = Text_config()
Background_css = Background_css()
Button_css = Button_css()


class PSP_UI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.Layout_stack = None  # 抽屉布局器，存放的是切换显示的Widget

        self.create_stacked_layout()  # 堆叠布局
        self.init_ui()  # 初始界面

    def create_stacked_layout(self):
        """ 创建堆叠布局(抽屉布局) """
        self.Layout_stack = QStackedLayout()
        # 创建单独的Widget
        win1 = Win_Local()
        win2 = Win_Net()
        win3 = Win_Tip()
        # 将创建的2个Widget添加到抽屉布局器中
        self.Layout_stack.addWidget(win1)
        self.Layout_stack.addWidget(win2)
        self.Layout_stack.addWidget(win3)

    def init_ui(self):
        # 设置窗口的基础属性
        self.resize(1000, 700)
        self.setWindowTitle(Text_config.WindowTitle)
        self.setStyleSheet(Background_css.WHITE)

        # 窗口的中心部件，用来放置其他控件
        self.central_widget = QWidget()
        self.central_widget.setStyleSheet(Background_css.WHITE)
        self.setCentralWidget(self.central_widget)  # 将self.central_widget设置为窗口的中心部件。使得窗口内容将显示在其上
        # 创建整体的水平布局器
        H_Layout_main = self._init_H_Layout_main()
        # 设置当前要显示的Widget，从而能够显示这个布局器中的控件，类似于QWidget里的self.setLayout(container)
        self.centralWidget().setLayout(H_Layout_main)

    def _init_H_Layout_main(self):
        # 1. 创建整体的水平布局器
        H_Layout_main = QHBoxLayout()
        # 2. 创建一个要显示具体内容的子Widget
        Widget_stack = self.__init_Widget_stack()
        # 3. 创建2个按钮，用来点击进行切换抽屉布局器中的widget
        Widget_btn = self.__init_Widget_btn()
        # 4. 将widget与btn添加到布局器中
        H_Layout_main.addWidget(Widget_btn)
        H_Layout_main.addWidget(Widget_stack)

        return H_Layout_main

    def __init_Widget_stack(self):
        Widget_stack = QWidget()
        Widget_stack.setLayout(self.Layout_stack)  # 设置为之前定义的抽屉布局
        Widget_stack.setStyleSheet(Background_css.WHITE)
        return Widget_stack

    def __init_Widget_btn(self):
        btn_widget = QWidget()
        btn_widget.setStyleSheet(Background_css.TianYi_BLUE)
        VLayout_btn = QVBoxLayout()
        btn_press1 = QPushButton("本地搜图")
        btn_press2 = QPushButton("网络搜图")
        btn_press3 = QPushButton("使用帮助")
        btn_press1.setStyleSheet(Button_css.BTN_ARONA)
        btn_press2.setStyleSheet(Button_css.BTN_ARONA)
        btn_press3.setStyleSheet(Button_css.BTN_ARONA)
        # 添加点击事件
        btn_press1.clicked.connect(self.__switch_to_local)
        btn_press2.clicked.connect(self.__switch_to_net)
        btn_press3.clicked.connect(self.__switch_to_tip)
        VLayout_btn.addWidget(btn_press1)
        # VLayout_btn.addWidget(btn_press2)
        VLayout_btn.addWidget(btn_press3)
        VLayout_btn.addStretch(1)
        btn_widget.setLayout(VLayout_btn)
        return btn_widget

    def __switch_to_local(self):
        self.Layout_stack.setCurrentIndex(0)  # 设置抽屉布局器的当前索引值，即可切换显示哪个Widget

    def __switch_to_net(self):
        self.Layout_stack.setCurrentIndex(1)

    def __switch_to_tip(self):
        self.Layout_stack.setCurrentIndex(2)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = PSP_UI()
    ui.show()
    sys.exit(app.exec())
