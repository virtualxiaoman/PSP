class Text:
    def __init__(self):
        self.WindowTitle = "PSP_V0.0"


class Background_css:
    def __init__(self):
        # 背景颜色
        self.WHITE = "background-color:white;"
        self.TianYi_BLUE = "background-color: rgba(102, 204, 255, 0.2);"  # 天依蓝
        self.XM_BLUE = "background-color: rgba(178,216,232, 0.3);"  # 小满蓝
        self.LIGHT_GRAY = "background-color: rgba(222,223,224, 0.3);"  # 浅灰色

class Button_css:
    def __init__(self):
        # 聚类算法得出阿洛娜浅蓝203 226 239 深蓝71 224 241，普拉娜浅紫238 228 241 深紫44 48 72
        # 淡蓝色+浅紫色边框，圆角，悬停时颜色变成淡蓝色的变体+浅紫色边框的变体
        self.BTN_ARONA = """
        QPushButton {
            background-color: rgba(71, 224, 241, 0.3);
            border: 2px solid rgba(238, 228, 241, 1);
            border-radius: 6px;
            padding: 4px 8px;
        }
        QPushButton:hover {
            background-color: rgba(203, 226, 239, 0.8);
            border-color: rgba(44, 48, 72, 0.6);
        }
        QPushButton:pressed {
            background-color: rgba(203, 226, 239, 1);
            border-color: rgba(44, 48, 72, 1);
        }
        """
        # 上面这个颜色不适合白色背景，所以我自己调了一下QAQ
        self.BTN_BLUE_PURPLE = """
        QPushButton {
            background-color: rgba(71, 224, 241, 0.3);
            border: 2px solid rgba(206,147,216, 0.8);
            border-radius: 6px;
            padding: 4px 8px;
        }
        QPushButton:hover {
            background-color: rgba(203, 226, 239, 0.8);
            border-color: rgba(156,39,176, 0.6);
        }
        QPushButton:pressed {
            background-color: rgba(203, 226, 239, 1);
            border-color: rgba(156,39,176, 1);
        }
        """
        # 淡绿色边框，圆角，悬停时颜色变成绿色，按下时颜色变成深绿色，checked时border浅绿色background蓝色
        self.BTN_BLUE_GREEN_CHECK = """
        QPushButton {
            background-color: rgba(71, 224, 241, 0.3);
            border: 2px solid rgba(149,212,117, 0.3);
            border-radius: 6px;
            padding: 4px 8px;
        }
        QPushButton:hover {
            border-color: rgba(149,212,117, 0.6);
        }
        QPushButton:pressed {
            border-color: rgba(149,212,117, 1);
        }
        QPushButton:checked {
            background-color: rgba(102, 204, 255, 0.5);
            border: 2px solid rgba(149,212,117, 0.8);
        }
        """

class Input_css:
    def __init__(self):
        # 输入框样式，圆角，蓝色边框，悬停时颜色变成红色
        self.INPUT_BLUE_PURPLE = """
        QLineEdit {
            border: 2px solid rgba(64,158,255, 0.8);
            border-radius: 3px;
            padding: 2px 4px;
            color: rgba(51,126,204, 1);
            font-family: "Times New Roman";
            font-size: 16px;
            width: 300px;
        }
        
        QLineEdit:hover {
            border: 2px solid rgba(216,27,96, 0.6);
        }
        """
        # 输入框样式，圆角，蓝色边框，悬停时颜色变成粉色
        self.INPUT_BLUE_PINK = """
        QLineEdit {
            border: 2px solid rgba(64,158,255, 0.3);
            border-radius: 3px;
            padding: 2px 4px;
            color: rgba(51,126,204, 1);
            font-family: "Times New Roman";
            font-size: 16px;
            width: 100px;
            height: 16px;
        }

        QLineEdit:hover {
            border: 2px solid rgba(233,30,99, 0.4);
        }
        """

class Text_css:
    def __init__(self):
        # 文本样式，字体大小16px，黑色，粗体
        self.TEXT_BLACK_BOLD_16 = "font-size: 16px; color: black; font-weight: bold;"
        # 文本样式，字体大小16px，黑色
        self.TEXT_BLACK_16 = "font-size: 16px; color: black;"
        # 文本样式，字体大小16px，红色
        self.TEXT_RED_16 = "font-size: 16px; color: red;"
        # 文本样式，字体大小14px，黑色
        self.TEXT_BLACK_14 = "font-size: 14px; color: black;"
        # 文本样式，字体大小12px，灰色
        self.TEXT_GRAY_12 = "font-size: 12px; color: gray;"

class List_css:
    def __init__(self):
        self.LIGHT_GRAY = """
        QListWidget {
            background-color: rgba(222,223,224, 0.3);
            border: 1px solid rgba(178,216,232, 0.3);
            font-size: 10px;
            padding: 2px;
        }
        QListWidget::item {
            padding: 3px;
            border-bottom: 1px solid rgba(255,192,203, 0.4)
        }
        QListWidget::item:selected {
            background-color: rgba(121,187,255, 0.8);
            color: rgba(0,0,0, 1);
        }
        """
