import os

from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt6.QtCore import Qt

from UI.config import Text as Text_config
from UI.config import Button_css, Text_css
Button_css = Button_css()
Text_css = Text_css()


class Win_Local(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        # QLabel("本地图片搜索\n", self)
        # self.setStyleSheet("background-color:lightblue;")

    def init_ui(self):
        self.V_Layout_main = self._init_V_Layout_main()

        self.setLayout(self.V_Layout_main)

    # 整体布局
    def _init_V_Layout_main(self):
        V_Layout_main = QVBoxLayout()
        self.H_Layout_localmodel = self._init_HLayout_localmodel()
        V_Layout_main.addLayout(self.H_Layout_localmodel)

        self.H_Layout_searchchoice = self._init_HLayout_searchchoice()
        V_Layout_main.addLayout(self.H_Layout_searchchoice)

        self.search_result = QLabel("搜索结果..")

        V_Layout_main.addWidget(self.search_result)
        V_Layout_main.addStretch(1)
        return V_Layout_main

    # 本地搜索(本地图库路径，图片数据路径)
    def _init_HLayout_localmodel(self):
        H_Layout_localmodel = QHBoxLayout()
        # 本地图库路径(用户选择路径)
        btn_choice = QPushButton("选择文件夹")
        btn_choice.setStyleSheet(Button_css.BTN_BLUE_PURPLE)
        btn_choice.clicked.connect(self.__on_choice_video_folder)

        # 创建文件夹路径显示文本
        self.Label_video_path = QLabel(
            f"当前路径：{self.__shorten_folder_path('测试')}")
        self.Label_video_path.setStyleSheet(Text_css.TEXT_GRAY_12)

        H_Layout_localmodel.addWidget(btn_choice)
        H_Layout_localmodel.addWidget(self.Label_video_path)
        H_Layout_localmodel.addStretch(1)
        return H_Layout_localmodel

    # 匹配规则
    def _init_HLayout_searchchoice(self):
        H_Layout_searchchoice = QHBoxLayout()
        # 匹配规则
        self.Label_search_choice = QLabel("匹配规则：")
        self.Label_search_choice.setStyleSheet(Text_css.TEXT_GRAY_12)
        self.Button_search_choice = QPushButton("选择")
        self.Button_search_choice.setStyleSheet(Button_css.BTN_BLUE_PURPLE)
        self.Button_search_choice.clicked.connect(self.__on_choice_search_rule)

        H_Layout_searchchoice.addWidget(self.Label_search_choice)
        H_Layout_searchchoice.addWidget(self.Button_search_choice)
        H_Layout_searchchoice.addStretch(1)
        return H_Layout_searchchoice

    # 选择本地图库文件夹
    def __on_choice_video_folder(self):
        pass

    # 选择匹配规则
    def __on_choice_search_rule(self):
        pass

    # 获取文件夹的显示文本
    def __shorten_folder_path(self, folder, max_length=50):
        """
        获取文件夹的显示文本
        :param folder: 文件夹路径
        :param max_length: 最大长度
        :return: 显示文本
        """
        folders = folder.split(os.path.sep)  # 使用操作系统的路径分隔符进行分割
        if len(folder) > max_length:
            total_chars = 0
            # 从后往前遍历，直到总字符数超过max_length
            i = len(folders) - 1
            for i in range(len(folders) - 1, -1, -1):
                total_chars += len(folders[i])
                if total_chars > max_length:
                    break
            folder_show = os.path.sep.join(folders[i:])
            folder_show = "...\\" + folder_show
        else:
            folder_show = folder
        return folder_show