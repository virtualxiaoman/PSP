import os

from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QButtonGroup,\
    QLineEdit, QStackedLayout
from PyQt6.QtCore import QThread, pyqtSignal

from Tools.search_pic import SP
from UI.config import Button_css, Text_css, Input_css
Button_css = Button_css()
Text_css = Text_css()
Input_css = Input_css()

class SPInit(QThread):
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path

    def run(self):
        self.sp = SP()
        self.sp.init_pic_df(path_local=self.model_path)
        self.finished.emit()

class Win_Local(QWidget):
    def __init__(self):
        super().__init__()
        self.model_name = None  # 本地图库模型的名字
        self.model_path = None  # 本地图库模型的路径(完整路径)
        self.search_choice = "ori"  # 默认设置为"ori"原图搜索。"sim"为相似图搜索。暂不支持"wat"水印搜索、"par"局部搜索
        self.init_ui()

    def init_ui(self):
        self.V_Layout_main = self.init_V_Layout_main()

        self.setLayout(self.V_Layout_main)

    # 整体布局
    def init_V_Layout_main(self):
        V_Layout_main = QVBoxLayout()
        self.H_Layout_localmodel = self.init_HLayout_localmodel()
        V_Layout_main.addLayout(self.H_Layout_localmodel)

        self.Label_localmodel = QLabel("请先选择图库，再选择搜索规则")
        self.Label_localmodel.setStyleSheet(Text_css.TEXT_GRAY_12)
        V_Layout_main.addWidget(self.Label_localmodel)
        V_Layout_main.addStretch(1)

        self.H_Layout_searchchoice = self.init_HLayout_searchchoice()
        V_Layout_main.addLayout(self.H_Layout_searchchoice)

        self.H_Layout_tipparam = self.init_HLayout_tipparam()
        V_Layout_main.addLayout(self.H_Layout_tipparam)

        self.search_result = QLabel("搜索结果..（占位符）")
        V_Layout_main.addWidget(self.search_result)
        V_Layout_main.addStretch(30)
        return V_Layout_main

    # 本地搜索(本地图库路径，图片数据路径)
    def init_HLayout_localmodel(self):
        H_Layout_localmodel = QHBoxLayout()
        Label_localmodel = QLabel("图库&模型：")
        Label_localmodel.setStyleSheet(Text_css.TEXT_BLACK_16)

        # 本地图库路径(用户选择路径)
        btn_local = QPushButton("选择图库")
        btn_local.setStyleSheet(Button_css.BTN_BLUE_PURPLE)
        btn_local.clicked.connect(self.__on_choice_local_folder)
        # 创建文件夹路径显示文本
        self.Label_local_path = QLabel("图库路径：当前还未选择文件夹")
        self.Label_local_path.setStyleSheet(Text_css.TEXT_GRAY_12)

        # 图库模型路径(用户选择路径)
        btn_model = QPushButton("选择模型")
        btn_model.setStyleSheet(Button_css.BTN_BLUE_PURPLE)
        btn_model.clicked.connect(self.__on_choice_model)
        # 创建文件夹路径显示文本
        self.Label_model_path = QLabel("模型路径：当前还未选择图库模型文件。PS.当前不要修改默认路径，我没写这个功能")
        self.Label_model_path.setStyleSheet(Text_css.TEXT_GRAY_12)

        H_Layout_localmodel.addWidget(Label_localmodel)
        H_Layout_localmodel.addStretch(1)
        H_Layout_localmodel.addWidget(btn_local)
        H_Layout_localmodel.addWidget(self.Label_local_path)
        H_Layout_localmodel.addStretch(2)
        H_Layout_localmodel.addWidget(btn_model)
        H_Layout_localmodel.addWidget(self.Label_model_path)
        H_Layout_localmodel.addStretch(30)
        return H_Layout_localmodel

    # 搜索规则
    def init_HLayout_searchchoice(self):
        H_Layout_searchchoice = QHBoxLayout()
        # 搜索规则
        self.Label_search_choice = QLabel("搜索规则：")
        self.Label_search_choice.setStyleSheet(Text_css.TEXT_BLACK_16)

        # 创建按钮
        self.ori_button = QPushButton('原图', self)
        self.sim_button = QPushButton('近似', self)
        self.mar_button = QPushButton('水印', self)
        self.par_button = QPushButton('局部', self)
        for btn in [self.ori_button, self.sim_button, self.mar_button, self.par_button]:
            btn.setStyleSheet(Button_css.BTN_BLUE_GREEN_CHECK)
            btn.setCheckable(True)
        self.ori_button.setChecked(True)
        # 创建一个按钮组
        self.button_group = QButtonGroup(self)
        self.button_group.addButton(self.ori_button)
        self.button_group.addButton(self.sim_button)
        self.button_group.addButton(self.mar_button)
        self.button_group.addButton(self.par_button)
        # 设置按钮组互斥
        self.button_group.setExclusive(True)
        # 连接信号
        self.ori_button.clicked.connect(self.__on_search_choice_ori)
        self.sim_button.clicked.connect(self.__on_search_choice_sim)
        self.mar_button.clicked.connect(self.__on_search_choice_mar)
        self.par_button.clicked.connect(self.__on_search_choice_par)

        H_Layout_searchchoice.addWidget(self.Label_search_choice)
        H_Layout_searchchoice.addWidget(self.ori_button)
        H_Layout_searchchoice.addWidget(self.sim_button)
        H_Layout_searchchoice.addWidget(self.mar_button)
        H_Layout_searchchoice.addWidget(self.par_button)
        H_Layout_searchchoice.addStretch(1)
        return H_Layout_searchchoice

    # 提示&参数
    def init_HLayout_tipparam(self):
        # 提示区域布局
        self.stacked_layout = QStackedLayout()

        # 原图提示
        ori_widget = QWidget()
        ori_layout = QHBoxLayout()
        self.Label_ori_tip = QLabel("当前选择原图搜索")
        self.Input_ori = QLineEdit()
        self.Input_ori.setPlaceholderText("搜索数量")
        self.Input_ori.textChanged.connect(self.__update_ori_num)
        self.Input_ori.setStyleSheet(Input_css.INPUT_BLUE_PINK)
        self.Label_ori_param = QLabel("搜索数量(默认为-1)")
        ori_layout.addWidget(self.Label_ori_tip)
        ori_layout.addStretch(1)
        ori_layout.addWidget(self.Input_ori)
        ori_layout.addWidget(self.Label_ori_param)
        ori_layout.addStretch(30)
        ori_widget.setLayout(ori_layout)

        # 近似提示
        sim_widget = QWidget()
        sim_layout = QHBoxLayout()
        self.Label_sim_tip = QLabel("当前选择近似搜索")
        self.Input_sim = QLineEdit()
        self.Input_sim.setPlaceholderText("容忍度")
        self.Input_sim.textChanged.connect(self.__update_sim_threshold)
        self.Input_sim.setStyleSheet(Input_css.INPUT_BLUE_PINK)
        self.Label_sim_param = QLabel("容忍度(默认为0.1)")
        sim_layout.addWidget(self.Label_sim_tip)
        sim_layout.addStretch(1)
        sim_layout.addWidget(self.Input_sim)
        sim_layout.addWidget(self.Label_sim_param)
        sim_layout.addStretch(30)
        sim_widget.setLayout(sim_layout)

        # 水印提示
        mar_widget = QWidget()
        mar_layout = QHBoxLayout()
        self.Label_mar_tip = QLabel("当前选择水印搜索，但该功能暂不支持")
        mar_layout.addWidget(self.Label_mar_tip)
        mar_layout.addStretch(1)
        mar_widget.setLayout(mar_layout)

        # 局部提示
        par_widget = QWidget()
        par_layout = QHBoxLayout()
        self.Label_par = QLabel("当前选择局部搜索，但该功能暂不支持")
        par_layout.addWidget(self.Label_par)
        par_layout.addStretch(1)
        par_widget.setLayout(par_layout)

        # 添加到堆叠布局
        self.stacked_layout.addWidget(ori_widget)
        self.stacked_layout.addWidget(sim_widget)
        self.stacked_layout.addWidget(mar_widget)
        self.stacked_layout.addWidget(par_widget)

        # 默认显示原图提示
        self.stacked_layout.setCurrentIndex(0)

        # # 提示区域布局
        # self.Label_search_tip = QLabel("当前选择原图搜索")
        # # self.Label_search_tip.setStyleSheet(Text_css.TEXT_BLACK_12)
        # self.Input_param1 = QLineEdit()
        # self.Input_param1.setPlaceholderText("搜索数量")
        # self.Input_param1.textChanged.connect(self.__update_ori_num)
        # self.Input_param1.setStyleSheet(Input_css.INPUT_BLUE_PINK)
        # self.Label_param1 = QLabel("搜索数量(默认为-1)")
        # # self.Label_param1.setStyleSheet(Text_css.TEXT_BLACK_12)
        #
        # H_Layout_tipparam.addWidget(self.Label_search_tip)
        # H_Layout_tipparam.addWidget(self.Input_param1)
        # H_Layout_tipparam.addWidget(self.Label_param1)
        # H_Layout_tipparam.addStretch(1)
        return self.stacked_layout

    # 模型初始化
    def init_sp(self):
        self.thread = SPInit(self.model_path)
        self.thread.finished.connect(self.__on_spinit_finished)
        self.thread.start()

    # 选择本地图库文件夹
    def __on_choice_local_folder(self):
        initial_path = "../"
        initial_path = os.path.abspath(initial_path)
        initial_path = os.path.normpath(initial_path).replace("\\", "/")
        print(f"[__on_choice_video_folder]当前路径为：{initial_path}")
        self.Label_local_path.setText(f"当前路径：{self.__shorten_folder_path(initial_path)}")
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹", initial_path)
        if folder:
            folder = os.path.normpath(folder)
            folder = folder.replace("\\", "/")
            # print(len(folder))
            folder_show = self.__shorten_folder_path(folder)  # 如果folder太长，只显示最后一部分
            # print(f"[__on_choice_video_folder]选择的文件夹为：{folder}")
            # print(f"[__on_choice_video_folder]显示的文件夹为：{folder_show}")
            self.Label_local_path.setText(f'图库路径：{folder_show}')
            # model_name 取 folder 的最后一个文件夹名
            self.model_name = folder.split('/')[-1]
            self.__model_name2model_path()
            self.Label_model_path.setText(f'模型路径(自动设置)：{self.model_path}')
            # 执行模型初始化
            self.init_sp()
        else:
            print("[__on_choice_video_folder]未选择文件夹")

    # 选择模型文件
    def __on_choice_model(self):
        self.__model_name2model_path()
        print(f"[__on_choice_model]当前路径为：{self.model_path}")
        # self.Label_model_path.setText(f"当前路径：{self.__shorten_folder_path(initial_path)}")
        file, file_type = QFileDialog.getOpenFileName(self, "选择文件", self.model_path)
        print(file)
        if file:
            folder = os.path.normpath(file)
            folder = folder.replace("\\", "/")
            print(len(folder))
            folder_show = self.__shorten_folder_path(folder)
            print(f"[__on_choice_model]选择的文件为：{folder}")
            print(f"[__on_choice_model]显示的文件为：{folder_show}")
            self.model_name = folder.split('/')[-1]
            self.Label_model_path.setText(f'模型路径：{folder_show}')
            self.model_path = folder
        else:
            print("[__on_choice_model]未选择模型文件")

    # 获取文件夹的显示文本
    def __shorten_folder_path(self, folder, max_length=50):
        """
        获取文件夹的显示文本
        :param folder: 文件夹路径
        :param max_length: 最大长度
        :return: 显示文本
        """
        folders = folder.split('/')  # 使用操作系统的路径分隔符进行分割
        if len(folder) > max_length:
            total_chars = 0
            # 从后往前遍历，直到总字符数超过max_length
            i = len(folders) - 1
            for i in range(len(folders) - 1, -1, -1):
                total_chars += len(folders[i])
                if total_chars > max_length:
                    break
            folder_show = os.path.sep.join(folders[i:])
            folder_show = ".../" + folder_show
        else:
            folder_show = folder
        return folder_show

    # 从model_name到model_path
    def __model_name2model_path(self):
        initial_path = "../data/"
        initial_path = os.path.abspath(initial_path)
        initial_path = os.path.normpath(initial_path).replace("\\", "/")
        self.model_path = f"{initial_path}/{self.model_name}.pkl"

    # 设置搜索方式
    def __on_search_choice_ori(self):
        self.search_choice = "ori"
        self.stacked_layout.setCurrentIndex(0)
        print("Search choice", self.search_choice)

    def __on_search_choice_sim(self):
        self.search_choice = "sim"
        self.stacked_layout.setCurrentIndex(1)
        print("Search choice", self.search_choice)

    def __on_search_choice_mar(self):
        self.search_choice = "mar"
        self.stacked_layout.setCurrentIndex(2)
        print("Search choice", self.search_choice)

    def __on_search_choice_par(self):
        self.search_choice = "par"
        self.stacked_layout.setCurrentIndex(3)
        print("Search choice", self.search_choice)

    # # 更新提示区域布局
    # def __update_tip_layout(self, label_text, placeholder_text, connect_function=None):
    #     self.Label_search_tip.setText(label_text)
    #     # 如果placeholder_text不为空，则设置placeholder_text，否则清空placeholder_text
    #     if placeholder_text:
    #         self.Input_param1.setPlaceholderText(placeholder_text)
    #         self.Input_param1.setEnabled(True)
    #         if connect_function:
    #             self.Input_param1.textChanged.disconnect()
    #             self.Input_param1.textChanged.connect(connect_function)
    #     else:
    #         self.Input_param1.setPlaceholderText("")
    #         self.Input_param1.setEnabled(False)

    def __update_ori_num(self, text):
        try:
            self.ori_num = int(text)
            print(f"Original number set to {self.ori_num}")
            self.Label_ori_param.setText(f"搜索数量(当前为{self.ori_num})")
        except ValueError:
            self.ori_num = -1
            self.Label_ori_param.setText(f"搜索数量设置错误，应该是整数(当前为{self.ori_num})")

    def __update_sim_threshold(self, text):
        try:
            self.sim_threshold = float(text)
            # 应该小于1
            if self.sim_threshold > 1 or self.sim_threshold < 0:
                self.sim_threshold = 0.1
                print(f"Similarity threshold set to {self.sim_threshold}")
                self.Label_sim_param.setText(f"容忍度设置错误，只能在0~1之间(当前为{self.sim_threshold})")
            else:
                print(f"Similarity threshold set to {self.sim_threshold}")
                self.Label_sim_param.setText(f"容忍度(当前为{self.sim_threshold})")
        except ValueError:
            self.sim_threshold = 0.1
            self.Label_sim_param.setText(f"容忍度设置错误，应该是小数(当前为{self.sim_threshold})")

    # sp初始化完成
    def __on_spinit_finished(self):
        self.Label_localmodel.setText(f"图库&模型：{self.model_name}加载完成")
        print("[__on_spinit_finished]SP初始化完成")
