import os
import numpy as np
import cv2
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QButtonGroup, \
    QLineEdit, QStackedLayout, QListWidget, QApplication
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt

from Tools.search_pic import SP
from UI.config import Button_css, Text_css, Input_css, Background_css, List_css

Button_css = Button_css()
Text_css = Text_css()
Input_css = Input_css()
Background_css = Background_css()
List_css = List_css()


class SPInit(QThread):
    finished = pyqtSignal()
    log_signal = pyqtSignal(str)  # 信号，用于传递字符串

    def __init__(self, local_path):
        super().__init__()
        self.local_path = local_path

    def run(self):
        self.sp = SP()
        self.sp.init_pic_df(path_local=self.local_path, log_callback=self.update_log)
        self.finished.emit()

    def update_log(self, log_text):
        self.log_signal.emit(log_text)


class SPSearch(QThread):
    finished = pyqtSignal()
    result_list = pyqtSignal(list)

    def __init__(self, img, model_path, search_choice, ori_num=None, sim_threshold=None, par_topk=None):
        super().__init__()
        self.img = img
        self.model_path = model_path
        self.search_choice = search_choice
        self.ori_num = ori_num
        self.sim_threshold = sim_threshold
        self.par_topk = par_topk

    def run(self):
        self.sp = SP()
        print(f"[SPSearch] 初始化SP对象，model_path: {self.model_path}, search_choice: {self.search_choice}")
        self.sp.init_pic_df(save_path=self.model_path)
        result_list = []
        print(f"[SPSearch] model_path: {self.model_path}, search_choice: {self.search_choice}, ", end="")
        # 查看self.img的类型
        print(
            f"[SPSearch] img type: {type(self.img)}, shape: {self.img.shape if isinstance(self.img, np.ndarray) else 'N/A'}")
        if self.search_choice == "ori":
            print(f"ori_num: {self.ori_num}")
            result_list = self.sp.search_origin(self.img, max_result=self.ori_num)
        elif self.search_choice == "sim":
            print(f"sim_threshold: {self.sim_threshold}")
            # print(self.img.shape)
            result_list = self.sp.search_similar(self.img, hash_threshold=self.sim_threshold)
        elif self.search_choice == "par":
            print(f"par_topk: {self.par_topk}")
            result_list = self.sp.search_partial(self.img, top_k=self.par_topk)
        self.result_list.emit(result_list)


class Win_Local(QWidget):
    def __init__(self):
        super().__init__()
        self.local_path = None  # 本地图库路径
        self.model_name = None  # 本地图库模型的名字
        self.model_path = None  # 本地图库模型的路径(完整路径)
        self.input_img = None  # 待搜索的图片(从剪贴板或上传的图片)
        # 默认设置为"ori"原图搜索。"sim"为相似图搜索。暂不支持"wat"水印搜索、"par"(partial)局部搜索
        self.search_choice = "ori"
        self.result_list = []  # 搜索结果列表
        self.ori_num = -1  # 原图搜索的数量
        self.sim_threshold = 0.2  # 相似图搜索的阈值
        self.par_topk = 5  # 局部搜索的top_k
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

        self.H_Layout_searchtip = self.init_HLayout_searchtip()
        V_Layout_main.addLayout(self.H_Layout_searchtip)

        self.H_Layout_inputreturn = self.init_HLayout_inputreturn()
        V_Layout_main.addLayout(self.H_Layout_inputreturn)

        V_Layout_main.addStretch(10)
        return V_Layout_main

    # 本地搜索(本地图库路径，图片模型路径)
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
        self.Label_model_path = QLabel("模型路径：当前还未选择图库模型文件。")
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

    # 搜索规则(原图、近似、水印、局部)
    def init_HLayout_searchchoice(self):
        H_Layout_searchchoice = QHBoxLayout()
        # 搜索规则
        self.Label_search_choice = QLabel("搜索规则：")
        self.Label_search_choice.setStyleSheet(Text_css.TEXT_BLACK_16)

        # 创建按钮
        self.ori_button = QPushButton('原图', self)
        self.sim_button = QPushButton('近似', self)
        # self.mar_button = QPushButton('水印', self)
        self.par_button = QPushButton('局部', self)
        # for btn in [self.ori_button, self.sim_button, self.mar_button, self.par_button]:
        #     btn.setStyleSheet(Button_css.BTN_BLUE_GREEN_CHECK)
        #     btn.setCheckable(True)
        for btn in [self.ori_button, self.sim_button, self.par_button]:
            btn.setStyleSheet(Button_css.BTN_BLUE_GREEN_CHECK)
            btn.setCheckable(True)
        self.ori_button.setChecked(True)
        # 创建一个按钮组
        self.button_group = QButtonGroup(self)
        self.button_group.addButton(self.ori_button)
        self.button_group.addButton(self.sim_button)
        # self.button_group.addButton(self.mar_button)
        self.button_group.addButton(self.par_button)
        # 设置按钮组互斥
        self.button_group.setExclusive(True)
        # 连接信号
        self.ori_button.clicked.connect(self.__on_search_choice_ori)
        self.sim_button.clicked.connect(self.__on_search_choice_sim)
        # self.mar_button.clicked.connect(self.__on_search_choice_mar)
        self.par_button.clicked.connect(self.__on_search_choice_par)

        H_Layout_searchchoice.addWidget(self.Label_search_choice)
        H_Layout_searchchoice.addWidget(self.ori_button)
        H_Layout_searchchoice.addWidget(self.sim_button)
        # H_Layout_searchchoice.addWidget(self.mar_button)
        H_Layout_searchchoice.addWidget(self.par_button)
        H_Layout_searchchoice.addStretch(1)
        return H_Layout_searchchoice

    # 提示&参数(原图、近似、水印、局部)
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
        self.Label_sim_param = QLabel("容忍度(默认为0.2)")
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
        self.Label_par_tip = QLabel("当前选择局部搜索")
        self.Input_par = QLineEdit()
        self.Input_par.setPlaceholderText("top_k")
        self.Input_par.textChanged.connect(self.__update_par_topk)
        self.Input_par.setStyleSheet(Input_css.INPUT_BLUE_PINK)
        self.Label_par_param = QLabel("搜索数量(默认为5)")
        par_layout.addWidget(self.Label_par_tip)
        par_layout.addStretch(1)
        par_layout.addWidget(self.Input_par)
        par_layout.addWidget(self.Label_par_param)
        par_layout.addStretch(30)
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

    # 搜索提示&按钮
    def init_HLayout_searchtip(self):
        H_Layout_searchtip = QHBoxLayout()

        btn_upload = QPushButton("图片上传")
        btn_upload.setStyleSheet(Button_css.BTN_BLUE_PURPLE)
        btn_upload.clicked.connect(self._upload_image)  # 连接上传事件

        self.Label_search_tip = QLabel("通过剪贴板粘贴图片 或 上传图片，点击右侧按钮即可开始搜索")

        btn_search = QPushButton("开始搜索")
        btn_search.setStyleSheet(Button_css.BTN_BLUE_PURPLE)
        btn_search.clicked.connect(self.search_pic)  # 点击按钮，获取剪贴板图片，然后搜索

        H_Layout_searchtip.addWidget(btn_upload)
        H_Layout_searchtip.addStretch(1)
        H_Layout_searchtip.addWidget(self.Label_search_tip)
        H_Layout_searchtip.addWidget(btn_search)
        H_Layout_searchtip.addStretch(20)
        return H_Layout_searchtip

    # 输入&返回(图片input，图片return、搜索结果list)
    def init_HLayout_inputreturn(self):
        H_main_layout = QHBoxLayout()
        H_main_layout.addStretch(2)

        V_left_layout = QVBoxLayout()
        self.image_label1 = QLabel("待搜索的图片(从剪贴板复制)")
        self.image_label1.setFixedSize(325, 225)
        self.image_label1.setStyleSheet(Background_css.XM_BLUE)
        V_left_layout.addWidget(self.image_label1)
        self.image_label2 = QLabel("搜索结果(点击右侧的路径列表以查看图片)")
        self.image_label2.setFixedSize(325, 225)
        self.image_label2.setStyleSheet(Background_css.XM_BLUE)
        V_left_layout.addWidget(self.image_label2)
        H_main_layout.addLayout(V_left_layout)
        H_main_layout.addStretch(1)

        right_layout = QVBoxLayout()

        self.address_list = QListWidget()
        self.address_list.setStyleSheet(List_css.LIGHT_GRAY)
        self.address_list.addItems(self.result_list)
        self.address_list.currentRowChanged.connect(self.__update_image_label2)
        self.address_list.setFixedWidth(350)
        right_layout.addWidget(self.address_list)
        H_main_layout.addLayout(right_layout)
        H_main_layout.addStretch(20)

        if isinstance(self.result_list, list) and len(self.result_list) > 0:
            self.__load_image(self.result_list[0], self.image_label2)
        # Clipboard listener
        self.clipboard = QApplication.clipboard()
        self.clipboard.dataChanged.connect(self.__paste_image)

        return H_main_layout

    # 模型初始化
    def init_sp(self):
        self.thread = SPInit(self.local_path)
        self.thread.finished.connect(self.__on_spinit_finished)
        self.thread.log_signal.connect(self.__update_log_label)
        self.thread.start()

    # 使用模型搜索图片
    def search_sp(self, img):
        # 检查model_path是否存在
        if not isinstance(self.model_path, str) or not os.path.exists(self.model_path):
            self.Label_search_tip.setText(f"模型文件{self.model_path}不存在，请先创建模型文件")
            return False
        # 检查search_choice是否正确
        if self.search_choice not in ["ori", "sim", "par"]:
            self.Label_search_tip.setText(f"搜索方式{self.search_choice}选择错误，请重新选择")
            return False
        # 检查ori_num是否正确(要么是-1要么是正整数)
        if self.search_choice == "ori" and \
                (not isinstance(self.ori_num, int)) and \
                (self.ori_num != -1 and self.ori_num <= 0):
            self.Label_search_tip.setText(f"搜索数量{self.ori_num}设置错误，请重新设置")
            return False
        # 检查sim_threshold是否正确(要么是0~1之间的小数)
        if self.search_choice == "sim" and \
                (not isinstance(self.sim_threshold, float)) and \
                (self.sim_threshold > 1 or self.sim_threshold < 0):
            self.Label_search_tip.setText(f"容忍度{self.sim_threshold}设置错误，请重新设置")
            return False
        # 检查par_topk是否正确(只能是正整数)
        if self.search_choice == "par" and \
                (not isinstance(self.ori_num, int)) and \
                (self.ori_num <= 0):
            self.Label_search_tip.setText(f"搜索数量{self.ori_num}设置错误，请重新设置")

        print(f"[search_sp] model_path: {self.model_path}, search_choice: {self.search_choice}",
              f" ori_num: {self.ori_num}, sim_threshold: {self.sim_threshold}, par_topk: {self.par_topk}")
        self.thread = SPSearch(img, self.model_path, self.search_choice,
                               self.ori_num, self.sim_threshold, self.par_topk)
        self.thread.result_list.connect(self.__update_result_list)
        self.thread.start()

    def search_pic(self):
        if self.input_img is None:
            self.Label_search_tip.setText("请先上传图片或从剪贴板粘贴图片")
            return
        else:
            self.search_sp(self.input_img)

    # 图片上传功能
    def _upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_name:
            # 使用QPixmap显示图片
            pixmap = QPixmap(file_name)
            if not pixmap.isNull():
                self.image_label1.setPixmap(
                    pixmap.scaled(self.image_label1.size(),
                                  Qt.AspectRatioMode.KeepAspectRatio)
                )
                # 使用OpenCV读取图片并存储
                img = cv2.imread(file_name)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self.input_img = img
                    self.Label_search_tip.setText("图片已上传，点击开始搜索按钮进行搜索")
                else:
                    self.Label_search_tip.setText("图片读取失败")
            else:
                self.Label_search_tip.setText("图片文件无法打开")
        else:
            # 用户取消选择，不做处理
            pass

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
            self.local_path = folder
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
        initial_path = "../assets/"
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

    # 更新搜索参数
    def __update_ori_num(self, text):
        try:
            self.ori_num = int(text)
            print(f"Original number set to {self.ori_num}")
            self.Label_ori_param.setText(f"搜索数量(当前为{self.ori_num})")
        except ValueError:
            self.ori_num = -1
            self.Label_ori_param.setText(f"搜索数量设置错误，应该是整数(当前重置为{self.ori_num})")

    def __update_sim_threshold(self, text):
        try:
            self.sim_threshold = float(text)
            # 应该小于1
            if self.sim_threshold > 1 or self.sim_threshold < 0:
                self.sim_threshold = 0.2
                print(f"Similarity threshold set to {self.sim_threshold}")
                self.Label_sim_param.setText(f"容忍度设置错误，只能在0~1之间(当前为{self.sim_threshold})")
                self.Label_sim_param.setText(f"容忍度设置错误，只能在0~1之间(当前为{self.sim_threshold})")
            else:
                print(f"Similarity threshold set to {self.sim_threshold}")
                self.Label_sim_param.setText(f"容忍度(当前为{self.sim_threshold})")
        except ValueError:
            self.sim_threshold = 0.2
            self.Label_sim_param.setText(f"容忍度设置错误，应该是小数(当前重置为{self.sim_threshold})")

    def __update_par_topk(self, text):
        try:
            self.par_topk = int(text)
            print(f"Partial top_k set to {self.par_topk}")
            self.Label_par_param.setText(f"搜索数量(当前为{self.par_topk})")
        except ValueError:
            self.par_topk = 5
            self.Label_par_param.setText(f"搜索数量设置错误，应该是整数(当前重置为{self.par_topk})")

    # sp初始化完成
    def __on_spinit_finished(self):
        self.Label_localmodel.setText(f"图库&模型：{self.model_name}加载完成")
        print("[__on_spinit_finished]SP初始化完成")

    # 更新sp日志
    def __update_log_label(self, log_text):
        self.Label_localmodel.setText(log_text)

    # 获取剪贴板图片
    def __paste_image(self):
        mime = self.clipboard.mimeData()  # 获取剪贴板数据
        # print("-------------------")
        # print(mime.formats())
        # print(mime.hasImage())
        # print(mime.data)
        if mime.hasImage():
            image = self.clipboard.image()
            pixmap = QPixmap.fromImage(image)
            self.image_label1.setPixmap(pixmap.scaled(self.image_label1.size(), Qt.AspectRatioMode.KeepAspectRatio))
            # 将QImage转为numpy数组
            img = self.__qimage2np(image)
            img = self.rgba_to_rgb(img)
            # print(img.shape)
            # 使用模型搜索图片
            # self.search_sp(img)
            self.input_img = img  # 保存剪贴板图片
        else:
            self.image_label1.clear()
            self.image_label1.setText("剪贴板里不是图片")

    # 更新下方图片
    def __update_image_label2(self, index):
        if index >= 0:
            image_path = self.result_list[index]
            self.__load_image(image_path, self.image_label2)
        else:
            self.image_label2.clear()

    # 加载path对应的图片
    def __load_image(self, image_path, label):
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            label.setPixmap(pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio))
        else:
            label.clear()

    # 更新搜索结果列表
    def __update_result_list(self, result_list):
        self.result_list = result_list
        self.address_list.clear()
        self.address_list.addItems(result_list)
        self.image_label2.clear()

        if isinstance(result_list, list) and len(result_list) > 0:
            self.__load_image(result_list[0], self.image_label2)
        print(f"[__update_result_list]搜索结果列表更新完成")
        print(self.result_list)

    # QImage转numpy数组
    # def __qimage2np(self, qimage):
    #     """
    #     本函数有bug。将QImage转换为numpy数组，正确处理图像格式和内存布局
    #     :param qimage: QImage对象
    #     :return: numpy数组 (height, width, channels)
    #     """
    #     # 转换为标准格式：RGB888
    #     qimage = qimage.convertToFormat(QImage.Format.Format_RGB888)
    #
    #     width = qimage.width()
    #     height = qimage.height()
    #     bytes_per_line = qimage.bytesPerLine()
    #
    #     # 获取图像数据指针
    #     ptr = qimage.constBits()
    #
    #     # 计算实际需要的字节数（无填充）
    #     expected_bytes_per_line = width * 3
    #
    #     if bytes_per_line == expected_bytes_per_line:
    #         # 无填充：可以直接创建连续数组
    #         arr = np.array(ptr).reshape((height, width, 3))
    #     else:
    #         # 有填充：逐行复制数据
    #         arr = np.empty((height, width, 3), dtype=np.uint8)
    #         for y in range(height):
    #             # 获取每行数据的起始位置
    #             line_start = y * bytes_per_line
    #             # 复制有效数据（跳过填充字节）
    #             line_data = np.array(ptr[line_start:line_start + expected_bytes_per_line],
    #                                  dtype=np.uint8, copy=False)
    #             arr[y] = line_data.reshape((width, 3))
    #
    #     return arr
    def __qimage2np(self, qimage):
        """
        将QImage转换为numpy数组
        :param qimage: QImage对象
        :return: numpy数组
        """
        # arr = np.array(qimage.convertToFormat(QImage.Format.Format_RGB888))
        # print(arr)
        # print(type(arr))
        # print(arr.dtype)
        size = qimage.size()
        s = qimage.bits().asstring(size.width() * size.height() * qimage.depth() // 8)  # format 0xffRRGGBB

        arr = np.fromstring(s, dtype=np.uint8).reshape((size.height(), size.width(), qimage.depth() // 8))

        # qimage = qimage.convertToFormat(QImage.Format.Format_RGB888)
        # width = qimage.width()
        # height = qimage.height()
        #
        # ptr = qimage.bits()
        # ptr.setsize(qimage.byteCount())
        # arr = np.array(ptr).reshape((height, width, 3))

        return arr

    @staticmethod
    def rgba_to_rgb(img, background=(255, 255, 255)):

        """
        将RGBA四通道图像转换为RGB三通道图像

        参数:
            img: numpy数组，形状为[H, W, 4]的RGBA图像
            background: 背景色元组(R, G, B)，默认为白色(255,255,255)

        返回:
            RGB三通道的numpy数组，形状为[H, W, 3]
        """
        print(f"img.ndim: {img.ndim}, img.shape: {img.shape}")
        if img.ndim != 3:
            raise ValueError("输入图像必须是三维数组 (H, W, C)")
        # 如果已经是RGB则直接返回
        if img.shape[2] == 3:
            return img
        # 否则是RGBA，进行如下处理：

        # 分离通道
        r, g, b, a = [img[:, :, i] for i in range(4)]

        # 归一化Alpha通道 (0-1范围)
        alpha = a.astype(float) / 255.0

        # 扩展背景色到图像尺寸
        bg_r = np.full_like(r, background[0])
        bg_g = np.full_like(g, background[1])
        bg_b = np.full_like(b, background[2])

        # 混合计算
        rgb = np.empty(img.shape[:2] + (3,), dtype=np.uint8)
        rgb[:, :, 0] = (r * alpha + bg_r * (1 - alpha)).clip(0, 255).astype(np.uint8)
        rgb[:, :, 1] = (g * alpha + bg_g * (1 - alpha)).clip(0, 255).astype(np.uint8)
        rgb[:, :, 2] = (b * alpha + bg_b * (1 - alpha)).clip(0, 255).astype(np.uint8)

        return rgb
