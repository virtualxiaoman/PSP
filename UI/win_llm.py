import sys
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import cv2
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QFileDialog, QMessageBox, QSplitter
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from volcenginesdkarkruntime import Ark

from Tools.ask_llm import image_to_base64, ask_about_image
from UI.config import Button_css, Background_css

Button_css = Button_css()
Background_css = Background_css()

# 初始化大模型客户端
CLIENT = Ark(api_key='df0d5567-6c5b-4ce0-bef0-a93f5a044e13')


class ModelWorker(QThread):
    """后台线程，用于处理大模型请求"""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, img_np, prompt):
        super().__init__()
        self.img_np = img_np
        self.prompt = prompt

    def run(self):
        try:
            response = ask_about_image(self.img_np, self.prompt)
            self.finished.emit(response)
        except Exception as e:
            self.error.emit(f"发生错误: {str(e)}")


class Win_LLM(QMainWindow):
    """主应用程序窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("图片内容分析")
        self.setGeometry(100, 100, 1000, 600)

        # 主控件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # 创建分割器实现左右布局
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # 左侧区域（图片和按钮）
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)

        # 顶部按钮布局
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 10)

        self.upload_btn = QPushButton("上传图片")
        self.upload_btn.clicked.connect(self.upload_image)
        self.upload_btn.setMinimumHeight(40)
        self.upload_btn.setStyleSheet(Button_css.BTN_ARONA)
        button_layout.addWidget(self.upload_btn)

        self.analyze_btn = QPushButton("分析图片")
        self.analyze_btn.clicked.connect(self.analyze_image)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setMinimumHeight(40)
        self.analyze_btn.setStyleSheet(Button_css.BTN_ARONA)
        button_layout.addWidget(self.analyze_btn)

        left_layout.addLayout(button_layout)

        # 图片显示区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet(Background_css.XM_BLUE)
        left_layout.addWidget(self.image_label)

        splitter.addWidget(left_widget)

        # 右侧区域（结果展示）
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 0, 0, 0)

        # 结果标题
        result_title = QLabel("分析结果")
        result_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 5px;")
        right_layout.addWidget(result_title)

        # 结果展示框
        self.result_edit = QTextEdit()
        self.result_edit.setPlaceholderText("分析结果将显示在这里...")
        self.result_edit.setReadOnly(True)
        self.result_edit.setStyleSheet("""
            QTextEdit {
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        right_layout.addWidget(self.result_edit)

        splitter.addWidget(right_widget)

        # 设置分割器初始比例
        splitter.setSizes([400, 600])

        # 当前图片数据
        self.current_image = None
        self.current_image_np = None

        # 固定提示词
        self.fixed_prompt = """
        请识别图片中的角色，并按以下要求回答：
        1. 如果图片中有可识别的角色，请列出所有角色的名字和出处（作品名）
        2. 每个角色单独一行，格式为：角色名，作品名
        3. 不要添加任何解释性文字
        """

    def upload_image(self):
        """上传图片并显示在界面上"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp)"
        )

        if file_path:
            try:
                # 读取图片并转换为NumPy数组
                self.current_image_np = self.load_image(file_path)

                # 显示图片
                pixmap = self.numpy_to_pixmap(self.current_image_np)
                self.image_label.setPixmap(
                    pixmap.scaled(
                        self.image_label.width(),
                        self.image_label.height(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                )
                self.analyze_btn.setEnabled(True)
                self.result_edit.clear()

            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法加载图片: {str(e)}")

    def load_image(self, file_path):
        """加载图片为NumPy数组（RGB格式）"""
        # 使用OpenCV读取
        img = cv2.imread(file_path)
        if img is None:
            try:
                img = np.array(Image.open(file_path))
                # 确保是RGB格式
                if img.ndim == 2:  # 灰度图
                    img = np.stack((img,) * 3, axis=-1)
                elif img.shape[2] == 4:  # RGBA
                    img = img[:, :, :3]
            except Exception as e:
                raise ValueError(f"图片加载失败: {str(e)}")
        else:
            # OpenCV读取的是BGR，转换为RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def numpy_to_pixmap(self, img_np):
        """将NumPy数组转换为QPixmap"""
        h, w, c = img_np.shape
        bytes_per_line = w * c
        qimg = QImage(
            img_np.data, w, h, bytes_per_line,
            QImage.Format.Format_RGB888
        )
        return QPixmap.fromImage(qimg)

    def analyze_image(self):
        """分析当前图片"""
        if self.current_image_np is None:
            QMessageBox.warning(self, "警告", "请先上传图片")
            return

        # 禁用按钮防止重复请求
        self.analyze_btn.setEnabled(False)
        self.result_edit.setPlainText("分析中，请稍候...")

        # 创建后台工作线程
        self.worker = ModelWorker(self.current_image_np, self.fixed_prompt)
        self.worker.finished.connect(self.handle_result)
        self.worker.error.connect(self.handle_error)
        self.worker.finished.connect(lambda: self.analyze_btn.setEnabled(True))
        self.worker.error.connect(lambda: self.analyze_btn.setEnabled(True))
        self.worker.start()

    def handle_result(self, result):
        """处理分析结果"""
        self.result_edit.setPlainText(result)

    def handle_error(self, error_msg):
        """处理错误信息"""
        self.result_edit.setPlainText(error_msg)
        QMessageBox.critical(self, "错误", error_msg)

    def resizeEvent(self, event):
        """窗口大小变化时调整图片大小"""
        super().resizeEvent(event)
        if self.current_image_np is not None:
            pixmap = self.numpy_to_pixmap(self.current_image_np)
            self.image_label.setPixmap(
                pixmap.scaled(
                    self.image_label.width(),
                    self.image_label.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Win_LLM()
    window.show()
    sys.exit(app.exec())
