from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout


class Win_Tip(QWidget):
    def __init__(self):
        super().__init__()

        # 创建一个垂直布局
        layout = QVBoxLayout()

        # 添加标题
        title_label = QLabel("图片搜索工具使用文档", self)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title_label)

        # 添加各个部分的说明
        sections = [
            ("程序概述", "这是一个功能强大的本地图片搜索工具，支持本地图片库的创建和图片搜索，可快速查找与目标图片相似、完全相同的图片，甚至可以进行局部特征匹配。"),
            ("图库&模型设置", "1. 点击“选择图库”，选择你希望搜索的图片文件夹。\n2. 若已有一个模型文件（后缀为.pkl），可直接点击“选择模型”，选择已有模型文件路径。"),
            ("搜索图片", "1. 上传图片：点击“图片上传”按钮，选择一张图片作为搜索目标。\n2. 或从剪贴板粘贴图片：将图片复制到剪贴板，然后粘贴到左侧“待搜索的图片”区域。\n3. 搜索图片规则选择：选择搜索规则（原图、近似、局部）并设置相关参数。\n4. 开始搜索：点击“开始搜索”按钮，程序会在图库中进行搜索并显示结果。"),
            ("搜索结果", "搜索框下的“搜索结果”列表会显示匹配的图片路径。点击列表中的图片路径，右侧的“搜索结果”区域会显示对应的图片。"),
            ("功能说明", "- 支持图库搜索\n- 提供原图、近似图、局部图等多种搜索方式\n- 支持从剪贴板直接粘贴图片进行搜索\n- 搜索图片结果列表展示"),
            ("常见问题解答", "Q: 为什么搜索结果为空？\nA: 可能是图库路径未正确设置或模型文件未加载成功。")
        ]

        for title, content in sections:
            section_title = QLabel(title, self)
            section_title.setStyleSheet("font-size: 14px; font-weight: bold; margin-top: 15px;")
            section_content = QLabel(content, self)
            section_content.setStyleSheet("font-size: 12px; margin-bottom: 10px;")

            layout.addWidget(section_title)
            layout.addWidget(section_content)

        # 设置布局
        self.setLayout(layout)

        # 设置窗口标题和大小
        self.setWindowTitle("使用文档")
        self.resize(600, 800)