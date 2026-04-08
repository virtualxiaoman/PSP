import cv2
import math
import numpy as np
import random

from src.utils.read_pic import read_image


# ==========================================
# 1. 基础组件模块 (Base Component)
# ==========================================
class DegradationComponent:
    """所有退化组件的基类"""

    def __init__(self, apply_prob=0.8):
        # 即使该组件被选中，也有 apply_prob 的概率才真正执行，增加随机性
        self.apply_prob = apply_prob

    def apply(self, img):
        """具体的退化逻辑，由子类实现"""
        raise NotImplementedError

    def __call__(self, img):
        # 概率执行
        if random.random() < self.apply_prob:
            return self.apply(img)
        return img


# ==========================================
# 2. 具体退化组件 (Specific Components)
# ==========================================

class Blur(DegradationComponent):
    """
    多模糊统一组件：
    在 apply() 时随机选择一种模糊方式：
        1. Gaussian Blur（各向同性高斯）
        2. Defocus Blur（散焦模糊）
        3. Motion Blur（运动模糊）

    设计目标：
        - 接口统一（对外只有一个 Blur）
        - 内部模块化（每种模糊独立实现）
        - 参数集中管理（方便调参/复现）
    """

    def __init__(
            self,
            apply_prob=0.8,
            blur_types=("gaussian", "defocus", "motion")
    ):
        super().__init__(apply_prob)
        self.blur_types = blur_types
        self.blur_probs = [0.4, 0.2, 0.4]

    def apply(self, img):
        blur_type = random.choices(self.blur_types, weights=self.blur_probs)[0]

        if blur_type == "gaussian":
            return self._gaussian_blur(img)
        elif blur_type == "defocus":
            return self._defocus_blur(img)
        elif blur_type == "motion":
            return self._motion_blur(img)
        else:
            raise ValueError(f"Unsupported blur type: {blur_type}")

    # 1. Gaussian Blur（各向同性高斯模糊）
    def _gaussian_blur(self, img):
        """
        模拟轻微失焦 / 低质量镜头

        特点：
            - 各向同性（无方向）
            - 平滑高频信息
        """
        kernel_size = random.choice([3, 5, 7, 9])
        sigma = random.uniform(0.5, 2.5)

        blurred = cv2.GaussianBlur(
            img,
            (kernel_size, kernel_size),
            sigmaX=sigma,
            sigmaY=sigma
        )
        return blurred

    # 2. Defocus Blur（散焦模糊）
    def _defocus_blur(self, img):
        """
        模拟真实镜头失焦（景深问题）

        实现方式：
            - 使用圆盘核（disk kernel）
            - 比 Gaussian 更接近真实光学模糊
        """
        radius = random.randint(2, 8)
        kernel = self.__disk_kernel(radius)
        blurred = cv2.filter2D(img, -1, kernel)
        return blurred

    def __disk_kernel(self, radius):
        """
        构造圆盘模糊核

        参数：
            radius: 模糊半径

        返回：
            归一化卷积核
        """
        size = radius * 2 + 1
        kernel = np.zeros((size, size), dtype=np.float32)

        # 在 kernel 中画一个实心圆
        cv2.circle(kernel, (radius, radius), radius, 1, thickness=-1)

        # 归一化（保证能量守恒）
        kernel /= kernel.sum() if kernel.sum() != 0 else 1.0
        return kernel

    # 3. Motion Blur（运动模糊）
    def _motion_blur(self, img):
        """
        模拟拍照时手抖 / 物体运动

        特点：
            - 有方向性
            - 边缘拖影明显
        """
        length = random.randint(5, 25)
        angle = random.uniform(0, np.pi)

        kernel = self.__motion_kernel(length, angle)
        blurred = cv2.filter2D(img, -1, kernel)
        return blurred

    def __motion_kernel(self, length, angle):
        """
        构造运动模糊核

        参数：
            length: 模糊长度（运动幅度）
            angle: 方向（弧度）

        返回：
            归一化卷积核
        """
        kernel = np.zeros((length, length), dtype=np.float32)
        center = length // 2

        # 根据角度计算线段端点
        dx = (length // 2) * math.cos(angle)
        dy = (length // 2) * math.sin(angle)

        x1 = int(center + dx)
        y1 = int(center + dy)
        x2 = int(center - dx)
        y2 = int(center - dy)

        # 在 kernel 上画线
        cv2.line(kernel, (x1, y1), (x2, y2), 1, thickness=1)

        # 归一化
        kernel /= kernel.sum() if kernel.sum() != 0 else 1.0
        return kernel


class ResizeDownsampling(DegradationComponent):
    def apply(self, img):
        h, w = img.shape[:2]
        # 随机缩放比例 (Downsampling)
        scale = random.uniform(0.5, 0.9)
        new_h, new_w = int(h * scale), int(w * scale)

        # 随机选择插值方法: bicubic, bilinear, area
        # Bicubic	平滑但保留结构
        # Bilinear	更模糊
        # Area	类似平均池化
        interpolations = [cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_AREA]
        interp_method = random.choice(interpolations)

        resized_img = cv2.resize(img, (new_w, new_h), interpolation=interp_method)
        return resized_img


class Noise(DegradationComponent):
    def apply(self, img):
        # 随机选择噪声类型：高斯噪声 (可扩展 Poisson, Gray noise 等)
        noise_type = random.choice(['gaussian', 'color_noise'])

        if noise_type == 'gaussian':
            mean = 0
            var = random.uniform(10, 50)
            sigma = var ** 0.5
            gaussian = np.random.normal(mean, sigma, img.shape).astype('float32')
            noisy_img = np.clip(img.astype('float32') + gaussian, 0, 255).astype('uint8')
            return noisy_img
        else:
            # 简化版 Color Noise
            noise = np.random.normal(0, 15, img.shape).astype('float32')
            noisy_img = np.clip(img.astype('float32') + noise, 0, 255).astype('uint8')
            return noisy_img


class ColorDistortion(DegradationComponent):
    def apply(self, img):
        # 色彩失真：随机调整亮度、对比度
        alpha = random.uniform(0.8, 1.2)  # 对比度
        beta = random.uniform(-20, 20)  # 亮度

        distorted_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        return distorted_img


class JPEGCompression(DegradationComponent):
    def apply(self, img):
        # 随机 JPEG 压缩质量 (1-100，越低压缩越狠)
        quality = random.randint(30, 95)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

        # 编码再解码以模拟 JPEG 伪影
        result, encimg = cv2.imencode('.jpg', img, encode_param)
        if result:
            jpeg_img = cv2.imdecode(encimg, 1)
            return jpeg_img
        return img


# ==========================================
# 3. 核心流程管理器 (Pipeline Managers)
# ==========================================
class RandomDegradationOrder:
    """单次（一阶）退化过程：随机选取组件，随机打乱顺序"""

    def __init__(self, component_pool):
        self.component_pool = component_pool

    def __call__(self, img):
        # 随机决定本次退化包含几个组件 (0 ~ 5 个)
        num_components = random.randint(0, len(self.component_pool))

        if num_components == 0:
            return img  # 0个组件，直接跳过本次退化

        # 从池子中随机无放回抽样
        selected_components = random.sample(self.component_pool, num_components)

        # 随机打乱执行顺序，或者尝试blur→downsample→noise→jpeg
        random.shuffle(selected_components)

        # 依次应用选中的退化
        for comp in selected_components:
            img = comp(img)

        return img


class HighOrderDegradationPipeline:
    """高阶退化流水线 (如 4 次退化)"""

    def __init__(self, num_orders=4, apply_prob=(0.8, 0.8, 0.8, 0.8, 0.8)):
        self.num_orders = num_orders

        # 初始化组件池 (每个组件也可以设置自带的触发概率)
        self.pool = [
            Blur(apply_prob=apply_prob[0]),
            ResizeDownsampling(apply_prob=apply_prob[1]),
            Noise(apply_prob=apply_prob[2]),
            ColorDistortion(apply_prob=apply_prob[3]),
            JPEGCompression(apply_prob=apply_prob[4])
        ]

    def process(self, img):
        current_img = img.copy()

        # 执行 4 次 (num_orders) 退化循环
        for order in range(self.num_orders):
            # 实例化单次退化过程
            single_order_process = RandomDegradationOrder(self.pool)

            # 应用单次退化
            current_img = single_order_process(current_img)

        return current_img


# ==========================================
# 4. 测试与运行
# ==========================================
if __name__ == "__main__":
    test_img = read_image("../input/BG_CS_Abydos_10.jpg")
    pipeline = HighOrderDegradationPipeline(num_orders=4)
    degraded_img = pipeline.process(test_img)
    # cv2.imshow("Original", cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))
    cv2.imshow("Degraded", cv2.cvtColor(degraded_img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
