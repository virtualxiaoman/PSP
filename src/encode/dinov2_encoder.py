import torch
import numpy as np
import cv2
import time

from src.utils.img_read import read_image


class DINOv2Encoder:
    def __init__(self, model_name="dinov2_vitl14", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DINOv2Encoder] 设备: {self.device}")

        # 加载模型
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval().to(self.device)

        # 标准化参数
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def _preprocess_single(self, img: np.ndarray):
        """处理单张图像，输出 (C, H, W) 的 numpy 数组"""
        # 基础校验
        if img is None: raise ValueError("输入图像为空")

        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        return np.transpose(img, (2, 0, 1))

    @torch.no_grad()
    def encode(self, imgs, max_batch_size=256):
        """
        主接口：自动识别单图或多图，并支持分块推理防止显存溢出

        Args:
            imgs: 单张 np.ndarray 或 List[np.ndarray]
            max_batch_size: 显存允许的最大单次推理数量
        """
        # 1. 自动识别输入模式
        is_list = isinstance(imgs, (list, tuple))
        input_list = imgs if is_list else [imgs]

        all_features = []

        # 2. 分块处理 (Chunking)
        for i in range(0, len(input_list), max_batch_size):
            batch_imgs = input_list[i: i + max_batch_size]

            # 预处理当前 batch
            processed = [self._preprocess_single(img) for img in batch_imgs]
            x = np.stack(processed, axis=0)
            x_tensor = torch.from_numpy(x).float().to(self.device)

            # 模型推理
            features = self.model(x_tensor)  # (current_batch_size, D)
            all_features.append(features.cpu().numpy())

        # 3. 拼接所有分块结果
        final_features = np.concatenate(all_features, axis=0)

        # 4. 根据输入类型返回对应维度
        return final_features if is_list else final_features[0]  # 返回 (N, D) 或者 (D,)


if __name__ == "__main__":
    # 模拟读取 512 张图片（这里为了方便直接重复使用已有的图）
    img_path = "F:/Picture/pixiv/BA/110182236_p0.jpg"
    raw_img = read_image(img_path)
    test_imgs = [raw_img] * 12  # 模拟 512 张图片的 batch

    # --- 1. 单张循环模式 (None-Batch) ---
    encoder_single = DINOv2Encoder()

    # 预热 (Warmup)
    _ = encoder_single.encode(raw_img)

    print("\n>>> 开始测试 [单张循环模式]...")
    start_time = time.time()
    for i in range(len(test_imgs)):
        _ = encoder_single.encode(test_imgs[i])
    single_duration = time.time() - start_time
    print(f"单张模式总耗时: {single_duration:.4f}s, 平均每张: {single_duration / len(test_imgs):.4f}s")

    # --- 2. Batch 模式 ---
    encoder_batch = DINOv2Encoder()

    # 预热 (Warmup)
    _ = encoder_batch.encode([raw_img])

    print("\n>>> 开始测试 [Batch 模式]...")
    start_time = time.time()
    _ = encoder_batch.encode(test_imgs)
    batch_duration = time.time() - start_time
    print(f"Batch 模式总耗时: {batch_duration:.4f}s, 平均每张: {batch_duration / len(test_imgs):.4f}s")

    # --- 结论 ---
    speedup = single_duration / batch_duration
    print(f"\n[结论] Batch 模式比单张模式快了 {speedup:.2f} 倍")
