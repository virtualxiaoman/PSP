import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import time
import pandas as pd

from Tools.local_matcher import LocalMatcher
from Tools.read_pic import read_image


class DFGalleryManager:
    def __init__(self, model_path, df):
        """
        初始化基于DataFrame的图库管理器

        参数:
            model_path: 模型权重文件路径
            df: 包含预计算特征的DataFrame，必须包含'dino'列和'path'列
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.df = df.copy()  # 创建副本以避免修改原始数据

        # 加载模型
        self.model = LocalMatcher().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # 准备特征和路径列表
        self.features = []
        self.paths = []

        # 将DataFrame中的特征转换为张量
        for _, row in self.df.iterrows():
            feature = row['dino']

            # 确保特征格式正确
            if isinstance(feature, np.ndarray):
                feature = torch.from_numpy(feature)
            elif not isinstance(feature, torch.Tensor):
                raise TypeError(f"Unsupported feature type: {type(feature)}")

            self.features.append(feature)
            self.paths.append(row['path'])

    def match_image(self, img2_path, top_k=5, batch_size=100):
        """
        匹配用户上传的图片与图库

        参数:
            img2_path: 用户上传的图片路径
            top_k: 返回最相似的前k个结果
            batch_size: 批处理大小，用于控制内存使用

        返回:
            包含top_k个匹配结果的列表，每个元素是字典:
                {'path': 图片路径, 'probability': 匹配概率}
        """
        # 处理用户图片
        img2 = Image.open(img2_path).convert("RGB")
        img2_tensor = self.transform(img2).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # 提取用户图片特征
            feat2 = self.model.backbone(img2_tensor)

            # 初始化概率数组
            all_probabilities = []

            # 分批处理图库特征
            num_images = len(self.features)
            for i in range(0, num_images, batch_size):
                # 获取当前批次的特征
                batch_end = min(i + batch_size, num_images)
                batch_features = self.features[i:batch_end]

                # 转换为张量并移动到设备
                feat1_batch = torch.stack(batch_features).to(self.device)
                feat2_batch = feat2.repeat(len(batch_features), 1)

                # 批量计算相似度
                query1 = feat1_batch.unsqueeze(1)  # [B, 1, feat_dim]
                key2 = feat2_batch.unsqueeze(1)
                value2 = feat2_batch.unsqueeze(1)
                attn1, _ = self.model.cross_attn(query1, key2, value2)
                attn1 = attn1.squeeze(1)  # [B, feat_dim]

                query2 = feat2_batch.unsqueeze(1)
                key1 = feat1_batch.unsqueeze(1)
                value1 = feat1_batch.unsqueeze(1)
                attn2, _ = self.model.cross_attn(query2, key1, value1)
                attn2 = attn2.squeeze(1)

                combined = torch.cat([attn1, attn2], dim=1)
                scores = self.model.mlp(combined)
                probabilities = torch.sigmoid(scores).cpu().numpy().flatten()

                # 存储当前批次的概率
                all_probabilities.append(probabilities)

        # 合并所有批次的概率
        all_probabilities = np.concatenate(all_probabilities)

        # 获取top-k结果
        sorted_indices = np.argsort(all_probabilities)[::-1][:top_k]
        results = []

        for idx in sorted_indices:
            results.append({
                'path': self.paths[idx],
                'probability': all_probabilities[idx]
            })

        return results

    def match_image_efficient(self, img2, top_k=5):
        """
        更高效的匹配实现（如果GPU内存足够）
        """
        # 处理用户图片
        # img2 = Image.open(img2_path).convert("RGB")
        img2_tensor = self.transform(img2).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # 提取用户图片特征
            feat2 = self.model.backbone(img2_tensor)

            # 准备批量计算
            feat1_batch = torch.stack(self.features).to(self.device)
            feat2_batch = feat2.repeat(len(self.features), 1)

            # 批量计算相似度
            query1 = feat1_batch.unsqueeze(1)  # [N, 1, feat_dim]
            key2 = feat2_batch.unsqueeze(1)
            value2 = feat2_batch.unsqueeze(1)
            # print(query1.shape, key2.shape, value2.shape)
            # torch.Size([4, 1, 768]) torch.Size([4, 1, 768]) torch.Size([4, 1, 768])
            attn1, _ = self.model.cross_attn(query1, key2, value2)
            attn1 = attn1.squeeze(1)  # [N, feat_dim]

            query2 = feat2_batch.unsqueeze(1)
            key1 = feat1_batch.unsqueeze(1)
            value1 = feat1_batch.unsqueeze(1)
            attn2, _ = self.model.cross_attn(query2, key1, value1)
            attn2 = attn2.squeeze(1)

            combined = torch.cat([attn1, attn2], dim=1)
            scores = self.model.mlp(combined)
            probabilities = torch.sigmoid(scores).cpu().numpy().flatten()

            # 获取top-k结果
            sorted_indices = np.argsort(probabilities)[::-1][:top_k]
            results = []

            for idx in sorted_indices:
                results.append({
                    'path': self.paths[idx],
                    'probability': probabilities[idx]
                })

        return results


# 使用示例
def demo_gal(df):
    # 使用到的df如下格式：
    # df = pd.DataFrame({
    #     'path': ['image1.jpg', 'image2.jpg', ...],
    #     'dino': [feat1, feat2, ...]  # 每个特征应该是torch.Tensor或numpy数组
    # })

    # 初始化管理器
    manager = DFGalleryManager(
        model_path="../assets/checkpoint_epoch_14.pth",
        df=df
    )

    # 匹配用户图片
    user_image_path = "../search/110182236_p0_clip.png"
    input_img = read_image(user_image_path)

    # 方法1: 分批处理（适合大型图库）
    # top_matches = manager.match_image(user_image_path, top_k=5, batch_size=100)

    # 方法2: 高效处理（适合小型图库或GPU内存充足）
    import time
    start_time = time.time()
    top_matches = manager.match_image_efficient(user_image_path, top_k=5)
    end_time = time.time()
    print(f"[demo_gal] Total time: {end_time - start_time:.2f} seconds.")

    # 打印结果
    print("Top matches:")
    for i, match in enumerate(top_matches):
        print(f"{i + 1}. {match['path']} - Probability: {match['probability']:.4f}")
