# from Tools.search_pic import demo

# original_path = "F:/Picture/pixiv/BA/110182236_p0.jpg"
# probability = demo(original_path)
# print(f"Probability that crop is from original: {probability:.4f}")

import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import pickle
from tqdm import tqdm
import time

from Tools.local_matcher import LocalMatcher


class GalleryManager:
    def __init__(self, model_path, gallery_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gallery_path = gallery_path

        # 加载完整模型
        self.model = LocalMatcher().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # 图库特征存储
        self.gallery_features = {}
        self.image_paths = []

        # 创建特征存储目录
        os.makedirs(self.gallery_path, exist_ok=True)

    def precompute_gallery_features(self, image_dir):
        """预计算图库中所有图片的特征"""
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for img_file in tqdm(image_files, desc="Precomputing gallery features"):
            img_path = os.path.join(image_dir, img_file)
            feature_path = os.path.join(self.gallery_path, f"{os.path.splitext(img_file)[0]}.pkl")

            # 如果特征已存在则跳过
            if os.path.exists(feature_path):
                continue

            # 提取并保存特征
            img = Image.open(img_path).convert("RGB")
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                feat = self.model.backbone(img_tensor)

            # 保存特征和元数据
            feature_data = {
                'feature': feat.cpu(),
                'path': img_path
            }

            with open(feature_path, 'wb') as f:
                pickle.dump(feature_data, f)

    def load_gallery(self):
        """加载所有预计算的特征到内存"""
        self.gallery_features = {}
        self.image_paths = []

        feature_files = [f for f in os.listdir(self.gallery_path) if f.endswith('.pkl')]

        for feat_file in tqdm(feature_files, desc="Loading gallery features"):
            feat_path = os.path.join(self.gallery_path, feat_file)

            with open(feat_path, 'rb') as f:
                feature_data = pickle.load(f)

            self.gallery_features[feature_data['path']] = feature_data['feature']
            self.image_paths.append(feature_data['path'])

    def match_image(self, img2_path, top_k=5):
        """匹配用户上传的图片与图库"""
        # 处理用户图片
        img2 = Image.open(img2_path).convert("RGB")
        img2_tensor = self.transform(img2).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # 提取用户图片特征
            feat2 = self.model.backbone(img2_tensor)

            # 准备批量计算
            all_feat1 = []
            for path in self.image_paths:
                all_feat1.append(self.gallery_features[path])

            # start_time = time.time()
            # 转换为批量张量
            feat1_batch = torch.cat(all_feat1, dim=0).to(self.device)
            feat2_batch = feat2.repeat(len(all_feat1), 1)

            # 批量计算相似度
            query1 = feat1_batch.unsqueeze(1)  # [N, 1, feat_dim]
            key2 = feat2_batch.unsqueeze(1)
            value2 = feat2_batch.unsqueeze(1)
            print(query1.shape, key2.shape, value2.shape)
            # torch.Size([427, 1, 768]) torch.Size([427, 1, 768]) torch.Size([427, 1, 768])
            attn1, _ = self.model.cross_attn(query1, key2, value2)
            attn1 = attn1.squeeze(1)  # [N, feat_dim]

            query2 = feat2_batch.unsqueeze(1)
            key1 = feat1_batch.unsqueeze(1)
            value1 = feat1_batch.unsqueeze(1)
            attn2, _ = self.model.cross_attn(query2, key1, value1)
            attn2 = attn2.squeeze(1)

            combined = torch.cat([attn1, attn2], dim=1)
            scores = self.model.mlp(combined)
            probabilities = torch.sigmoid(scores).cpu().numpy().flatten()  # [batch_size, 1] -> [batch_size]
            print(probabilities)
            # end_time = time.time()
            # print(f"[match_image] {end_time - start_time:.2f} seconds.")

        # 获取top-k结果
        sorted_indices = np.argsort(probabilities)[::-1][:top_k]
        print(sorted_indices)
        results = []

        for idx in sorted_indices:
            results.append({
                'path': self.image_paths[idx],
                'probability': probabilities[idx]
            })

        return results


if __name__ == "__main__":
    # 初始化图库管理器
    gallery_manager = GalleryManager(
        model_path="../assets/checkpoint_epoch_14.pth",
        gallery_path="../assets/gallery_features"
    )

    # 预计算图库特征（只需运行一次）
    gallery_manager.precompute_gallery_features("F:/Picture/pixiv/BA")
    # gallery_manager.precompute_gallery_features("F:/Picture/pixiv")
    # 加载图库特征到内存
    gallery_manager.load_gallery()

    # 用户上传图片进行匹配
    user_image_path = "F:/Picture/pixiv/BA/110182236_p0.jpg"

    top_matches = gallery_manager.match_image(user_image_path, top_k=5)

    # 打印结果
    print("Top matches:")
    for i, match in enumerate(top_matches):
        print(f"{i + 1}. {match['path']} - Probability: {match['probability']:.4f}")
