import os
import torch
import numpy as np
import cv2
import faiss
import pickle
from tqdm import tqdm

from src.encode.dinov2_encoder import DINOv2Encoder
from src.utils.img_read import read_image


# 假设 read_image 已经定义在你的环境中
# from src.utils.img_read import read_image

class DINOv2FAISSManager:
    # gallery_index.faiss：存的是 FAISS 向量索引本体（向量数据 + 索引结构）。
    # gallery_index.faiss.pkl：存的是 路径映射表（Python 列表），把“索引里的第 i 个向量”对应回“第 i 张图片的文件路径”。
    def __init__(self, encoder: DINOv2Encoder):
        self.encoder = encoder  # 直接注入已经实例化的 encoder
        self.index = None
        self.path_mapping = []

    def build_index(self, folder_path, save_path="gallery_index.faiss"):
        """扫描文件夹并构建索引"""
        img_paths = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    abs_path = os.path.abspath(os.path.join(root, file)).replace('\\', '/')
                    img_paths.append(abs_path)

        print(f"找到 {len(img_paths)} 张图片，开始提取特征...")

        all_feats = []
        # 这里的大 batch 是为了减少磁盘读取次数，小 batch 是为了不炸显存
        read_batch_size = 512
        for i in tqdm(range(0, len(img_paths), read_batch_size)):
            chunk_paths = img_paths[i: i + read_batch_size]
            # 读取并过滤无效图片
            chunk_imgs = []
            valid_paths = []
            for p in chunk_paths:
                img = read_image(p)
                if img is not None:
                    chunk_imgs.append(img)
                    valid_paths.append(p)

            if not chunk_imgs:
                continue

            # 直接调用 encoder 的接口
            feats = self.encoder.encode(chunk_imgs, max_batch_size=256)

            # 重要：FAISS 相似度计算前通常需要 L2 归一化
            faiss.normalize_L2(feats)
            all_feats.append(feats)
            self.path_mapping.extend(valid_paths)

        final_feats = np.concatenate(all_feats, axis=0)

        # 初始化 FAISS 索引 (Inner Product + 归一化 = 余弦相似度)
        self.index = faiss.IndexFlatIP(final_feats.shape[1])
        self.index.add(final_feats)

        # 保存
        faiss.write_index(self.index, save_path)
        with open(save_path + ".pkl", "wb") as f:
            pickle.dump(self.path_mapping, f)
        print(f"成功构建并保存索引，共 {len(self.path_mapping)} 张图")

    def load_index(self, index_path="gallery_index.faiss"):
        self.index = faiss.read_index(index_path)
        with open(index_path + ".pkl", "rb") as f:
            self.path_mapping = pickle.load(f)
        print(f"索引已加载，库内图片数: {len(self.path_mapping)}")

    def search(self, query_img, k=5):
        """支持传入路径或 numpy 数组进行搜索"""
        if isinstance(query_img, str):
            query_img = read_image(query_img)

        # 1. 提取并归一化特征
        feat = self.encoder.encode([query_img])  # 得到 (1, D)
        faiss.normalize_L2(feat)

        # 2. 检索
        distances, indices = self.index.search(feat, k)

        # 3. 返回结果
        return [{"path": self.path_mapping[idx], "score": float(distances[0][i])}
                for i, idx in enumerate(indices[0]) if idx != -1]


# --- 使用示例 ---
if __name__ == "__main__":
    gallery_dir = "F:/Picture/pixiv/LuoTianyi"
    # query_image_path = "F:/Picture/pixiv/BA/Shiroko/140776508_p0.png"  # 你想用来搜的图
    query_image_path = "F:/Picture/pixiv/LuoTianyi/澪漉llu/76759452_p3.jpg"  # 你想用来搜的图
    encoder = DINOv2Encoder()
    manager = DINOv2FAISSManager(encoder)

    # 第一次运行：构建索引
    manager.build_index(gallery_dir)

    # 后续运行：直接加载
    if os.path.exists("gallery_index.faiss"):
        manager.load_index("gallery_index.faiss")

        # 执行检索
        print(f"\n正在检索相似图片...")
        import time

        start_time = time.time()
        matches = manager.search(query_image_path, k=5)
        print(time.time() - start_time)

        for i, res in enumerate(matches):
            print(f"Top-{i + 1}: Score={res['score']:.4f} | Path={res['path']}")
