import os
import torch
import numpy as np
import cv2
import faiss
import pickle
import json
import hashlib
from tqdm import tqdm
import time

from src.config.supported import SUPPORTED_IMG_FORMAT
from src.encode.dinov2_encoder import DINOv2Encoder
from src.utils.img_read import read_image
from src.config.path import MODEL_DINOV2_DIR


class DINOv2FAISSManager:
    # gallery_index.faiss：存的是 FAISS 向量索引本体（向量数据 + 索引结构）。
    # gallery_index.faiss.pkl：存的是 路径映射表（Python 列表），把“索引里的第 i 个向量”对应回“第 i 张图片的文件路径”。
    def __init__(self, encoder: DINOv2Encoder):
        self.encoder = encoder  # 直接注入已经实例化的 encoder
        self.index = None
        self.path_mapping = []

        # 确保模型存放目录存在
        os.makedirs(MODEL_DINOV2_DIR, exist_ok=True)
        # 固定状态记录文件路径
        self.manifest_path = os.path.join(MODEL_DINOV2_DIR, "gallery_manifest.json")

    def _get_gallery_hash(self, folder_path):
        """计算图库状态的 MD5 哈希值（包含所有图片路径和最后修改时间）"""
        file_info_list = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(SUPPORTED_IMG_FORMAT):
                    abs_path = os.path.join(root, file)
                    mtime = os.path.getmtime(abs_path)  # 获取文件最后修改时间
                    file_info_list.append(f"{abs_path}_{mtime}")

        # 排序保证哈希的稳定性
        file_info_list.sort()
        # 计算 MD5
        hash_md5 = hashlib.md5()
        hash_md5.update("".join(file_info_list).encode('utf-8'))
        return hash_md5.hexdigest()

    def _load_manifest(self):
        """加载状态清单文件"""
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"读取 manifest 文件{self.manifest_path} 失败: {e}")
                return {}
        return {}

    def _save_manifest(self, manifest_data):
        """保存状态清单文件"""
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=4, ensure_ascii=False)

    def auto_load_or_build(self, folder_path):
        """自动判断是加载已有索引还是重新构建"""
        # 提取最后一级文件夹名作为 gallery_name，例如 "LuoTianyi" 或 "Shiroko"
        gallery_name = os.path.basename(os.path.normpath(folder_path))
        if not gallery_name:
            raise ValueError("无效的图库路径")

        # 构建对应的模型保存路径
        faiss_save_path = os.path.join(MODEL_DINOV2_DIR, f"{gallery_name}.faiss")
        pkl_save_path = os.path.join(MODEL_DINOV2_DIR, f"{gallery_name}.pkl")

        print(f"正在检查图库状态 [{gallery_name}]...")
        current_hash = self._get_gallery_hash(folder_path)
        manifest = self._load_manifest()

        # 判断是否需要重新构建：
        # 1. manifest里没有记录
        # 2. 记录的哈希值与当前不一致（图片有增删改）
        # 3. 实际的 .faiss 或 .pkl 文件丢失
        needs_build = True
        if gallery_name in manifest:
            saved_info = manifest[gallery_name]
            if saved_info.get("hash") == current_hash and \
                    os.path.exists(faiss_save_path) and \
                    os.path.exists(pkl_save_path):
                needs_build = False

        if needs_build:
            print(f"图库 [{gallery_name}] 有更新或索引不存在，开始构建...")
            self.build_index(folder_path, faiss_save_path, pkl_save_path)

            # 构建成功后更新并保存 manifest
            manifest[gallery_name] = {
                "hash": current_hash,
                "faiss_path": faiss_save_path.replace('\\', '/'),
                "pkl_path": pkl_save_path.replace('\\', '/')
            }
            self._save_manifest(manifest)
        else:
            print(f"图库 [{gallery_name}] 状态未改变，直接加载最新缓存...")
            self.load_index(faiss_save_path, pkl_save_path)

    def build_index(self, folder_path, faiss_save_path, pkl_save_path):
        """扫描文件夹并构建索引"""
        img_paths = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(SUPPORTED_IMG_FORMAT):
                    abs_path = os.path.abspath(os.path.join(root, file)).replace('\\', '/')
                    img_paths.append(abs_path)

        print(f"找到 {len(img_paths)} 张图片，开始提取特征...")

        all_feats = []
        # 这里的大 batch 是为了减少磁盘读取次数，小 batch 是为了不炸显存
        read_batch_size = 512
        self.path_mapping = []  # 重置 path_mapping 防止追加

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
        faiss.write_index(self.index, faiss_save_path)
        with open(pkl_save_path, "wb") as f:
            pickle.dump(self.path_mapping, f)
        print(f"成功构建并保存索引至: {faiss_save_path}，共 {len(self.path_mapping)} 张图")

    def load_index(self, faiss_path, pkl_path):
        """加载已有的索引文件"""
        self.index = faiss.read_index(faiss_path)
        with open(pkl_path, "rb") as f:
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
    # 你可以随时切换这两个图库，系统会自动管理各自的模型状态
    gallery_dir = "F:/Picture/pixiv/LuoTianyi"
    # gallery_dir = "F:/Picture/pixiv/BA/Shiroko"

    query_image_path = "F:/Picture/pixiv/LuoTianyi/澪漉llu/76759452_p3.jpg"

    encoder = DINOv2Encoder()
    manager = DINOv2FAISSManager(encoder)

    # 核心调用：一句代码自动处理所有逻辑（判断是否需要重新生成、管理路径等）
    manager.auto_load_or_build(gallery_dir)

    # 执行检索
    print(f"\n正在检索相似图片...")

    start_time = time.time()
    matches = manager.search(query_image_path, k=5)
    print(f"检索耗时: {time.time() - start_time:.4f} 秒")

    for i, res in enumerate(matches):
        print(f"Top-{i + 1}: Score={res['score']:.4f} | Path={res['path']}")
