import os
import torch
import numpy as np
import cv2
import faiss
import pickle
from tqdm import tqdm


# 假设 read_image 已经定义在你的环境中
# from src.utils.img_read import read_image

class DINOv2FAISSManager:
    def __init__(self, model_name="dinov2_vitl14", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"载入 DINOv2 [{model_name}] 到 {self.device}...")
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval().to(self.device)

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        self.index = None
        self.path_mapping = []  # 存储 ID -> 绝对路径的映射

    def _preprocess(self, img: np.ndarray):
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        return np.transpose(img, (2, 0, 1))

    @torch.no_grad()
    def extract_features(self, img_list, max_batch_size=256):
        """分块提取特征并归一化（为了余弦相似度）"""
        all_features = []
        for i in range(0, len(img_list), max_batch_size):
            batch = img_list[i: i + max_batch_size]
            processed = [self._preprocess(img) for img in batch]
            x = torch.from_numpy(np.stack(processed)).float().to(self.device)

            feats = self.model(x).cpu().numpy()
            # 归一化特征，这样后续使用 FAISS 的 Inner Product 索引等同于余弦相似度
            faiss.normalize_L2(feats)
            all_features.append(feats)

        return np.concatenate(all_features, axis=0)

    def build_index(self, folder_path, save_path="gallery_index.faiss"):
        """扫描文件夹，提取特征并构建 FAISS 索引"""
        img_paths = []
        raw_imgs = []

        print(f"正在扫描文件夹: {folder_path}")
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    abs_path = os.path.abspath(os.path.join(root, file)).replace('\\', '/')
                    img_paths.append(abs_path)

        print(f"找到 {len(img_paths)} 张图片，开始提取特征...")

        # 为了防止内存溢出，我们小批量读取原图并提取特征
        all_feats = []
        read_batch_size = 512  # 这里是读取图片的 batch
        for i in tqdm(range(0, len(img_paths), read_batch_size)):
            chunk_paths = img_paths[i: i + read_batch_size]
            chunk_imgs = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in chunk_paths]
            # 过滤读取失败的图片
            valid_imgs = [img for img in chunk_imgs if img is not None]

            if not valid_imgs: continue

            feats = self.extract_features(valid_imgs, max_batch_size=256)
            all_feats.append(feats)

        final_feats = np.concatenate(all_feats, axis=0)
        dim = final_feats.shape[1]

        # 构建 FAISS 索引 (使用 FlatL2 或 FlatIP)
        # 因为我们做了 L2 归一化，使用 IndexFlatIP (内积) 实际上就是计算余弦相似度
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(final_feats)
        self.path_mapping = img_paths

        # 保存索引和路径映射
        faiss.write_index(self.index, save_path)
        with open(save_path + ".pkl", "wb") as f:
            pickle.dump(self.path_mapping, f)
        print(f"索引构建完成，已保存至 {save_path}")

    def load_index(self, index_path="gallery_index.faiss"):
        self.index = faiss.read_index(index_path)
        with open(index_path + ".pkl", "rb") as f:
            self.path_mapping = pickle.load(f)
        print(f"已加载索引，包含 {len(self.path_mapping)} 条数据")

    def search(self, query_img, k=5):
        """输入一张图，检索最相似的 k 张"""
        if self.index is None:
            raise ValueError("请先构建或加载索引")

        # 1. 提取查询图特征
        if isinstance(query_img, str):  # 如果传的是路径
            query_img = cv2.cvtColor(cv2.imread(query_img), cv2.COLOR_BGR2RGB)

        feat = self.extract_features([query_img])  # (1, D)

        # 2. 检索
        distances, indices = self.index.search(feat, k)

        # 3. 解析结果
        results = []
        for i in range(k):
            idx = indices[0][i]
            if idx != -1:
                results.append({
                    "path": self.path_mapping[idx],
                    "score": float(distances[0][i])
                })
        return results


# --- 使用示例 ---
if __name__ == "__main__":
    gallery_dir = "F:/Picture/pixiv/BA/Shiroko"
    query_image_path = "F:/Picture/pixiv/BA/Shiroko/140776508_p0.png"  # 你想用来搜的图

    manager = DINOv2FAISSManager()

    # 第一次运行：构建索引
    manager.build_index(gallery_dir)

    # 后续运行：直接加载
    if os.path.exists("gallery_index.faiss"):
        manager.load_index("gallery_index.faiss")

        # 执行检索
        print(f"\n正在检索相似图片...")
        matches = manager.search(query_image_path, k=5)

        for i, res in enumerate(matches):
            print(f"Top-{i + 1}: Score={res['score']:.4f} | Path={res['path']}")