import numpy as np
import os
import pandas as pd

from Tools.read_pic import imgs2df, read_image
from Tools.pic_util import HashPic
from Tools.local_manager import DFGalleryManager


class SP:
    def __init__(self):
        self.df = None
        # self.id_origin = []  # 匹配到的原图的id
        # self.id_similar = []  # 匹配到的近似图片的id
        self.id_result = []  # 匹配到的图片(原图/近似)的id
        self.manager = None  # DFGalleryManager实例，用于局部图片搜索

    # 初始化图片数据，必做
    def init_pic_df(self, path_local=None, save_name=None, save_path=None, log_callback=None):
        """
        初始化图片数据
        :param path_local: 本地图库路径
        :param save_name: 保存的模型名，默认是f"../assets/{save_name}.pkl"
        :param save_path: 主动指定保存模型的完整路径
        :param log_callback: 日志回调函数，用于将日志传回到QT里去
        :return:
        """
        if path_local is None and save_name is None and save_path is None:
            raise ValueError("path_local和save_name和save_path不能同时为空")
        if save_path is None:
            if save_name is None:
                # save_path 取 path_local 的最后一个文件夹名
                save_name = path_local.split('/')[-1]
            save_path = f"../assets/{save_name}.pkl"
        # 检查save_path是否存在，如果存在就直接读取，否则就重新生成
        if os.path.exists(save_path):
            df = pd.read_pickle(save_path)
        else:
            print(f"[init_pic_df] {save_path}无模型文件，正在重新生成")
            df = imgs2df(path_local, save_path=save_path, log_callback=log_callback)
        # print(df.head(20))
        self.df = df
        print(f"[init_pic_df] 从{save_path}初始化dataframe完成")

    # 搜索原图，先查验size，再逐个像素点比较
    def search_origin(self, input_img, max_result=-1):
        """
        搜原图，也就是查找只有文件名不同的图
        :param input_img: np.array，待搜索的图片数组
        :param max_result: int，在本地图库中查询到多少个才停止，-1表示不提前停止，1表示一找到就停止
        :return: list，值为本地图库的path
        """
        # self.id_origin = []  # 清空
        input_size = input_img.size
        input_shape = input_img.shape
        hp = HashPic()
        input_hash = hp.get_hash(input_img, "phash")  # np.array
        # print(f"[search_origin] input_size: {input_size}, input_shape: {input_shape} input_mean: {np.mean(input_img)}")

        found_paths = []
        for index, row in self.df.iterrows():
            # 先比较size与shape
            if row['size'] == input_size and row['shape'] == input_shape and np.array_equal(row['hash'], input_hash):
                # print(f"\r[search_origin] 匹配到 {row['path']}", end=' ')
                local_img_path = row['path']
                # self.id_origin.append(row["id"])
                local_img = read_image(local_img_path, gray_pic=False, show_details=False)
                # 逐个像素点比较
                if np.array_equal(input_img, local_img):
                    self.id_result.append(row["id"])  # 记录匹配到的图片的id
                    found_paths.append(local_img_path)
                    if max_result != -1 and len(found_paths) >= max_result:
                        break
        return found_paths

    # 搜索近似图片(不支持局部搜索)，先查验phash，找出phash小于hash_threshold的
    def search_similar(self, input_img, hash_threshold=0.2, mean_threshold=40):
        """
        搜差不多的原图(允许小规模水印)
        :param input_img: np.array 待搜索的图片数组
        :param hash_threshold: float，phash忍耐阈值，因为是64个值，0.1就是容忍6个点不同
        :param mean_threshold: int，像素均值忍耐阈值
        :return: list，值为本地图库的path
        """
        # self.id_similar = []  # 清空
        hp = HashPic()
        input_hash = hp.get_hash(input_img, "phash")
        input_mean = np.mean(input_img)

        found_paths_with_sim = []
        # print(self.df)
        for index, row in self.df.iterrows():
            sim = hp.cal_hash_distance(input_hash, row["hash"])
            if sim < hash_threshold:
                local_mean = row["mean"]
                # print("\r[search_similar] input_mean:", input_mean, "local_mean:", local_mean, end=' ')
                # print(input_mean - local_mean)
                if abs(input_mean - local_mean) > mean_threshold:
                    # print(f"\r[search_similar] 忽略 {row['path']}，均值差异过大{input_mean} vs {local_mean}", end=' ')
                    continue
                local_img_path = row['path']
                # self.id_similar.append(row["id"])
                self.id_result.append(row["id"])  # 记录匹配到的图片的id，如果后续需要，可以用这个id去df里取数据
                found_paths_with_sim.append((sim, local_img_path))
        # 根据 sim 对 found_paths_with_sim 进行排序
        found_paths_with_sim.sort(key=lambda x: x[0])
        # 提取排序后的路径
        found_paths = [path for sim, path in found_paths_with_sim]
        return found_paths

    # 搜索局部图片
    def search_partial(self, input_img, top_k=5, dinov2_path="../assets/checkpoint_epoch_14.pth"):
        if self.manager is None:
            self.manager = DFGalleryManager(
                model_path=dinov2_path,
                df=self.df
            )
        # print(f"[search_partial] 进行局部图片搜索")
        top_matches = self.manager.match_image_efficient(input_img, top_k=top_k)  # 含有path和probability
        found_paths = [match['path'] for match in top_matches]
        return found_paths

# def demo(original_path, crop_size=224):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = LocalMatcher().to(device)  # 需要定义LocalMatcher类
#     model.load_state_dict(torch.load("../assets/checkpoint_epoch_14.pth", map_location=device))
#     model.eval()
#
#     original_img = Image.open(original_path).convert("RGB")
#
#     transform = transforms.Compose([
#         transforms.Resize((400, 400)),  # 先调整到稍大尺寸
#         transforms.RandomCrop(crop_size),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#
#     crop_img = transform(original_img).unsqueeze(0).to(device)  # 添加批次维度
#
#     original_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     original_input = original_transform(original_img).unsqueeze(0).to(device)
#
#     import time
#     start_time = time.time()
#     with torch.no_grad():
#         # print(f"original_input shape: {original_input.shape}, crop_img shape: {crop_img.shape}")
#         output = model(original_input, crop_img)
#         probability = torch.sigmoid(output).item()
#     end_time = time.time()
#     print(f"[demo] 推理时间: {end_time - start_time:.4f}秒")
#
#     fig, ax = plt.subplots(1, 2, figsize=(12, 6))
#
#     ax[0].imshow(original_img)
#     ax[0].set_title(f"Original Image\n{original_path}")
#     ax[0].axis('off')
#
#     crop_display = crop_img.squeeze(0).cpu().permute(1, 2, 0).numpy()
#     crop_display = crop_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
#     crop_display = np.clip(crop_display, 0, 1)
#     ax[1].imshow(crop_display)
#     ax[1].set_title(f"Cropped Region\nProbability: {probability:.4f}")
#     ax[1].axis('off')
#
#     result = "Same Image" if probability > 0.5 else "Different Image"
#     plt.suptitle(f"Prediction: {result} (Confidence: {probability:.4f})", fontsize=16)
#
#     plt.tight_layout()
#     plt.show()
#
#     return probability
