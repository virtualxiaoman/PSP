import numpy as np
import os
import pandas as pd

from Tools.read_pic import imgs2df, read_image
from Tools.pic_util import HashPic


class SP:
    def __init__(self):
        self.df = None
        # self.id_origin = []  # 匹配到的原图的id
        # self.id_similar = []  # 匹配到的近似图片的id
        self.id_result = []  # 匹配到的图片(原图/近似)的id

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

        found_paths = []
        for index, row in self.df.iterrows():
            # 先比较size与shape
            if row['size'] == input_size and row['shape'] == input_shape and np.array_equal(row['hash'], input_hash):
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

    # 搜索近似图片(不支持局部搜索)，先查验phash，找出phash小于threshold(0.1)的
    def search_similar(self, input_img, hash_threshold=0.2, mean_threshold=20):
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

        found_paths_with_sim = []
        for index, row in self.df.iterrows():
            sim = hp.cal_hash_distance(input_hash, row["hash"])
            if sim < hash_threshold:
                input_mean = np.mean(input_img)
                local_mean = row["mean"]
                if abs(input_mean - local_mean) > mean_threshold:
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
