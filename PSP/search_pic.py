import numpy as np
import cv2
import os
import pickle
import pandas as pd

from PSP.read_pic import imgs2df, read_image
from PSP.pic_util import HashPic
from PSP.util import ColoredText as CT


class SP:
    def __init__(self):
        self.df = None
        self.id_origin = []  # 匹配到的原图的id
        self.id_similar = []  # 匹配到的近似图片的id

    # 初始化图片数据，必选
    def init_pic_df(self, path_local=None, save_name=None):
        if path_local is None and save_name is None:
            raise ValueError("path_local和save_name不能同时为空")

        if save_name is None:
            # save_path 取 path_local 的最后一个文件夹名
            save_name = path_local.split('/')[-1]
        save_path = f"../data/{save_name}.pkl"
        # 检查save_path是否存在，如果存在就直接读取，否则就重新生成
        if os.path.exists(save_path):
            df = pd.read_pickle(save_path)
        else:
            df = imgs2df(path_local, save_path=save_path)
        # print(df.head(10))
        self.df = df
        print(f"从{save_path}初始化dataframe完成")

    # 搜索原图，先查验size，再逐个像素点比较
    def search_origin(self, input_img, nums=-1):
        """
        搜原图
        :param input_img: np.array，待搜索的图片数组
        :param nums: int，在本地图库中查询到多少个才停止，-1表示不提前停止，1表示一找到就停止
        :return: list，值为本地图库的path
        """
        self.id_origin = []  # 清空
        input_size = input_img.size
        input_shape = input_img.shape
        hp = HashPic()
        input_hash = hp.get_hash(input_img, "phash")  # np.array

        found_paths = []
        for index, row in self.df.iterrows():
            # 先比较size与shape
            if row['size'] == input_size and row['shape'] == input_shape and np.array_equal(row['hash'], input_hash):
                local_img_path = row['path']
                self.id_origin.append(row["id"])
                local_img = read_image(local_img_path, gray_pic=False, show_details=False)
                # 逐个像素点比较
                if np.array_equal(input_img, local_img):
                    found_paths.append(local_img_path)
                    if nums != -1 and len(found_paths) >= nums:
                        break
        return found_paths

    # 搜索近似图片，先查验phash，找出phash小于threshold(0.1)的
    def search_similar(self, input_img, threshold=0.1):
        """
        搜差不多的原图(允许小规模水印)
        :param input_img: np.array 待搜索的图片数组
        :param threshold: float，phash忍耐阈值，因为是64个值，0.1就是容忍6个点不同
        :return: list，值为本地图库的path
        """
        self.id_similar = []  # 清空
        hp = HashPic()
        input_hash = hp.get_hash(input_img, "phash")

        found_paths = []
        for index, row in self.df.iterrows():
            sim = hp.cal_hash_distance(input_hash, row["hash"])
            if sim < threshold:
                local_img_path = row['path']
                self.id_similar.append(row["id"])
                # 这里最好还比较一次原像素点，但是我开摆了
                found_paths.append(local_img_path)

        return found_paths




# 下面的代码正在重构，不要使用了
class BaseSearch:
    def __init__(self):
        # todo 这里可以加载配置文件

        # _search_img() 函数的参数
        # search_type='similar' 时的参数
        self.param_similar_atol = 10  # 绝对误差
        self.param_similar_rtol = 0.01  # 相对误差
        # search_type='mean' 时的参数
        self.param_mean_threshold = 20  # 均值差异阈值

        # __resize_img() 函数的参数
        self.param_wh_rate = 1.2  # 宽高比, w/h

        # 用字典存储每个目标图片对应的相似图片列表，key是目标图片target的绝对路径，value是本地相似图片source的绝对路径列表
        self.target2source = {}
        pass

    def search_imgs(self, target_dict, source_dict, **kwargs):
        """
        搜索文件夹下的所有图片
        :param target_dict: 目标图片数组字典
        :param source_dict: 本地图片数组字典
        :param kwargs: 其他参数，包括：
            search_type='similar': 搜索类型，包括 'strict', 'similar', 'local'
            local_type='surf': 局部搜索类型 (在 search_type='local' 时有效)，包括 'surf', 'orb', 'template'
        :return: self.target2source: 每个目标图片对应的相似图片列表，key是目标图片的绝对路径，value是本地相似图片的绝对路径列表
        """
        search_type = kwargs.get('search_type', 'similar')
        local_type = kwargs.get('local_type', 'surf')

        self.target2source = {}
        for i in target_dict.keys():
            self.target2source[i] = []

        # 对于target_dict中的每一张图片，都在source_dict中搜索是否存在相同的图片
        for target_k, target_v in target_dict.items():
            for source_k, source_v in source_dict.items():
                if self._search_img(target_v, source_v, search_type=search_type, local_type=local_type):
                    # print("找到相同图片：", target_k, '<----->', source_k)
                    self.target2source[target_k].append(source_k)  # 将找到的相似图片的地址存放在列表中
        return self.target2source

    def log_result(self):
        """
        打印搜索结果
        """
        print("☆" * 10)
        print("搜索完成，全部结果如下：")
        for target_k in self.target2source.keys():
            if self.target2source[target_k]:
                print(f"找到{len(self.target2source[target_k])}张与{target_k}相似的本地图片：{self.target2source[target_k]}")
            else:
                print(f"未找到{target_k}的相似图片")

    def log_param(self, specific_param=None):
        """
        打印类的参数
        :param specific_param: 指定的属性名
        """
        print(CT("类" + str(self.__class__.__name__) + "的参数如下：").pink())
        for attr in dir(self):
            if not callable(getattr(self, attr)) and not attr.startswith("__"):
                if specific_param is None or attr in specific_param:
                    value = getattr(self, attr)
                    print(f"{CT(attr).pink()}: {value}")

    def _search_img(self, target_img, source_img, search_type='strict', local_type='surf'):
        """
        对比两张图片是否相同
        :param target_img: 一张目标图片
        :param source_img: 一张本地图片
        :param search_type: 搜索类型，包括 'strict', 'similar', 'local'
        :param local_type: 局部搜索类型 (在 search_type='local' 时有效)，包括 'surf', 'orb', 'template'
        :return: 是否找到相同图片, True or False
        """
        # 严格搜索，需要两张图片完全相同
        if search_type == 'strict':
            if target_img.shape != source_img.shape:
                return False
            if np.all(target_img == source_img):
                return True
            else:
                return False

        # 相似搜索，允许两张图片有一定的差异
        # 1.要注意reshape之后可能像素差异会拉大，但因为两张图片清晰度可能不一样，不reshape不好比较
        # 2.allclose对图片的全部像素点要求比较严格，可以考虑使用mean
        elif search_type == 'similar':
            target_img, source_img = self.__resize_img(target_img, source_img)  # 将图片压缩到相同的大小
            # 如果两张图片的shape不同，则直接跳过（当二者的图片横宽比差异很大时，应该选择local方法）
            if target_img.shape != source_img.shape:
                print("两张图片的shape不同")
                return False
            # absolute(a - b) <= (atol + rtol * absolute(b))
            if np.allclose(target_img, source_img, atol=self.param_similar_atol, rtol=self.param_similar_rtol):
                return True
            else:
                return False

        # 均值搜索，比较两张图片的均值差异
        elif search_type == 'mean':
            target_mean = np.mean(target_img)
            source_mean = np.mean(source_img)
            if abs(target_mean - source_mean) <= self.param_mean_threshold:
                return True
            else:
                return False

        # 局部搜索，对于target_img(小图)，在source_img(大图)中搜索其中的一部分是否存在相同的图片
        elif search_type == 'local':
            if local_type == 'surf':
                # 使用SURF
                surf = cv2.SIFT_create()
                kp1, des1 = surf.detectAndCompute(target_img, None)
                kp2, des2 = surf.detectAndCompute(source_img, None)
                # 使用FLANN匹配器
                FLANN_INDEX_KDTREE = 0
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(des1, des2, k=2)
                # 比率测试，只保留良好匹配点
                good = []
                for m, n in matches:
                    if m.distance < 0.3 * n.distance:
                        good.append(m)
                if len(good) > 10:
                    return True
                else:
                    return False
            elif local_type == 'orb':
                # 使用ORB
                orb = cv2.ORB_create()
                kp1, des1 = orb.detectAndCompute(target_img, None)
                kp2, des2 = orb.detectAndCompute(source_img, None)
                # 使用BF匹配器
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)
                if len(matches) > 10:
                    return True
                else:
                    return False
            elif local_type == 'template':
                # 使用模板匹配
                if target_img.shape[0] > source_img.shape[0] or target_img.shape[1] > source_img.shape[1]:
                    # 目标图像的尺寸大于源图像的尺寸，需要调整目标图像的大小
                    target_img = cv2.resize(target_img, (source_img.shape[1], source_img.shape[0]))
                result = cv2.matchTemplate(source_img, target_img, cv2.TM_CCOEFF_NORMED)
                threshold = 0.8
                loc = np.where(result >= threshold)
                # 在使用“阿洛娜_局部.jpg”图片的情况时输出的是：
                # (array([0, 0, 0, 1], dtype=int64), array([548, 549, 550, 549], dtype=int64))
                # 第一个数组表示行索引，第二个数组表示列索引。
                # 因此，(0, 548)、(0, 549)、(0, 550) 和 (1, 549) 分别是匹配结果矩阵中的四个最大值所在的位置。
                # 相当于此处是在source_img中找到了target_img的位置，也就是图片的左上角坐标
                if len(loc[0]) > 0:
                    return True
                else:
                    return False
            else:
                print("请指定正确的local_type参数")
                return False
        else:
            print("请指定正确的search_type参数")
            return False

    def __resize_img(self, target_img, source_img):
        """
        依据图片是横屏还是竖屏还是近似正方形，将图片压缩到相同的大小
        [Warning]:
            该函数不会判断target_img和source_img在压缩后的大小是否相同，需要在调用该函数之后自行判断
        :param target_img:
        :param source_img:
        :return:
        """
        def get_new_size(img, rate=1.2):
            if rate < 1:
                rate = 1.2
            h, w = img.shape[:2]
            aspect_ratio = w / h
            if aspect_ratio > rate:  # 横屏
                return 300, 200
            elif aspect_ratio < 1/rate:  # 竖屏
                return 200, 300
            else:  # 近似正方形
                return 250, 250
        target_size = get_new_size(target_img, rate=self.param_wh_rate)
        source_size = get_new_size(source_img, rate=self.param_wh_rate)
        target_img = cv2.resize(target_img, target_size)
        source_img = cv2.resize(source_img, source_size)
        return target_img, source_img


# 搜索图片(传入原图)
class SearchPic(BaseSearch):
    def __init__(self):
        # todo 这里可以加载配置文件
        super().__init__()  # 调用父类的构造函数
        self.param_similar_atol = 10
        self.param_similar_rtol = 0.01
        pass


# 减小搜索域(传入压缩图片)
class DomainReducer(BaseSearch):
    def __init__(self):
        # todo 这里可以加载配置文件
        super().__init__()
        self.param_similar_atol = 50
        self.param_similar_rtol = 0.01

    def get_dr_dict(self, target_dict, source_dict, target_cp, source_cp, search_type='mean', local_type='surf'):
        """
        获取压缩后的搜索域图片
        :param target_dict: 目标图片字典，key是图片的绝对路径，value是图片数组
        :param source_dict: 本地图片字典，key是图片的绝对路径，value是图片数组
        :param dr_type: 压缩类型，包括 '1px', '9px'
        :param search_type: 搜索类型，包括 'strict', 'similar', 'local'
        :param local_type: 局部搜索类型 (在 search_type='local' 时有效)，包括 'surf', 'orb', 'template'
        :param kwargs: 其他参数
        :return: imgs_dr: 压缩后的搜索域图片，key是图片的绝对路径，value是图片数组
        """
        print("----------减小搜索域---------")
        imgs_s_p = self.search_imgs(target_cp, source_cp, search_type=search_type, local_type=local_type)
        len_1 = len(list(imgs_s_p.values())[0])
        len_2 = len(list(source_dict.values()))
        if len_2 == 0:
            print("本地图片为空")
        else:
            print(f"压缩比例：{len_1} / {len_2} = {len_1 / len_2}")


        # target_dict是目标图片，k是路径，v是数组。source_dict是本地图片，k是路径，v是数组
        # imgs_s_p记录新的搜索域，k是目标图片路径，value是一个list(里面的元素是待搜索的图片路径)
        # 首先根据target_dict的key查找imgs_t_p的k，然后根据imgs_s_p的v查找target_dict的k，将这些k存入imgs_dr[target_key]
        # 最后根据target_dict的key查找target_dict的v，将这些v存入imgs_t_p_origin
        # 因此imgs_dr是一个字典的字典，k是目标图片路径，v是一个字典(k是本地图片路径，v是图片数组)。
        imgs_dr = {}
        for target_key in target_dict.keys():
            target_key = os.path.normpath(target_key)  # normpath使得路径标准化，比如将'\\'转换为'/'
            # 查找 imgs_s_p 中是否存在 target_key
            if target_key in map(os.path.normpath, imgs_s_p.keys()):
                search_paths = imgs_s_p[target_key]
                if target_key not in imgs_dr:
                    imgs_dr[target_key] = {}  # 初始化每个target_key对应的搜索域字典
                for search_path in search_paths:
                    search_path = os.path.normpath(search_path)
                    for key, value in source_dict.items():
                        if os.path.normpath(key) == search_path:
                            imgs_dr[target_key][key] = value
        # 返回imgs_dr的value，todo 这里将来应该重构，只让target_dict输入一张图片

        return list(imgs_dr.values())[0]
        # return imgs_dr


# 压缩图片信息
class CompressPic:
    def __init__(self, **kwargs):
        img_1pixel = kwargs.get('img_1px', "img_1px")
        img_9pixel = kwargs.get('img_9px', "img_9px")
        cp_path = kwargs.get('cp_path', "data/compressed_imgs")

        # todo 这里可以加载配置文件

        # self.img_1px = self.compress_imgs(self.imgs_dict, resize=(1, 1), path=cp_path, name=img_1pixel)  # 把图片变成1个像素点
        # self.img_9px = self.compress_imgs(self.imgs_dict, resize=(3, 3), path=cp_path, name=img_9pixel)  # 把图片变成9个像素点

        pass

    def compress_imgs(self, imgs_dict, resize, path="data/compressed_imgs", name="compressed"):
        """
        压缩文件夹下的所有图片，并保存压缩后的图片字典
        :param imgs_dict: 图像数组字典，key是图片的绝对路径，value是图片数组
        :param resize: 压缩比例 或者 压缩后的大小(h, w)
        :param path: 压缩后的图片存放路径
        :param name: 压缩后的图片的名称
        :return: 压缩后的图片数组字典，key是图片的绝对路径，value是图片数组
        """
        # 检查path, name + ".pkl"文件是否存在，如果存在则直接读取
        if os.path.exists(os.path.join(path, name + ".pkl")):
            return self._load_compressed_imgs_dict(os.path.join(path, name + ".pkl"))
        else:
            compressed_imgs_dict = {}  # 存放压缩后的所有图片，key是图片的绝对路径，value是图片数组
            for k, v in imgs_dict.items():
                compressed_imgs_dict[k] = self._compress_img(v, resize)

            # 保存compressed_imgs_dict
            if not os.path.exists(path):
                os.makedirs(path)
            self._save_compressed_imgs_dict(compressed_imgs_dict, os.path.join(path, name + ".pkl"))
            return compressed_imgs_dict

    def _compress_img(self, img, resize):
        """
        压缩图片
        :param img: 图片数组
        :param resize: 压缩比例rate 或者 压缩后的大小(h, w)
        :return: 压缩后的图片数组
        """
        # 如果resize是二元组
        if isinstance(resize, tuple):
            img = cv2.resize(img, (resize[1], resize[0]))
            return img
        # 如果resize是单个数值
        elif isinstance(resize, (int, float)):
            h, w = img.shape[:2]
            img = cv2.resize(img, (int(w * resize), int(h * resize)))
            return img
        else:
            print("resize参数输入有误")
            return False

    def _save_compressed_imgs_dict(self, compressed_imgs_dict, path):
        with open(path, 'wb') as file:
            pickle.dump(compressed_imgs_dict, file)

    def _load_compressed_imgs_dict(self, path):
        with open(path, 'rb') as file:
            compressed_imgs_dict = pickle.load(file)
        return compressed_imgs_dict


    def _add_compressed_imgs_dict(self, compressed_imgs_dict, path):
        with open(path, 'ab') as file:
            pickle.dump(compressed_imgs_dict, file)
