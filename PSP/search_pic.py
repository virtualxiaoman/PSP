import numpy as np
import cv2

from PSP.util import ColoredText as CT


class BaseSearch:
    def __init__(self):
        # todo 这里可以加载配置文件

        # _search_img() 函数的参数
        # search_type='similar' 时的参数
        self.param_similar_atol = 10  # 绝对误差
        self.param_similar_rtol = 0.01  # 相对误差

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
        :return:
        """
        search_type = kwargs.get('search_type', 'similar')
        local_type = kwargs.get('local_type', 'surf')

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
            # 如果两张图片的shape不同，则直接跳过
            if target_img.shape != source_img.shape:
                return
            # 如果两张图片的shape相同，则进一步比较两张图片是否相同
            if np.all(target_img == source_img):
                return True
            else:
                return False

        # 相似搜索，允许两张图片有一定的差异
        elif search_type == 'similar':
            target_img, source_img = self.__resize_img(target_img, source_img)  # 将图片压缩到相同的大小
            # 如果两张图片的shape不同，则直接跳过（此时说明应该选择local方法）
            if target_img.shape != source_img.shape:
                return False
            # absolute(a - b) <= (atol + rtol * absolute(b))
            if np.allclose(target_img, source_img, atol=self.param_similar_atol, rtol=self.param_similar_rtol):
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
        self.param_similar_atol = 30
        self.param_similar_rtol = 0.01
        pass




# 压缩图片信息
class CompressPic:
    def __init__(self, imgs_dict):
        # todo 这里可以加载配置文件
        self.imgs_dict = imgs_dict
        # 把图片变成一个像素点
        self.img_1pixel = self.compress_imgs(self.imgs_dict, resize=(1, 1))
        pass

    def compress_imgs(self, imgs_dict, resize):
        """
        压缩文件夹下的所有图片
        :param imgs_dict: 图像数组字典，key是图片的绝对路径，value是图片数组
        :param resize: 压缩比例 或者 压缩后的大小(h, w)
        :return: 压缩后的图片数组字典
        """
        compressed_imgs_dict = {}  # 存放压缩后的所有图片，key是图片的绝对路径，value是图片数组
        for k, v in imgs_dict.items():
            compressed_imgs_dict[k] = self._compress_img(v, resize)
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


