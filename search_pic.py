import numpy as np
import cv2


# todo 待重构
# # base_search.py
# class BaseSearch:
#     def _search_img(self, image, search_params):
#         pass
# 然后让SearchPic 与 DomainReducer 继承 BaseSearch
# class SearchPic(BaseSearch):
#     剩下的跟之前一样


# 搜索图片(传入原图)
class SearchPic:
    def __init__(self):
        # todo 这里可以加载配置文件
        pass

    def search_imgs(self, target_dict, source_dict):
        """
        搜索文件夹下的所有图片
        :param target_dict: 目标图片数组字典
        :param source_dict: 本地图片数组字典
        :return:
        """
        source_imgs = {}  # 用字典存储每个目标图片对应的相似图片列表
        for i in target_dict.keys():
            source_imgs[i] = []

        # 对于target_dict中的每一张图片，都在source_dict中搜索是否存在相同的图片
        for target_k, target_v in target_dict.items():
            for source_k, source_v in source_dict.items():
                if self._search_img(target_v, source_v, search_type='local'):
                    print("找到相同图片：", target_k, '<----->', source_k)
                    source_imgs[target_k].append(source_k)  # 将找到的相似图片的地址存放在列表中

        print("☆" * 10)
        print("搜索完成，全部结果如下：")
        for target_k in source_imgs.keys():
            if source_imgs[target_k]:
                print(f"找到的{target_k}全部相同图片的本地图片：{source_imgs[target_k]}")
            else:
                print(f"未找到{target_k}的相同图片")

    def _search_img(self, target_img, source_img, search_type='strict', local_type='surf'):
        """
        对比两张图片是否相同
        :param target_img: 一张目标图片
        :param source_img: 一张本地图片
        :param search_type: 搜索类型，包括 'strict', 'similar', 'local'
        :param local_type: 局部搜索类型(在 search_type='local' 时有效)，包括 'surf', 'orb', 'template'
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
            # 将图片压缩到相同的大小，比如300*200, todo: 这里可以优化，比如对于横屏竖屏使用不一样的缩放大小
            target_img = cv2.resize(target_img, (300, 200))
            source_img = cv2.resize(source_img, (300, 200))
            # absolute(a - b) <= (atol + rtol * absolute(b))
            if np.allclose(target_img, source_img, atol=10, rtol=0.01):
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


# 减小搜索域(传入压缩图片)
class DomainReducer:
    def __init__(self):
        # todo 这里可以加载配置文件
        pass

    def reduce_domain(self, target_dict, source_dict):
        """
        减小搜索域
        :param target_dict: 目标图片数组字典
        :param source_dict: 本地图片数组字典
        :return: source_imgs: 减小后的本地图片数组字典
        """
        source_imgs = {}  # 用字典存储每个目标图片对应的相似图片列表
        for i in target_dict.keys():
            source_imgs[i] = []

        # 对于target_dict中的每一张图片，都在source_dict中搜索是否存在相同的图片
        for target_k, target_v in target_dict.items():
            for source_k, source_v in source_dict.items():
                if self._search_img(target_v, source_v, search_type='similar'):
                    print("找到相同图片：", target_k, '<----->', source_k)
                    source_imgs[target_k].append(source_k)  # 将找到的相似图片的地址存放在列表中

        print("☆" * 10)
        print("搜索完成，全部结果如下：")
        for target_k in source_imgs.keys():
            if source_imgs[target_k]:
                print(f"找到的{target_k}全部相同图片的本地图片：{source_imgs[target_k]}")
            else:
                print(f"未找到{target_k}的相同图片")

        return source_imgs

    # 对比图片
    def _search_img(self, target_img, source_img, search_type='strict'):
        """
        对比两张图片是否相同
        :param target_img: 一张目标图片
        :param source_img: 一张本地图片
        :param compare_type: 对比类型，包括 'strict', 'similar'
        :return: 是否找到相同图片, True or False
        """
        if search_type == 'strict':
            if np.all(target_img == source_img):
                return True
            else:
                return False
        elif search_type == 'similar':
            # absolute(a - b) <= (atol + rtol * absolute(b))
            if np.allclose(target_img, source_img, atol=50, rtol=0.01):
                return True
            else:
                return False



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


