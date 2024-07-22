import imagehash
from scipy.spatial.distance import hamming, euclidean, cosine
from PIL import Image
import numpy as np

class HashPic:
    def __init__(self):
        pass

    def get_hash(self, img, hash_type='phash'):
        """
        获取图片的hash值
        :param img: np.ndarray，图片
        :param hash_type: str，hash方法，默认为phash，可选['ahash', 'phash', 'dhash', 'whash']
        :return: np.array，hash值(bool类型)
        """
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        img_pil = Image.fromarray(img)

        if hash_type == 'ahash':
            hash_value = imagehash.average_hash(img_pil)
        elif hash_type == 'phash':
            hash_value = imagehash.phash(img_pil)
        elif hash_type == 'dhash':
            hash_value = imagehash.dhash(img_pil)
        elif hash_type == 'whash':
            hash_value = imagehash.whash(img_pil)
        else:
            raise ValueError("Unsupported hash type. Choose from ['ahash', 'phash', 'dhash', 'whash']")

        return np.array(hash_value.hash).flatten()

    def cal_hash_distance(self, hash1, hash2, cal_type="hamming"):
        """
        根据两个图片hash计算相似度
        :param hash1: np.ndarray，图1的hash
        :param hash2: np.ndarray，图2的hash
        :param cal_type: str，计算类型，有["hamming", "cosine"]
        :return: float，相似度
        """
        if cal_type == "hamming":
            distance = hamming(hash1, hash2)
        # elif cal_type == "euclidean":
        #     distance = euclidean(hash1, hash2)  # TypeError: numpy boolean subtract, the `-` operator is not supported
        elif cal_type == "cosine":
            distance = cosine(hash1, hash2)
        else:
            raise ValueError("Unsupported calculation type. Choose from ['hamming', 'euclidean', 'cosine']")

        return distance


