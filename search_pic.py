import numpy as np


def search_imgs(target_dict, source_dict):
    """
    搜索文件夹下的所有图片
    :param target_dict: 目标图片数组字典
    :param source_dict: 本地图片数组字典
    :return:
    """
    # 对于target_dict中的每一张图片，都在source_dict中搜索是否存在相同的图片
    for target_k, target_v in target_dict.items():
        for source_k, source_v in source_dict.items():
            # 如果两张图片的shape不同，则直接跳过
            if target_v.shape != source_v.shape:
                continue
            # 如果两张图片的shape相同，则进一步比较两张图片是否相同
            if np.all(target_v == source_v):
                print("找到相同图片：", target_k, source_k)
                break
        else:
            print("未找到相同图片：", target_k)