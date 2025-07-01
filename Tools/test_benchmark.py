from Tools.search_pic import SP
from Tools.read_pic import read_images
import time
import cv2
import random
import os
import logging


# 防止Using cache found in C:\Users\HP/.cache\torch\hub\facebookresearch_dinov2_main的输出


def blur_images(imgs_dict, kernel_size=(5, 5)):
    """
    对图像字典中的所有图像进行高斯模糊处理

    参数:
    imgs_dict -- 图像字典，key为路径，value为图像数组
    kernel_size -- 模糊核大小，默认为(5, 5)

    返回:
    新的图像字典，key保持不变，value为模糊后的图像
    """
    blurred_dict = {}

    for img_path, img in imgs_dict.items():
        # 应用高斯模糊
        blurred_img = cv2.GaussianBlur(img, kernel_size, 0)
        blurred_dict[img_path] = blurred_img

    return blurred_dict


def crop_images(imgs_dict, crop_ratio=0.8):
    """
    对图像字典中的所有图像进行随机位置裁剪

    参数:
    imgs_dict -- 图像字典，key为路径，value为图像数组
    crop_ratio -- 裁剪比例(0-1)，表示保留原图的比例

    返回:
    新的图像字典，key保持不变，value为裁剪后的图像
    """
    cropped_dict = {}

    for img_path, img in imgs_dict.items():
        height, width = img.shape[:2]
        crop_width = int(width * crop_ratio)
        crop_height = int(height * crop_ratio)

        # 随机生成裁剪起始位置
        max_x = width - crop_width
        max_y = height - crop_height
        start_x = random.randint(0, max_x) if max_x > 0 else 0
        start_y = random.randint(0, max_y) if max_y > 0 else 0

        cropped_img = img[start_y:start_y + crop_height, start_x:start_x + crop_width]
        cropped_dict[img_path] = cropped_img

    return cropped_dict


def test_search_origin(path_local):
    correct = 0
    total = 0
    sp = SP()
    sp.init_pic_df(path_local=path_local)
    imgs_dict = read_images(path_local)

    start_time = time.time()
    for img_path, img in imgs_dict.items():
        results = sp.search_origin(img)
        if isinstance(results, list) and len(results) > 0:
            top_result = results[0]  # 取第一个搜索结果
            if top_result == img_path:
                correct += 1
        else:
            pass  # 没有找到匹配的图片
        total += 1  # 说明没有找到匹配的图片
        print(f"Processed: {total}/{len(imgs_dict)}", end='\r')  # 进度显示

    recall_rate = (correct / total) * 100 if total > 0 else 0
    print(f"\n[ML-原图搜索 top-1] Recall rate: {correct}/{total} = {recall_rate:.2f}%")
    end_time = time.time()
    print(f"[ML-原图搜索] Total time: {end_time - start_time:.2f} s")  # 显示总时间


def test_search_similar(path_local, top_k=5):
    correct = 0
    total = 0
    sp = SP()
    sp.init_pic_df(path_local=path_local)
    imgs_dict = read_images(path_local)
    imgs_dict = blur_images(imgs_dict, kernel_size=(5, 5))  # 对图像进行模糊处理

    start_time = time.time()
    for img_path, img in imgs_dict.items():
        results = sp.search_similar(img, hash_threshold=0.3)
        if isinstance(results, list) and len(results) >= top_k:
            if any(result == img_path for result in results[:top_k]):
                correct += 1
        elif isinstance(results, list) and len(results) > 0:
            if any(result == img_path for result in results):
                correct += 1
        else:
            pass  # 没有找到匹配的图片
        total += 1
        print(f"Processed: {total}/{len(imgs_dict)} (Top-{top_k})", end='\r')  # 显示top_k信息

    recall_rate = (correct / total) * 100 if total > 0 else 0
    print(f"\n[ML-近似搜索 Top-{top_k}] Recall rate: {correct}/{total} = {recall_rate:.2f}%")
    end_time = time.time()
    print(f"[ML-近似搜索] Total time: {end_time - start_time:.2f} s")  # 显示总时间


def test_search_partial(path_local, top_k=5, crop_ratio=0.8):
    correct = 0
    total = 0
    sp = SP()
    sp.init_pic_df(path_local=path_local)
    imgs_dict = read_images(path_local)
    imgs_dict = crop_images(imgs_dict, crop_ratio=crop_ratio)  # 对图像进行随机裁剪

    start_time = time.time()
    for img_path, img in imgs_dict.items():
        results = sp.search_partial(img, top_k=top_k)
        if isinstance(results, list) and len(results) >= top_k:
            if any(result == img_path for result in results[:top_k]):
                correct += 1
        elif isinstance(results, list) and len(results) > 0:
            if any(result == img_path for result in results):
                correct += 1
        else:
            pass  # 没有找到匹配的图片
        total += 1
        print(f"Processed: {total}/{len(imgs_dict)} (Top-{top_k})", end='\r')  # 显示top_k信息

    recall_rate = (correct / total) * 100 if total > 0 else 0
    print(f"\n[DL-局部搜索 Top-{top_k}] Partial Recall rate: {correct}/{total} = {recall_rate:.2f}%")
    end_time = time.time()
    print(f"[DL-局部搜索] Total time: {end_time - start_time:.2f} s")  # 显示总时间


test_search_origin(path_local='F:/Picture/甘城なつき')
test_search_similar(path_local='F:/Picture/甘城なつき', top_k=1)
test_search_similar(path_local='F:/Picture/甘城なつき', top_k=3)
# test_search_similar(path_local='F:/Picture/甘城なつき', top_k=10)
test_search_partial(path_local='F:/Picture/甘城なつき', top_k=5)
test_search_partial(path_local='F:/Picture/甘城なつき', top_k=10)
