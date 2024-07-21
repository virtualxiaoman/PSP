import os
import sys

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from PSP.pic_util import HashPic

# 读取一张图片
def read_image(img_path, gray_pic=False, show_details=False):
    """
    读取图片(支持中文路径)
    [使用示例]：
        path = 'input'
        img = read_image(path, gray_pic=True, show_details=True)  # 读取为灰度图
    [Tips]:
        1.使用cv2.imread()读取图片时，如果路径中含有中文，会导致读取失败，可以使用cv2.imdecode()解决，如下：
          先使用np.fromfile()取得二进制数据，再使用cv2.imdecode()解码，-1代表自动检测图像的颜色通道数和位深度。
    :param img_path: 图像路径
    :param gray_pic: 是否读取灰度图像
    :param show_details: 是否输出图片的shape以及显示图片
    :return: img, 图像数组，类型为np.ndarray。大小是(H, W, 3)或(H, W)
    """
    if gray_pic:
        # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        # img_gbr = cv2.imread(img_path)
        img_gbr = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        img = cv2.cvtColor(img_gbr, cv2.COLOR_BGR2RGB)

    if show_details:
        print(img.shape)
        if gray_pic:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
        plt.show()
        plt.close()
    return img

# 读取文件夹下的所有图片
def read_imgs(path, gray_pic=False, show_details=False):
    """
    读取文件夹下的所有图片
    [使用示例]：
        path = 'input'
        imgs = read_imgs(path)
        print(len(imgs))
    :param path: 文件夹路径
    :param gray_pic: 是否读取灰度图像
    :param show_details: 是否输出图片的shape以及显示图片
    :return: imgs_dict: 图像数组字典，key是图片的绝对路径，value是图片数组
    """
    # 检查path是否存在
    if not os.path.exists(path):
        print(f"路径不存在：{path}")
        return -1

    # imgs = []
    imgs_dict = {}  # 存放读取到的所有图片，key是图片的绝对路径，value是图片数组

    # os.walk()返回一个三元组，分别是：当前路径，当前路径下的文件夹，当前路径下的文件
    for root, dirs, files in os.walk(path):
        # print("当前目录:", root)
        # print("子目录:", dirs)
        # print("文件:", files)
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg'):

                # print("相对路径", os.path.join(root, file))
                # print("绝对路径", os.path.abspath(os.path.join(root, file)))

                img_path = os.path.join(root, file)
                img = read_image(img_path, gray_pic=gray_pic, show_details=show_details)
                img_abs_path = os.path.abspath(img_path).replace('\\', '/')
                imgs_dict[img_abs_path] = img

    return imgs_dict

# 将某个文件夹下的所有图片读取为所需要的dataframe
def imgs2df(path, hash_type='phash', save_path=None):
    """
    将某个文件夹下的所有图片读取为所需要的dataframe
    :param path: 文件夹路径
    :param hash_type: 暂时只使用phash
    :param save_path: 保存路径
    :return: df: dataframe，有八列：
        "id"是递增序列(用于查询图片)，"path"是图片的绝对路径，"hash"是图片的hash值(phash)，"size"是图片的占据的空间大小，
        "shape"是图片的shape，"mean"是图片的像素均值，"std"是图片的像素标准差，"25p"将图片变为的5*5大小之后的像素
    """
    # 检查path是否存在
    if not os.path.exists(path):
        raise FileNotFoundError(f"[imgs2df]路径不存在：{path}")
    if hash_type != 'phash':
        raise ValueError("[imgs2df]虽然HashPic这个类已经支持多种hash方法，但为了减少可能的错误，"
                         "目前imgs2df只写了phash这一种方法，如有需求请删除这个ValueError")

    data = []
    hp = HashPic()
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            # 'bmp', 'tiff' 没测试过
            if file.endswith(('jpg', 'jpeg', 'png', 'bmp', 'tiff')):
                file_path = os.path.join(root, file)
                file_path = os.path.abspath(file_path).replace('\\', '/')
                file_list.append(file_path)
    file_list_length = len(file_list)

    for idx, file_path in enumerate(file_list):
        img = read_image(file_path, gray_pic=False, show_details=False)
        if img is not None:
            hash_value = hp.get_hash(img, hash_type)
            size, shape, mean, pixel_25p = get_image_info(img)
            # 注释是运行时间，以365张BA的图为例：
            # read_image是9.076秒，get_image_info是13.849秒，imgs2df合计26.117秒
            data.append({
                'id': len(data),
                'path': file_path,
                'hash': hash_value,  # 365    0.056    0.000    1.770    0.005 __init__.py:260(phash)
                'size': size,
                'shape': shape,
                'mean': mean,  # 730    0.003    0.000    1.611    0.002 fromnumeric.py:3385(mean)
                # 'std': std,  # 365    0.002    0.000   10.589    0.029 fromnumeric.py:3513(std)，时间太长不要了
                '25p': pixel_25p  # 365    1.653    0.005    1.653    0.005 {resize}
            })
            del img
        print(f"\r已经读取第{idx + 1}/{file_list_length}张图片", end='')
        if idx % 100 == 0:
            print("\ndata占用的存储空间为：", sys.getsizeof(data) / (1024 * 1024), "MB")
            df = pd.DataFrame(data, columns=['id', 'path', 'hash', 'size', 'shape', 'mean', 'std', '25p'])
            df.to_pickle(save_path)
            print(f"已保存{idx}行的dataframe")
            del df

    df = pd.DataFrame(data, columns=['id', 'path', 'hash', 'size', 'shape', 'mean', 'std', '25p'])
    df.to_pickle(save_path)
    print(f"\ndataframe已全部保存到{save_path}")
    return df

# 获取图片的信息
def get_image_info(img):
    """
    获取图片的信息
    :param img: np.ndarray，图片数组
    :return: size, shape, mean, std, pixel_25p
    """
    size = img.size
    shape = img.shape
    mean = np.mean(img)
    resized_img = cv2.resize(img, (5, 5), interpolation=cv2.INTER_AREA)
    pixel_25p = resized_img.flatten()
    return size, shape, mean, pixel_25p
