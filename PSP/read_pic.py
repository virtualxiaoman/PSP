import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

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
                imgs_dict[os.path.abspath(img_path)] = img

    return imgs_dict
