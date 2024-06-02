import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_image(img_path, gray_pic=False, show_details=False):
    """
    读取图片
    [使用示例]：
        path = '../output/arona.jpg'
        img = read_image(path, gray_pic=True, show_details=True)  # 读取为灰度图
    :param img_path: 图像路径
    :param gray_pic: 是否读取灰度图像
    :param show_details: 是否输出图片的shape以及显示图片
    :return: 图像数组，类型为np.ndarray。大小是(H, W, 3)或(H, W)
    """
    if gray_pic:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    else:
        img_gbr = cv2.imread(img_path)
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