import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from PIL import Image
from torchvision import transforms

from Tools.local_matcher import LocalMatcher
from Tools.pic_util import HashPic


# # 在import cv2前设置，避免libpng warning: iCCP: known incorrect sRGB profile
# os.environ['OPENCV_IO_IGNORE_ICC_PROFILE'] = '1'  # 注: 此方法无效（即使放在import cv2前面也无效），我红温了


# 读取一张图片
def read_image(img_path, gray_pic=False, show_details=False):
    """
    读取图片(支持中文路径)
    [使用示例]：
        path = 'input'
        img = read_image(path, gray_pic=True, show_details=True)  # 读取为灰度图
    [Tips]:
        1.使用cv2.imread()读取图片时，如果路径中含有中文，会导致读取失败，可以使用cv2.imdecode()解决，具体如下：
          先使用np.fromfile()取得二进制数据，再使用cv2.imdecode()解码，-1代表自动检测图像的颜色通道数和位深度。
    :param img_path: 图像路径
    :param gray_pic: 是否读取灰度图像
    :param show_details: 是否输出图片的shape以及显示图片
    :return: img, 图像数组，类型为np.ndarray。大小是(H, W, 3)或(H, W)
    """
    if gray_pic:
        # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 不使用该方法的原因见函数注释的Tips
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
def read_images(path, gray_pic=False, show_details=False):
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
def imgs2df(path, hash_type='phash', save_path=None, log_callback=None):
    """
    将某个文件夹下的所有图片读取为所需要的dataframe
    :param path: 文件夹路径
    :param hash_type: 暂时只能使用phash
    :param save_path: 保存路径
    :param log_callback: 日志传回到QT里去
    :return: df: dataframe，有七列：
        "id"是递增序列(用于查询图片)，"path"是图片的绝对路径，"hash"是图片的hash值(phash)，"size"是h*w*c，
        "shape"是(h,w,c)，"mean"是图片的像素均值
    """
    # 检查path是否存在
    if not os.path.exists(path):
        raise FileNotFoundError(f"[imgs2df] 路径不存在：{path}")
    if hash_type != 'phash':
        raise ValueError("[imgs2df]虽然HashPic这个类已经支持多种hash方法，但因为其余地方不全支持其余hash方法，"
                         "并且计算多种hash也需要时间，为了加快构建速度并减少可能的错误，"
                         "目前imgs2df只写了phash这一种方法，如有需求请删除这个ValueError")

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LocalMatcher().to(device)
    model.load_state_dict(torch.load("../assets/checkpoint_epoch_14.pth", map_location=device))
    model.eval()

    file_list = []  # 图片绝对路径
    for root, dirs, files in os.walk(path):
        for file in files:
            # 注意：'bmp', 'tiff' 没测试过
            if file.endswith(('jpg', 'jpeg', 'png', 'bmp', 'tiff')):
                file_path = os.path.join(root, file)
                file_path = os.path.abspath(file_path).replace('\\', '/')
                file_list.append(file_path)
    file_list_length = len(file_list)

    start_time = time.time()
    data = []
    total_processed = 0

    def process_img(img_path):
        """ 多线程处理单个文件的函数 """
        try:
            img = read_image(img_path, gray_pic=False, show_details=False)
            if img is None:
                return None
            hp = HashPic()  # 每个线程独立实例保证线程安全
            hash_value = hp.get_hash(img, hash_type)
            size, shape, mean = get_image_info_ml(img)
            dinov2_feature = get_image_info_dl(img, device, model)  # 获取Dinov2特征向量
            del img
            return {
                'path': img_path,
                'hash': hash_value,
                'size': size,
                'shape': shape,
                'mean': mean,
                'dino': dinov2_feature
            }
        except Exception as e:
            if log_callback:
                log_callback(f"处理失败: {img_path} - {str(e)}")
            return None

    # 使用线程池并行处理
    with ThreadPoolExecutor() as executor:
        # 提交所有任务
        futures = {executor.submit(process_img, fp): fp for fp in file_list}

        # 实时处理完成的任务
        for future in as_completed(futures):
            total_processed += 1

            # 更新进度日志
            log_msg = f"已处理第 {total_processed}/{file_list_length} 张图片"
            print(f"\r[imgs2df] {log_msg}", end='')
            if log_callback:
                log_callback(log_msg)  # 传回到QT界面

            # 获取处理结果
            result = future.result()
            if result:
                # 分配递增ID
                result['id'] = len(data)
                data.append(result)

                # 定期保存（每100个任务保存一次）
                if total_processed % 100 == 0:
                    df = pd.DataFrame(data)
                    if save_path:
                        df.to_pickle(save_path)
                        data_size = sys.getsizeof(data) / (1024 * 1024)
                        print(f"\n[imgs2df] data占用{data_size:.2f} MB, 已保存{total_processed}行到{save_path}")
                    del df
    # print(data)
    # print(save_path)
    # 查看绝对路径
    print(f"[imgs2df] 保存路径: {os.path.abspath(save_path)}")
    df = pd.DataFrame(data)
    df.to_pickle(save_path)
    print(f"\n[imgs2df] dataframe已全部保存到{save_path}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    # 计算每张图片的耗时
    elapsed_time_per_img = elapsed_time / file_list_length
    print(f"[imgs2df] 总耗时: {elapsed_time:.2f} 秒, 每张图片耗时: {elapsed_time_per_img:.4f} 秒")
    return df


# 将某个文件夹下的所有图片读取为所需要的dataframe，这是先前的版本，作为备份
# def imgs2df(path, hash_type='phash', save_path=None, log_callback=None):
#     """
#     将某个文件夹下的所有图片读取为所需要的dataframe
#     :param path: 文件夹路径
#     :param hash_type: 暂时只能使用phash
#     :param save_path: 保存路径
#     :param log_callback: 日志传回到QT里去
#     :return: df: dataframe，有七列：
#         "id"是递增序列(用于查询图片)，"path"是图片的绝对路径，"hash"是图片的hash值(phash)，"size"是h*w*c，
#         "shape"是(h,w,c)，"mean"是图片的像素均值，"25p"将图片变为的5*5大小之后的像素
#     """
#     # 检查path是否存在
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"[imgs2df] 路径不存在：{path}")
#     if hash_type != 'phash':
#         raise ValueError("[imgs2df]虽然HashPic这个类已经支持多种hash方法，但因为其余地方不全支持其余hash方法，"
#                          "并且计算多种hash也需要时间，为了加快构建速度并减少可能的错误，"
#                          "目前imgs2df只写了phash这一种方法，如有需求请删除这个ValueError")
#
#     file_list = []  # 图片绝对路径
#     for root, dirs, files in os.walk(path):
#         for file in files:
#             # 注意：'bmp', 'tiff' 没测试过
#             if file.endswith(('jpg', 'jpeg', 'png', 'bmp', 'tiff')):
#                 file_path = os.path.join(root, file)
#                 file_path = os.path.abspath(file_path).replace('\\', '/')
#                 file_list.append(file_path)
#     file_list_length = len(file_list)
#
#     data = []  # 图片信息
#     hp = HashPic()
#     start_time = time.time()
#     for idx, file_path in enumerate(file_list):
#         img = read_image(file_path, gray_pic=False, show_details=False)
#         if img is not None:
#             hash_value = hp.get_hash(img, hash_type)
#             size, shape, mean, pixel_25p = get_image_info(img)
#             data.append({
#                 'id': len(data),
#                 'path': file_path,
#                 'hash': hash_value,
#                 'size': size,
#                 'shape': shape,
#                 'mean': mean,
#                 '25p': pixel_25p
#             })
#             del img
#         log_text = f"已经读取第{idx + 1}/{file_list_length}张图片"
#         print(f"\r[imgs2df] {log_text}", end='')
#         if log_callback:
#             log_callback(log_text)  # 传回到QT界面
#         if idx % 100 == 0:
#             df = pd.DataFrame(data, columns=['id', 'path', 'hash', 'size', 'shape', 'mean', '25p'])
#             df.to_pickle(save_path)
#             data_size = sys.getsizeof(data) / (1024 * 1024)
#             print(f"\n[imgs2df] data占用的存储空间为：{data_size:.2f} MB, 已保存{idx}行的dataframe到{save_path}")
#             del df
#
#     df = pd.DataFrame(data, columns=['id', 'path', 'hash', 'size', 'shape', 'mean', '25p'])
#     df.to_pickle(save_path)
#     print(f"\n[imgs2df] dataframe已全部保存到{save_path}")
#
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     # 计算每张图片的耗时
#     elapsed_time_per_img = elapsed_time / file_list_length
#     print(f"[imgs2df] 总耗时: {elapsed_time:.2f} 秒, 每张图片耗时: {elapsed_time_per_img:.4f} 秒")
#     return df


# 获取图片的信息
def get_image_info_ml(img):
    """
    获取图片的信息
    :param img: np.ndarray，图片数组
    :return: size, shape, mean
    """
    size = img.size
    shape = img.shape
    mean = np.mean(img)
    # resized_img = cv2.resize(img, (5, 5), interpolation=cv2.INTER_NEAREST)  # 原先是INTER_AREA，速度慢一些
    # pixel_25p = resized_img.flatten()
    return size, shape, mean


def get_image_info_dl(img, device, model):
    """
    (H, W, C) -> (1, C, 224, 224) -> 1, 768(Dinov2特征向量)
    :param img:
    :param device:
    :param model:
    :return:
    """
    # print(111)
    # 预处理原始图像
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # original_input = original_transform(img).unsqueeze(0).to(device)  # [1, 3, 224, 224]
    # with torch.no_grad():
    #     # print(f"original_input shape: {original_input.shape}, crop_img shape: {crop_img.shape}")
    #     output = model(original_input, crop_img)
    #     probability = torch.sigmoid(output).item()
    #
    # return feature.squeeze(0).cpu()  # 移除批次维度 [C, H, W] 或 [D]
    img_tensor = transform(img).unsqueeze(0).to(device)
    # print(img_tensor.shape)
    with torch.no_grad():
        dinov2_feature = model.backbone(img_tensor)
    # print(dinov2_feature.shape)  # 输出特征向量的形状
    return dinov2_feature.squeeze(0).cpu().numpy()


if __name__ == "__main__":
    # 测试读取单张图片
    img_path = "F:/Picture/pixiv/BA/110182236_p0.jpg"
    img = read_image(img_path, gray_pic=False, show_details=False)
    print(f"读取图片形状: {img.shape}")
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LocalMatcher().to(device)
    model.load_state_dict(torch.load("../assets/checkpoint_epoch_14.pth", map_location=device))
    model.eval()
    get_image_info_dl(img, device, model)  # 获取Dinov2特征向量

# class FeatureExtractor:
#     def __init__(self, model_path="../assets/checkpoint_epoch_14.pth", backbone='dinov2_vitb14'):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#         # 加载模型但不加载匹配头
#         self.model = torch.hub.load('facebookresearch/dinov2', backbone).to(self.device)
#
#         # 加载完整模型状态，然后只保留backbone部分
#         full_model = LocalMatcher(backbone)
#         full_model.load_state_dict(torch.load(model_path, map_location=self.device))
#         self.model.load_state_dict(full_model.backbone.state_dict())
#
#         self.model.eval()
#         self.transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225]),
#         ])
#
#     def extract_feature(self, image_path):
#         """提取单张图片的特征向量"""
#         img = Image.open(image_path).convert("RGB")
#         input_tensor = self.transform(img).unsqueeze(0).to(self.device)  # [1, 3, 224, 224]
#
#         with torch.no_grad():
#             feature = self.model(input_tensor)
#
#         return feature.squeeze(0).cpu()  # 移除批次维度 [feat_dim]
#
#
# # 使用示例
# if __name__ == "__main__":
#     extractor = FeatureExtractor()
#
#     # 提取单张图片特征
#     feature_vector = extractor.extract_feature("F:/Picture/pixiv/BA/110182236_p0.jpg")
#     print(f"特征向量形状: {feature_vector.shape}")
#     print(f"特征示例: {feature_vector[:5]}")
