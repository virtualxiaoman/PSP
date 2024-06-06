import os

from PSP.read_pic import read_imgs, read_image
from PSP.search_pic import SearchPic, CompressPic, DomainReducer

path = 'input'
path = r'F:\图片存储 Picture\blue archive'
imgs_dict = read_imgs(path)
# for k, v in imgs_dict.items():
#     print(k, v.shape)
print('-------------------')



# print('-------------------')
target_path = 'search'
target_imgs_dict = {r'search\阿洛娜_原图.jpg': read_image(target_path + r'\阿洛娜_原图.jpg'),
                    r'search\阿洛娜_局部.jpg': read_image(target_path + r'\阿洛娜_局部.jpg'),
                    r'search\阿洛娜_水印.png': read_image(target_path + r'\阿洛娜_水印.png'),
                    r'search\阿洛娜_比较糊.png': read_image(target_path + r'\阿洛娜_比较糊.png'),
                    r'search\阿洛娜_图片被修改.png': read_image(target_path + r'\阿洛娜_图片被修改.png'),
                    r'search\阿洛娜_网图.jpg': read_image(target_path + r'\阿洛娜_网图.jpg')}
target_imgs_dict = {r'search\阿洛娜_网图.jpg': read_image(target_path + r'\阿洛娜_网图.jpg')}


cp_s = CompressPic(imgs_dict)
imgs1_s = cp_s.img_1pixel
cp_t = CompressPic(target_imgs_dict)
imgs1_t = cp_t.img_1pixel

print("\n"*5)
print("----------减小搜索域---------")
dr = DomainReducer()
imgs1_s = dr.search_imgs(imgs1_t, imgs1_s, search_type='similar')


# 对比imgs1_s与imgs_dict，依据imgs_dict里的key，选择imgs_dict里的value相同的，将这些图片的路径存入imgs1_s_origin里

# target_imgs_dict是目标图片，key是图片的绝对路径，value是图片数组
# imgs_dict是本地的原始数据图片，key是图片的绝对路径，value是图片数组
# imgs1_s是压缩后的搜索域图片，key是目标图片的绝对路径，value是一个list，list里面的元素是待搜索的图片路径
# 首先根据target_imgs_dict的key查找imgs1_s的key，然后根据imgs1_s的value查找imgs_dict的key，将这些key存入imgs1_s_origin
# 最后根据imgs_dict的key查找imgs_dict的value，将这些value存入imgs1_s_origin
imgs1_s_origin = {}

# 遍历 target_imgs_dict 的 key
for target_key in target_imgs_dict.keys():
    # 标准化 target_key
    normalized_target_key = os.path.normpath(target_key)
    # 查找 imgs1_s 中是否存在标准化的 target_key
    if normalized_target_key in map(os.path.normpath, imgs1_s.keys()):
        # 获取 imgs1_s 中对应的 value (待搜索的图片路径列表)
        search_image_paths = imgs1_s[normalized_target_key]
        # 遍历待搜索的图片路径列表
        for search_image_path in search_image_paths:
            # 标准化 search_image_path
            normalized_search_image_path = os.path.normpath(search_image_path)
            # 查找 imgs_dict 中是否存在标准化的 search_image_path
            for key, value in imgs_dict.items():
                if os.path.normpath(key) == normalized_search_image_path:
                    imgs1_s_origin[key] = value




# imgs1_s_origin = {}
# for k, v in imgs_dict.items():
#     normalized_key = os.path.normpath(k)
#     if normalized_key in map(os.path.normpath, imgs1_s.values()):
#         imgs1_s_origin[normalized_key] = v

print("----------搜索域减小完成---------")
for k, v in imgs1_s_origin.items():
    print(k, v.shape)

print("----------搜索图片---------")
# 计时
import time
start1 = time.time()
# 搜索图片
psp = SearchPic()
psp.search_imgs(target_imgs_dict, imgs1_s_origin, search_type='local')
end1 = time.time()
psp.log_result()


start = time.time()
psp.search_imgs(target_imgs_dict, imgs_dict, search_type='local')
end = time.time()
psp.log_result()
print("DR算法耗时：", end1 - start1)
print("传统算法耗时：", end - start)

# print('---------输出----------')
# # 输出imgs_dict与imgs1_s
# for k, v in imgs_dict.items():
#     print(k, v.shape)
# print('-------------------')
# for k, v in imgs1_s.items():
#     print(k, v)





