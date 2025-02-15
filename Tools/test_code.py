import time
import cProfile

from Tools.read_pic import read_image, read_images, imgs2df
from Tools.pic_util import HashPic
from Tools.search_pic import SP
import pandas as pd
import os

pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.expand_frame_repr', False)  # 不允许水平拓展
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.width', None)  # 不换行

# path1 = '../search/阿洛娜_水印_多.png'
# path2 = '../search/阿洛娜_模糊_略微裁剪.jpg'
# img1 = read_image(path1, gray_pic=False, show_details=False)
# img2 = read_image(path2, gray_pic=False, show_details=False)


# path_origin = '../search/阿洛娜_原图.jpg'
# path_similar = '../search/阿洛娜_水印_重复.jpg'
# img_origin = read_image(path_origin, gray_pic=False, show_details=False)
# img_similar = read_image(path_similar, gray_pic=False, show_details=False)

path_origin = "../search/白州梓_模糊.jpg"
img_origin = read_image(path_origin, gray_pic=False, show_details=False)

start_time = time.time()
sp = SP()
sp.init_pic_df(path_local='F:/Picture/pixiv')
end_time = time.time()
elapsed_time = end_time - start_time
print("初始化： {:.2f} 秒".format(elapsed_time))

start_time = time.time()
ans = sp.search_similar(img_origin)
print(ans)
end_time = time.time()
elapsed_time = end_time - start_time
print("总时间： {:.2f} 秒".format(elapsed_time))
exit(11)

img_dict = read_imgs(path_local)  # key是图片的绝对路径，value是图片数组
# 创建一个dataframe，有三列，"id"是递增序列(用于查询图片)，"path"是图片的绝对路径，"hash"是图片的hash值(phash)
df = pd.DataFrame(columns=['id', 'path', 'hash'])
hp = HashPic()
for i, (k, v) in enumerate(img_dict.items()):
    hash_value = hp.get_hash(v, hash_type='phash')
    new_row = pd.DataFrame({'id': [i], 'path': [k], 'hash': [hash_value]})
    df = pd.concat([df, new_row], ignore_index=True)
# print(df)


hp = HashPic()
hash1 = hp.get_hash(img1, hash_type='phash')
hash2 = hp.get_hash(img2, hash_type='phash')
similarity = hp.cal_hash_distance(hash1, hash2, cal_type='hamming')
print(similarity)

# 对于df中的每一行，计算其hash值与hash1的hamming距离，将结果存入新的一列"distance"
df['distance'] = df['hash'].apply(lambda x: hp.cal_hash_distance(x, hash1, cal_type='hamming'))
# print(df)
# 以"distance"列的值从小到大排序
df = df.sort_values(by='distance')
print(df.head(10))

df['distance'] = df['hash'].apply(lambda x: hp.cal_hash_distance(x, hash2, cal_type='hamming'))
# print(df)
# 以"distance"列的值从小到大排序
df = df.sort_values(by='distance')
print(df.head(10))


