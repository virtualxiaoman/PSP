import time

from Tools.read_pic import read_image, read_images, imgs2df
from Tools.pic_util import HashPic
from Tools.search_pic import SP

path_origin = '../search/阿洛娜_原图.jpg'
path_similar = '../search/阿洛娜_水印_重复.jpg'
path_local = '../search/110182236_p0_clip.png'
img_origin = read_image(path_origin, gray_pic=False, show_details=False)
img_similar = read_image(path_similar, gray_pic=False, show_details=False)
img_local = read_image(path_local, gray_pic=False, show_details=False)

start_time = time.time()
sp = SP()
sp.init_pic_df(path_local='../input')
ans = sp.search_origin(img_origin)
print(ans)  # ['E:/Py-Project/PSP/input/阿洛娜/arona.jpg']
ans = sp.search_similar(img_similar)
print(ans)  # ['E:/Py-Project/PSP/input/白子/Shiroko.jpeg']
end_time = time.time()
elapsed_time = end_time - start_time
print("[ML] 总时间： {:.2f} 秒".format(elapsed_time))  # 0.09 秒

# print(sp.df.head)
# 查看其中的dino这一列
# print(sp.df['dino'].head())
# 查看dino第一个元素的shape
# print(sp.df['dino'].iloc[0].shape)

start_time = time.time()
sp = SP()
sp.init_pic_df(path_local='F:/Picture/pixiv')
ans = sp.search_partial(img_local, top_k=3)
print(ans)
end_time = time.time()
elapsed_time = end_time - start_time
print("[DL] 总时间： {:.2f} 秒".format(elapsed_time))  # 6.43 秒(需要把模型加载到GPU上)

start_time = time.time()
ans = sp.search_partial(img_local)
# print(ans)
end_time = time.time()
elapsed_time = end_time - start_time
print("[DL] 总时间： {:.2f} 秒".format(elapsed_time))  # 3.14 秒

# [search_origin] input_size: 12572253, input_shape: (1731, 2421, 3)
# 查看df中路径为：F:/Picture/pixiv/BA/110182236_p0.jpg的size和shape
print(sp.df[sp.df['path'] == 'F:/Picture/pixiv/BA/110182236_p0.jpg'][['size', 'shape', 'mean']])
# 12572253  (1731, 2421, 3)  198.689028
