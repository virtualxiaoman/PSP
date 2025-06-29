import time
import pandas as pd

from Tools.read_pic import read_image, read_images, imgs2df
from Tools.pic_util import HashPic
from Tools.search_pic import SP
from Tools.local_manager import demo_gal

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

path_origin = '../search/110182236_p0.jpg'
path_similar = '../search/110182236_p0_clip.png'
# path_similar = '../input/白子/Shiroko.jpeg'
img_origin = read_image(path_origin, gray_pic=False, show_details=False)
img_similar = read_image(path_similar, gray_pic=False, show_details=False)

start_time = time.time()
sp = SP()
sp.init_pic_df(path_local='F:/Picture/pixiv')
ans = sp.search_origin(img_origin)
print(ans)  # ['E:/Py-Project/PSP/input/阿洛娜/arona.jpg']
ans = sp.search_similar(img_similar)
print(ans)  # ['E:/Py-Project/PSP/input/白子/Shiroko.jpeg']
end_time = time.time()
elapsed_time = end_time - start_time
print("总时间： {:.2f} 秒".format(elapsed_time))

print(sp.df.head(5))
# 查看其中的dino这一列
# print(sp.df['dino'].head())
# 查看dino第一个元素的shape
# print(sp.df['dino'].iloc[0].shape)

# demo_gal(df=sp.df)
