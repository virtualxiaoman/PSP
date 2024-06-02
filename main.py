import cv2
import numpy as np
import matplotlib.pyplot as plt

from read_pic import read_imgs, read_image
from search_pic import search_imgs

path = 'input'
imgs_dict = read_imgs(path)
for k, v in imgs_dict.items():
    print(k, v.shape)


print('-------------------')
target_path = 'search_pic'
target_imgs_dict = {'search_pic/test.jpg': read_image(target_path + '/test.jpg')}

search_imgs(target_imgs_dict, imgs_dict)





