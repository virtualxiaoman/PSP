# # https://segmentfault.com/a/1190000042234114    towhee
# # https://segmentfault.com/a/1190000038308093    dHash
# # https://blog.csdn.net/sinat_27382047/article/details/83040411    较全
#
# # 其他
# # https://blog.csdn.net/wsp_1138886114/article/details/84766965
# # https://segmentfault.com/a/1190000004467183
#
# # https://blog.csdn.net/imwaters/article/details/117426488
# # https://blog.csdn.net/weixin_41809530/article/details/109258984
# import os
# import time
#
# from Tools.read_pic import read_imgs, read_image
# from Tools.search_pic import SearchPic, CompressPic, DomainReducer
#
# path = 'input'
# # path = r'F:\图片存储 Picture\blue archive'
# path = r"F:\TEMP awa\Cosplay Tales"
# source_imgs_dict = read_imgs(path)
# # for k, v in source_imgs_dict.items():
# #     print(k, v.shape)
# print('-------------------')
#
# target_path = r'search'
# # target_imgs_dict = {r'search\阿洛娜_网图.jpg': read_image(target_path + r'\阿洛娜_网图.jpg')}
# target_imgs_dict = {r'search\阿洛娜_水印.png': read_image(target_path + r'\阿洛娜_水印.png')}
# target_path = r"D:\HP\Desktop\111"
# target_imgs_dict = read_imgs(target_path)
#
#
# cp = CompressPic()
# target_imgs_1x_dict = cp.compress_imgs(target_imgs_dict, resize=(1, 1), name="测试数据_target_1x")
# target_imgs_9x_dict = cp.compress_imgs(target_imgs_dict, resize=(3, 3), name="测试数据_target_9x")
# source_imgs_1x_dict = cp.compress_imgs(source_imgs_dict, resize=(1, 1), name="测试数据_source_1x")
# source_imgs_9x_dict = cp.compress_imgs(source_imgs_dict, resize=(3, 3), name="测试数据_source_9x")
#
# imgs_s_origin = source_imgs_dict
# for k, v in target_imgs_dict.items():
#     dr = DomainReducer()
#     # imgs_s_origin = dr.get_dr_dict({k: v}, source_imgs_dict, target_imgs_1x_dict, source_imgs_1x_dict, search_type='mean')
#     # imgs_s_origin = dr.get_dr_dict({k: v}, imgs_s_origin, target_imgs_9x_dict, source_imgs_9x_dict, search_type='mean')
#     start1 = time.time()
#     psp = SearchPic()
#     psp.search_imgs({k: v}, imgs_s_origin, search_type='local')
#     psp.log_result()
#     end1 = time.time()
#     print("DR算法耗时：", end1 - start1)
#
#
#
# exit(1)
# # imgs_s_origin = DomainReducer().get_dr_dict(target_imgs_dict, source_imgs_dict, dr_type='1px', search_type='mean')
# # print(imgs_s_origin)
# # imgs_s_origin = DomainReducer().get_dr_dict(target_imgs_dict, imgs_s_origin, dr_type='9px', search_type='mean')
#
# print("----------搜索域减小完成---------")
#
# print("----------搜索图片---------")
# # 计时
# import time
# start1 = time.time()
# # 搜索图片
#
# # 遍历 target_imgs_dict 的每个 key
# for target_key, target_value in target_imgs_dict.items():
#     # 查找 imgs_s_origin 中对应的 value
#     if target_key in imgs_s_origin:
#         print(f"-------Searching {target_key}...")
#         source_imgs = imgs_s_origin[target_key]
#         psp = SearchPic()
#         psp.param_similar_atol = 100
#         psp.search_imgs({target_key: target_value}, source_imgs, search_type='local')
#         psp.log_result()
#     else:
#         print(f"Warning: {target_key} not found in imgs_s_origin")
# end1 = time.time()
#
#
# print("DR算法耗时：", end1 - start1)
#
#
# # print('---------输出----------')
# # # 输出imgs_dict与imgs1_s
# # for k, v in source_imgs_dict.items():
# #     print(k, v.shape)
# # print('-------------------')
# # for k, v in imgs1_s.items():
# #     print(k, v)
#
#
#
#
#

import sys
from PyQt6.QtWidgets import QApplication
from UI.ui import PSP_UI

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = PSP_UI()
    ui.show()
    sys.exit(app.exec())
