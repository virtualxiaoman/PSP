# 1. 检查是否含有对应的包
try:
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError as e:
    print(e)
    print('请检查是否含有对应版本的包')
    exit(1)

# 2. 版本号
__version__ = '1.0.0'


# 3. 没啥用的代码
# 3.1 使用ORB特征检测和描述进行图像匹配，因为区分度太低了，所以不用
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
query_image_path = '../search/阿洛娜_网图.jpg'
gallery_image_path = '../search/阿洛娜_原图.jpg'

query_img = cv2.imdecode(np.fromfile(query_image_path, dtype=np.uint8), -1)
query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
gallery_img = cv2.imdecode(np.fromfile(gallery_image_path, dtype=np.uint8), -1)
gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)

# 转换为灰度图像
query_gray = cv2.cvtColor(query_img, cv2.COLOR_RGB2GRAY)
gallery_gray = cv2.cvtColor(gallery_img, cv2.COLOR_RGB2GRAY)

# ORB特征检测和描述
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(query_gray, None)
keypoints2, descriptors2 = orb.detectAndCompute(gallery_gray, None)

# BFMatcher匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# 按距离排序
matches = sorted(matches, key=lambda x: x.distance)


# 计算匹配的相似度评分
num_matches = len(matches)
total_keypoints = min(len(keypoints1), len(keypoints2))
similarity_score = num_matches / total_keypoints if total_keypoints > 0 else 0

print(f'Number of Matches: {num_matches}'
      f'\nTotal Keypoints: {total_keypoints}'
      f'\nSimilarity Score: {similarity_score:.4f}')
      
# 绘制匹配结果
img_matches = cv2.drawMatches(query_img, keypoints1, gallery_img, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize=(15, 10))
plt.imshow(img_matches)
plt.show()
"""