# 检查是否含有对应版本的包
# 1. 检查是否含有对应版本的包
# 检查import cv2 import numpy as np  import matplotlib.pyplot as plt
try:
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError as e:
    print(e)
    print('请检查是否含有对应版本的包')
    exit(1)
# 2. 检查是否含有对应版本的包
