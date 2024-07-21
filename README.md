## 1.项目功能
- 本项目PSP(PicSearchPic)是带GUI的以图搜图工具。
- 目前计划开发网络图片搜索与本地图片搜索两个功能。
- 本项目中，一般用`local`代表本地图片，`input`代表输入图片。

**示例：**
本地图库有365张，106 MB。初始化图库数据： 15.57 秒

-------原图`local`--------------------原图`input`--------------------近似图片`input`-------
<div>
    <img src="input/阿洛娜/arona.jpg" width="200">
    <img src="search/阿洛娜_原图.jpg" width="200">
    <img src="search/阿洛娜_水印_重复.jpg" width="200">
</div>
查原图：0.09秒，查近似：0.03秒（暂时不知道为什么查原图还是慢一些，可能需要更精确的匹配）

**Quick Start：**

请注意更换为你自己的图片路径，另外第一次运行时会初始化图库数据，需要一定时间，第二次运行就很快了。
```python
path_origin = '../search/阿洛娜_原图.jpg'
path_similar = '../search/阿洛娜_水印_重复.jpg'
img_origin = read_image(path_origin, gray_pic=False, show_details=False)
img_similar = read_image(path_similar, gray_pic=False, show_details=False)

start_time = time.time()
sp = SP()
sp.init_pic_df(path_local='F:/图片存储 Picture/blue archive')
ans = sp.search_origin(img_origin)
print(ans)
ans = sp.search_similar(img_similar)
print(ans)
end_time = time.time()
elapsed_time = end_time - start_time
print("总时间： {:.2f} 秒".format(elapsed_time))
```
输出：
```
从../data/blue archive.pkl初始化dataframe完成  # 这是自带的初始化输出
['F:/图片存储 Picture/blue archive/官图/arona.jpg']  # 原图的搜索结果
['F:/图片存储 Picture/blue archive/官图/arona.jpg']  # 近似图的搜索结果
总时间： 0.11 秒  # 总时间
```

目前就是初始化图库数据比较慢，后续会优化。

对于我本地的一个`15.3 GB(5443张图)`的图库，初始化用时`638.60秒`，但是查询只需要`0.25秒`。

另外中文路径的似乎一直比英文的慢，暂时不知道原因。

**注：下面的先别看，没翻修。**

## 2.项目结构
```
.
├─ input                 # 本地图片文件夹[测试用]
├─ search                # 测试图片文件夹[测试用]
├─ PSP                   # 本项目主要功能的代码
│  ├─ __init__.py        # 初始化文件
│  ├─ read_pic.py        # 读取图片功能
│  ├─ search_pic.py      # 图片搜索功能
│  ├─ util.py            # 工具
├─ __init__.py           # 初始化文件
├─ main.py               # 主程序入口(运行此文件以启动)
```
#### 测试输入图片input的目录结构:
```
.
├─ input
│  ├─ 白子
│  │  ├─ Shiroko.jpeg
│  ├─ 阿洛娜
│  │  ├─ arona.jpg
│  ├─ BG_CS_Abydos_01.jpg
│  ├─ BG_CS_Abydos_10.jpg
```
#### 测试搜索图片search_pic的目录结构:
```
.
├─ search_pic        # 用于测试搜索功能（内部图片此处不展示）
```

## 3.进度
#### 3.1 本地图片搜索
- [x] 精确搜索
- [x] 模糊搜索
- [x] 局部搜索
- [ ] 优化搜索算法、使用VGG16等深度学习模型
#### 3.2 网络图片搜索
- [ ] 网络图片搜索
#### 3.3 GUI
- [ ] 图形化界面