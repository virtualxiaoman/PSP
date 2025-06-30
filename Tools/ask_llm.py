import base64
import numpy as np
from io import BytesIO
from PIL import Image
import cv2
from volcenginesdkarkruntime import Ark

# 初始化一次即可重复使用
client = Ark(api_key='df0d5567-6c5b-4ce0-bef0-a93f5a044e13')


def image_to_base64(img_np: np.ndarray, format: str = 'JPEG') -> str:
    """
    将NumPy数组转换为Base64编码的图片字符串

    参数:
    img_np: 形状为[H, W, C]的NumPy数组，支持RGB或RGBA格式
    format: 输出格式 ('JPEG', 'PNG'等)

    返回:
    Base64编码的图片字符串
    """
    # 确保数据类型为uint8
    if img_np.dtype != np.uint8:
        img_np = (img_np * 255).astype(np.uint8)

    # 处理RGBA图像转换为RGB
    if img_np.shape[2] == 4:
        img_np = img_np[:, :, :3]

    img = Image.fromarray(img_np)
    buffered = BytesIO()
    img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def ask_about_image(img_np: np.ndarray, prompt: str) -> str:
    """
    向大模型提问关于图片内容的问题

    参数:
    img_np: 形状为[H, W, 3]的RGB图像NumPy数组
    prompt: 提问文本

    返回:
    模型的纯文本回复
    """
    # 转换图片为Base64
    base64_image = image_to_base64(img_np)

    # 构建请求
    response = client.chat.completions.create(
        model="doubao-1-5-thinking-vision-pro-250428",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    )

    # 返回纯文本内容
    return response.choices[0].message.content


# # 读取图片为NumPy数组
# image_path = "F:/Picture/pixiv/BA/119444334_p0.png"
# img = cv2.imread(image_path)
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转为RGB格式
#
# # 调用函数
# response_text = ask_about_image(
#     img_rgb,
#     "图片里面的角色是谁? 你只需要回答名字和出处"
# )
# print(response_text)
