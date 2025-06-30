import base64
import os
# 通过 pip install volcengine-python-sdk[ark] 安装方舟SDK
from volcenginesdkarkruntime import Ark

# 初始化一个Client对象，从环境变量中获取API Key
client = Ark(
    api_key='df0d5567-6c5b-4ce0-bef0-a93f5a044e13',
)


# 定义方法将指定路径图片转为Base64编码
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# 需要传给大模型的图片
image_path = "F:/Picture/pixiv/BA/119811200_p0.png"

# 将图片转为Base64编码
base64_image = encode_image(image_path)

response = client.chat.completions.create(
    # 替换 <MODEL> 为模型的Model ID
    model="doubao-1-5-thinking-vision-pro-250428",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        # 需要注意：传入Base64编码前需要增加前缀 data:image/{图片格式};base64,{Base64编码}：
                        # PNG图片：
                        "url":  f"data:image/png;base64,{base64_image}"
                        # JPEG图片："url":  f"data:image/jpeg;base64,{base64_image}"
                        # WEBP图片："url":  f"data:image/webp;base64,{base64_image}"
                        # "url": f"data:image/jpg;base64,{base64_image}"
                    },
                },
                {
                    "type": "text",
                    "text": "图片里面的角色是谁?你只需要回答名字和出处",
                },
            ],
        }
    ],
)

print(response.choices[0])
