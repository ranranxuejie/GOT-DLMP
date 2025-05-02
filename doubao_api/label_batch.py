import json
import os
from PIL import Image
import io
try:
    os.chdir('./doubao_api')
except:
    print('Already in doubao_api')
from volcenginesdkarkruntime import Ark
import base64
from tqdm import tqdm
from md2latex import md_to_latex_table,template

image_folder = "./val_imgs/"

finished_list = [file.split('.')[0]+".jpg" for file in os.listdir('./latex')]
# 请确保您已将 API Key 存储在环境变量 ARK_API_KEY 中
# 初始化Ark客户端，从环境变量中读取您的API Key
client = Ark(
    # 此为默认路径，您可根据业务所在地域进行配置
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    # 从环境变量中获取您的 API Key。此为默认方式，您可根据需要进行修改
    api_key="fc4479b9-0f72-43b4-8b87-a72dfb07cd0b",
)
# 定义方法将指定路径图片转为Base64编码


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        img_data = image_file.read()
        # 强制转换为RGB格式的JPEG图片
        img = Image.open(io.BytesIO(img_data))
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        # 统一保存为JPEG格式
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        img_data = buffer.getvalue()
        return base64.b64encode(img_data).decode('utf-8')

for file in tqdm(os.listdir(image_folder)):
    if not file.endswith(".jpg"):
        continue
    if file in finished_list:
        continue
    # print(f'Processing {file}')
    # 需要传给大模型的图片
    image_path = f'{image_folder}{file}'
    IMAGE_FORMAT=image_path.split(".")[-1]
    IMAGE_FORMAT = 'jpeg' if IMAGE_FORMAT == 'jpg' else IMAGE_FORMAT
    # 将图片转为Base64编码
    base64_image = encode_image(image_path)
    try:
        response = client.chat.completions.create(
            # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
            model="doubao-1-5-vision-pro-32k-250115",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请仔细识别以下表格文本，注意准确分辨行列结构、各单元格的内容，包括数字、文字、特殊符号等。对于模糊或难以辨认的部分，请基于上下文合理推断。确保输出的识别结果格式清晰、内容完整且准确无误，若有表头，需准确提取表头信息。请注意识别表格上方的设备名称，并将其放在输出的表格内。"},
                        {
                            "type": "image_url",
                            "image_url": {
                                # "url": "https://ark-project.tos-cn-beijing.volces.com/images/view.jpeg"
                                "url":  f"data:image/{IMAGE_FORMAT};base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
        )
    except Exception as e:
        print(f"Error: {e}")
        continue
    response_js = json.loads(response.choices[0].to_json())
    text = response_js['message']['content']
    latex_table = md_to_latex_table(text)
    latex_json = template((latex_table,file))
    with open(f"./new_labels/{file.split(".")[0]}.json","w",encoding="utf-8") as f:
        json.dump(response_js,f,ensure_ascii=False,indent=4)
    with open(f"./latex/{file.split(".")[0]}.json","w",encoding="utf-8") as f:
        json.dump(latex_json,f,ensure_ascii=False,indent=4)
