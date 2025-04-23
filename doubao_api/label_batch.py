import json
import os

from PIL.Image import Image

try:
    os.chdir('./doubao_api')
except:
    print('Already in doubao_api')
from volcenginesdkarkruntime import Ark
import base64
from tqdm import tqdm
from md2latex import md_to_latex_table,template

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
      # 判断image_file 的像素大小，如过大（超过2000*2000），则等比缩放，使最大边为2000
      if image_file.size > 2000*2000:
          image_file = Image.open(image_path)
          image_file.thumbnail((2000, 2000))
          image_file.save(image_path)
image_folder = "./eval_imgs/"
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
                        {"type": "text", "text": "提取表格，并仅仅识别表格内容，不要输出其他内容（包括“未识别”、“无法识别”等）："},
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
