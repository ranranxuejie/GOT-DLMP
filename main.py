#%%
import glob  # 添加文件遍历功能
import argparse
import json
from transformers import TextStreamer
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from GOT.utils.conversation import conv_templates, SeparatorStyle
from GOT.utils.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from GOT.model import *
from GOT.utils.utils import KeywordsStoppingCriteria

from PIL import Image

import os
import requests
from PIL import Image
from io import BytesIO
from GOT.model.plug.blip_process import BlipImageEvalProcessor
from transformers import TextStreamer
from natsort import natsorted
import glob

#
def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'
disable_torch_init()

model_name = "GOT_weights/"

print(f'loading model……',end='')
model_name = os.path.expanduser(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = GOTQwenForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True,
                                           device_map='cuda', use_safetensors=True, pad_token_id=151643).eval()
model.to(device='cuda', dtype=torch.bfloat16)
print('\t\tDone!')


def eval_model(image_folder,template="mpt"):
    try:
        # 图像处理器初始化
        image_processor = BlipImageEvalProcessor(image_size=1024)
        image_processor_high = BlipImageEvalProcessor(image_size=1024)
        # 固定提示词
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * 256 + DEFAULT_IM_END_TOKEN + '\nOCR: '# with format
        # 对话模板
        conv = conv_templates[template].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        inputs = tokenizer([prompt])
        input_ids = torch.as_tensor(inputs.input_ids).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        # 验证文件夹
        if not os.path.exists(image_folder):
            print(f"Image folder {image_folder} does not exist.")
            return
        # 获取并处理所有图片
        image_files = glob.glob(os.path.join(image_folder, "*.jpg"))  # 添加通配符匹配
        try:
            with open('result.json','r') as f:
                result_text = json.load(f)
        except:
            result_text = {}
        for img_path in natsorted(image_files):  # 保持自然排序
            img_name = os.path.basename(img_path)
            if img_name in result_text:
                print(f'{img_name} has been processed, skip!')
                continue
            print(f'========== processing {img_name} ============:')
            print('\tgenerating input tensor……',end='')



            # 单图处理流程
            image = load_image(img_path)
            image_1 = image.copy()
            image_tensor = image_processor(image)
            image_tensor_1 = image_processor_high(image_1)

            print('\t\tDone!')
            print('\tpredicting……',end='')
            print(f"\rResult for {os.path.basename(img_path)}:\n", end='')
            # 创建 TextStreamer 实例
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_ids = model.generate(
                    input_ids,
                    images=[(image_tensor.unsqueeze(0).half().cuda(), image_tensor_1.unsqueeze(0).half().cuda())],
                    do_sample=False,
                    num_beams=1,
                    no_repeat_ngram_size=20,
                    max_new_tokens=512,
                    stopping_criteria=[stopping_criteria],
                    streamer=streamer
                )
            print('end of text')
            # 输出结果
            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            # print(f"\nResult for {img_name}:\n{outputs}\n{'=' * 50}")
            result_text[img_name] = outputs
            print('\t\tDone!')
            # 清空缓存变量
            del image_tensor, output_ids, outputs
            torch.cuda.empty_cache()
        print('All Well Done!')
    # 按键中断保存结果
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt, saving result.json……',end='')

    finally:
        # 按照键值排序
        result_text = dict(sorted(result_text.items()))
        with open('result.json','w') as f:
            json.dump(result_text, f,indent=4)

        return result_text

#%%


image_folder = "datasets/DLMP/org_imgs/"    # 直接设置图片文件夹路径

result = eval_model(image_folder,template='mpt')

# 将字典result保存为json文件
with open('result.json', 'w') as f:
    json.dump(result, f,indent=4,ensure_ascii=False)

# with open('result.json', 'r') as f:
#     result_text = json.load(f)
