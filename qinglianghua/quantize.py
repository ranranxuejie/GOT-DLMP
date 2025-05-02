import glob  # 添加文件遍历功能
import argparse
import json
import pickle
import copy
import pandas as pd
from natsort import natsorted
from torch import nn
# from flash_attn import flash_attn_qkvpacked_func  # 添加flash attention导入
from transformers import TextStreamer
from tqdm import tqdm


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from GOT.utils.conversation import conv_templates, SeparatorStyle
from GOT.utils.utils import disable_torch_init
# from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from GOT.model import *
from GOT.utils.utils import KeywordsStoppingCriteria
# from PIL import Image
import os
os.chdir('/mnt/d/PycharmProjects/2024B/GOT-DLMP/')
import requests
from PIL import Image
from io import BytesIO


use_result = False
save_json_path = '../../2025A/new_vue/vue-element-admin/datasets/result.json'
# 使用解析后的参数
# image_folder = '../../2025A/new_vue/vue-element-admin/datasets/backlog'
model_name = "results/dlmp/checkpoint-8000-encoder"
# model_name = "results/dlmp/checkpoint-8000-encoder-int8"
# model_name = "GOT_weights"
# model_name = "results/dlmp-encoder"
image_folder = "datasets/DLMP_got/org_imgs/"    # 直接设置图片文件夹路径
dataset_name = image_folder.split('/')
dataset_name = dataset_name[-2] if dataset_name[-1] == '' else dataset_name[-1]

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

print(f'loading model……',end='')
model_name = os.path.expanduser(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = GOTQwenForCausalLM.from_pretrained(model_name,
       low_cpu_mem_usage=True,
       device_map='auto', use_safetensors=True,
       pad_token_id=151643,
        ).eval()
model.to(device='cuda', dtype=torch.bfloat16)
print('\t\tDone!')
#

# 假设model是已经加载的模型
model_count = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
        param_count = sum(p.numel() for p in module.parameters())
        model_count.append({"Module": name, "Parameter Count": param_count,"Module_real":module})
model_count = pd.DataFrame(model_count)
print(model_count['Parameter Count'].sum())
#%%
from torch.quantization import get_default_qconfig, prepare, convert, float_qparams_weight_only_qconfig

model.to('cpu')  # 将模型移动到 CPU

from torch.quantization import float_qparams_weight_only_qconfig

def selective_quantize(model):
    # 量化视觉编码器的卷积和线性层
    try:
        torch.quantization.quantize_dynamic(
            model.model.vision_tower_high,
            {nn.Conv2d, nn.Linear},
            dtype=torch.qint8,
            inplace=True
        )
    except AssertionError as e:
        print(f"Skipping vision_tower_high due to error: {e}")

    # 量化投影层（单个线性层）
    try:
        torch.quantization.quantize_dynamic(
            model.model.mm_projector_vary,
            {nn.Linear},
            dtype=torch.qint8,
            inplace=True
        )
    except AssertionError as e:
        print(f"Skipping mm_projector_vary due to error: {e}")

    # 量化语言模型中间层（保留首尾层精度）
    for i, layer in enumerate(model.model.layers[4:-4]):  # 跳过前4层和后4层
        try:
            torch.quantization.quantize_dynamic(
                layer,
                {nn.Linear},
                dtype=torch.qint8,
                inplace=True
            )
        except AssertionError as e:
            print(f"Skipping layer {i + 4} due to error: {e}")

    # 量化嵌入层
    try:
        torch.quantization.quantize_dynamic(
            model,
            {nn.Embedding},
            dtype=torch.quint8,
            # qconfig=float_qparams_weight_only_qconfig,
            inplace=True
        )
    except AssertionError as e:
        print(f"Skipping embedding quantization due to error: {e}")

    return model

# 调用函数
model = selective_quantize(model)

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
        print(f"Module: {name}, Weight dtype: {module.weight.dtype}")
model.to('cuda')
model.save_pretrained('results/dlmp/checkpoint-8000-encoder-int8')
tokenizer.save_pretrained('results/dlmp/checkpoint-8000-encoder-int8')
template="mpt"
image_processor = BlipImageEvalProcessor(image_size=1024)
image_processor_high = BlipImageEvalProcessor(image_size=1024)
# 固定提示词
qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * 256 + DEFAULT_IM_END_TOKEN + '\nOCR with format: '  # with format
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
image_files = glob.glob(os.path.join(image_folder, "*.jpg"))  # 添加通配符匹配
for img_path in natsorted(image_files):  # 保持自然排序
    img_name = os.path.basename(img_path)
image = load_image(img_path)
image_1 = image.copy()
image_tensor = image_processor(image)
image_tensor_1 = image_processor_high(image_1)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
with torch.autocast("cuda"):
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
