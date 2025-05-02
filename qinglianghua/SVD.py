#%%
import glob  # 添加文件遍历功能
import argparse
import json
import pickle
import copy
import pandas as pd
from scipy.special.cython_special import modstruve
from torch import nn
from flash_attn import flash_attn_qkvpacked_func  # 添加flash attention导入
from transformers import TextStreamer
from tqdm import tqdm
from torch.quantization import get_default_qconfig, prepare, convert
import time
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoConfig
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
from GOT.model.plug.blip_process import BlipImageEvalProcessor
from transformers import TextStreamer
from natsort import natsorted
import glob

use_result = False
save_json_path = 'results/val/val'
# 使用解析后的参数
image_folder = 'doubao_api/val_imgs'
model_name = "results/dlmp/checkpoint-8000-encoder"
k=int(512*7/8)
k=512
# model_name = "results/dlmp/checkpoint-8000-encoder-int8"
# model_name = "GOT_weights"
# model_name = "results/dlmp-encoder"
# image_folder = "datasets/DLMP_got/org_imgs/"    # 直接设置图片文件夹路径
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
        # use_flash_attention_2=True,
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
# layer = model.model.layers
#%%
class SVDAugmentedModel:
    def __init__(self, original_model, k, quantize_int8=False):
        self.original_model = copy.deepcopy(model)
        self.k = k
        self.quantize_int8 = quantize_int8
        svd_done = self._apply_svd_to_linear_layers()
    def _apply_svd_to_linear_layers(self):
        i=1
        for layer in self.original_model.model.layers:
            print(i)
            i+=1
            # 处理self_attn和mlp的线性层(代码与之前相同)
            for attr in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                k = int(self.k)
                linear_layer = getattr(layer.self_attn, attr)
                weight_matrix = linear_layer.weight.data.float()
                org_x, org_y = weight_matrix.shape
                # 进行 SVD 分解
                U, S, V = torch.svd(weight_matrix)
                # S += 1e-10  # 防止奇异值为零
                # 选取前 k 个奇异值及其对应的奇异向量
                U_k = U[:, :k]
                S_k = S[:k]
                Vt_k = V.t()[:k, :]
                new_layer_1 = nn.Linear(org_x, k, bias=False)
                new_layer_1.weight.data = Vt_k
                if linear_layer.bias is not None:
                    new_layer_2 = nn.Linear(k, org_y, bias=True)
                    new_layer_2.bias.data = linear_layer.bias.data
                else:
                    new_layer_2 = nn.Linear(k, org_y, bias=False)
                new_layer_2.weight.data = torch.mm(torch.diag(S_k), U_k.t()).t()
                new_layer = nn.Sequential(new_layer_1, new_layer_2)
                setattr(layer.self_attn, attr, new_layer)
        torch.cuda.empty_cache()
        return True
    def __getattr__(self, name):
        # 转发所有其他属性访问到原始模型
        return getattr(self.original_model, name)

    def __call__(self, *args, **kwargs):
        # 转发调用到原始模型
        return self.original_model(*args, **kwargs)

# model_svd = SVDAugmentedModel(model, k=k, quantize_int8=False).original_model
model_svd=model

model_count_svd = []
for name, module in model_svd.named_modules():
    if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
        param_count = sum(p.numel() for p in module.parameters())
        model_count_svd.append({"Module": name, "Parameter Count": param_count,"Module_real":module})
model_count_svd = pd.DataFrame(model_count_svd)
print(model_count_svd['Parameter Count'].sum())
#%%
def eval_model(image_folder,template="mpt"):
    try:

        # 图像处理器初始化
        image_processor = BlipImageEvalProcessor(image_size=1024)
        image_processor_high = BlipImageEvalProcessor(image_size=1024)
        # 固定提示词
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * 256 + DEFAULT_IM_END_TOKEN + '\nOCR with format: '# with format
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
            with open(save_json_path,'r') as f:
                result_text = json.load(f)
            if not use_result:
                result_text = {}
        except:
            result_text = {}
        result_time = []

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

            print(f"\rResult for {os.path.basename(img_path)}:\n", end='')

            # 创建 TextStreamer 实例
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            start_time = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_ids = model_svd.generate(
                    input_ids,
                    images=[(image_tensor.unsqueeze(0).half().cuda(), image_tensor_1.unsqueeze(0).half().cuda())],
                    do_sample=False,
                    num_beams=1,
                    no_repeat_ngram_size=20,
                    max_new_tokens=512,
                    stopping_criteria=[stopping_criteria],
                    streamer=streamer
                )
            cost_time = time.time() - start_time
            print(f'end of text, time cost:{cost_time:.2f} s')
            result_time.append([img_name,cost_time])
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
        result_time = pd.DataFrame(result_time,columns=['img_name','time'])
        result_time.to_csv('.results/val/result_time.csv',index=False)
        print('All Well Done!')
    # 按键中断保存结果
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt, saving result.json……',end='')

    finally:
        # 按照键值排序
        result_text = dict(sorted(result_text.items()))
        return result_text


result = eval_model(image_folder,template='mpt')
# 将字典result保存为json文件
with open(f'{save_json_path}_{k}.json', 'w') as f:
     json.dump(result, f,indent=4,ensure_ascii=False)

def calculate_flops(model, input_shape, image_shape=(2, 1, 1024, 1024)):
    total_flops = 0
    linear_flops = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 全连接层FLOPs = 2 * input_dim * output_dim
            flops = 2 * module.in_features * module.out_features
            total_flops += 2 * module.in_features * module.out_features
            linear_flops += 2 * module.in_features * module.out_features
        # elif isinstance(module, nn.Sequential):
        #     # 处理SVD分解后的Sequential结构
        #     if len(module) == 2 and all(isinstance(l, nn.Linear) for l in module):
        #         flops = 2 * module[0].in_features * module[0].out_features+\
        #         2 * module[1].in_features * module[1].out_features
        #         print('sequential',flops)
        #         linear_flops += flops
        elif isinstance(module, nn.Conv2d):
            # 卷积层FLOPs = kernel_h * kernel_w * in_channels * out_channels * output_h * output_w
            output_h = (image_shape[2] + 2 * module.padding[0] - module.kernel_size[0]) // module.stride[0] + 1
            output_w = (image_shape[3] + 2 * module.padding[1] - module.kernel_size[1]) // module.stride[1] + 1

            # 每个位置的计算量 = 核高 * 核宽 * 输入通道数 * 输出通道数
            kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels

            # 总FLOPs = 每个位置计算量 * 输出位置数 * 输出通道数 * 2 (乘加各算一次)
            conv_flops = 2 * kernel_ops * output_h * output_w * module.out_channels

        elif 'Qwen2SdpaAttention' in str(type(module)):
            # 处理自定义注意力层
            if hasattr(module, 'q_proj'):
                # 检查是否是Sequential结构
                if isinstance(module.q_proj, nn.Sequential):
                    # QKV投影(Sequential)
                    total_flops += 3 * (2 * module.q_proj[0].in_features * module.q_proj[0].out_features +
                                       2 * module.q_proj[1].in_features * module.q_proj[1].out_features)
                else:
                    # 原始QKV投影
                    embed_dim = module.q_proj.in_features
                    total_flops += 3 * 2 * embed_dim * embed_dim

                # 注意力计算
                embed_dim = module.q_proj[0].in_features if isinstance(module.q_proj, nn.Sequential) else module.q_proj.in_features
                num_heads = module.num_heads
                head_dim = embed_dim // num_heads
                seq_len = input_shape[1]
                total_flops += 2 * seq_len * seq_len * head_dim * num_heads

                # 输出投影
                if isinstance(module.o_proj, nn.Sequential):
                    total_flops += 2 * module.o_proj[0].in_features * module.o_proj[0].out_features
                    total_flops += 2 * module.o_proj[1].in_features * module.o_proj[1].out_features
                else:
                    total_flops += 2 * embed_dim * embed_dim

        elif 'Qwen2MLP' in str(type(module)):
            # MLP层计算
            total_flops += 2 * module.gate_proj.in_features * module.gate_proj.out_features  # gate_proj
            total_flops += 2 * module.up_proj.in_features * module.up_proj.out_features  # up_proj
            total_flops += 2 * module.down_proj.in_features * module.down_proj.out_features  # down_proj
    # print(linear_flops)
    return total_flops

# 使用示例
input_shape = (1,288)  # (batch_size, seq_len, embed_dim)
vision_input_shape = (2, 1, 1024, 1024)  # 视觉输入形状
total_flops = calculate_flops(model_svd, input_shape)
print(f"Total FLOPs: {(total_flops) / 1e9:.2f} GFLOPs")
