#%%
print('Initializing……',end='')
import gradio as gr
from threading import Thread
import glob  # 添加文件遍历功能
import argparse
import json
from transformers import TextStreamer,TextIteratorStreamer
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
template="mpt"

print(f'\t\tDone!\nloading model……',end='')
model_name = os.path.expanduser(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = GOTQwenForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True,
                                           device_map='cuda', use_safetensors=True, pad_token_id=151643).eval()
model.to(device='cuda', dtype=torch.bfloat16)

image_processor = BlipImageEvalProcessor(image_size=1024)
image_processor_high = BlipImageEvalProcessor(image_size=1024)
# 固定提示词
qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * 256 + DEFAULT_IM_END_TOKEN + '\nOCR with format: '
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
print('\t\tDone!')

def eval_model(img_path):
    try:
        # 单图处理流程
        image = load_image(img_path)
        image_1 = image.copy()
        image_tensor = image_processor(image)
        image_tensor_1 = image_processor_high(image_1)

        print('\t\tDone!')
        print('\tpredicting……',end='')
        print(f"\rResult for {os.path.basename(img_path)}:\n", end='')
        # 创建 TextStreamer 实例
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            generation_kwargs = dict(
                input_ids=input_ids,
                images=[(image_tensor.unsqueeze(0).half().cuda(), image_tensor_1.unsqueeze(0).half().cuda())],
                do_sample=False,
                num_beams=1,
                no_repeat_ngram_size=20,
                max_new_tokens=4096,
                stopping_criteria=[stopping_criteria],
                streamer=streamer
            )
            # 启动生成线程
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            # 流式输出结果
            generated_text = ""
            for new_text in streamer:
                generated_text += new_text
                yield generated_text  # 逐步返回生成结果
        print('Well Done!')
    except Exception as e:
        print(f'an error occurred: {e}')
    # outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    # if outputs.endswith(stop_str):
    #     outputs = outputs[:-len(stop_str)]
    # outputs = outputs.strip()
    # return outputs
    # del image_tensor, output_ids
    # torch.cuda.empty_cache()


with gr.Blocks(title="OCR识别系统") as demo:
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="filepath", label="上传图片")
            text_input = gr.Textbox(label="附加描述（可选）")
            process_btn = gr.Button("开始处理")

        with gr.Column():
            output_text = gr.Textbox(label="识别结果", interactive=False, lines=20)

    # 绑定点击事件
    process_btn.click(
        fn=eval_model,
        inputs=img_input,
        outputs=output_text
    )
demo.launch()
