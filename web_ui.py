#%%
import datetime
from datetime import time

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
from GOT.demo.process_results import punctuation_dict, svg_to_html
from PIL import Image
translation_table = str.maketrans(punctuation_dict)

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
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            output_ids = model.generate(
                input_ids=input_ids,
                images=[(image_tensor.unsqueeze(0).half().cuda(), image_tensor_1.unsqueeze(0).half().cuda())],
                do_sample=False,
                num_beams=1,
                no_repeat_ngram_size=20,
                max_new_tokens=4096,
                stopping_criteria=[stopping_criteria],
                streamer=streamer
            )
        # buffer = ""
        # for new_text in streamer:
        #     buffer += new_text
        #     # 去除可能的结束字符串
        #     if buffer.endswith(stop_str):
        #         buffer = buffer[:-len(stop_str)]
        #     buffer = buffer.strip()
        #     if buffer:
        #         # 流式输出当前处理好的文本
        #         yield buffer
        #         buffer = ""
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        return outputs
        #     # 流式输出结果
        # generated_text = ""
        # for new_text in streamer:
        #     generated_text += new_text
        #     yield generated_text  # 逐步返回生成结果

    except Exception as e:
        print(f'an error occurred: {e}')

def render_result(outputs):
    if '【渲染结果已保存】' in outputs:
        return outputs
    else :
        print('Rendering……',end='')
        outputs_ = outputs + '\n【渲染结果已保存】'
        # 获取当前时间
        current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        html_path_2 = f"./results/{current_time}.html"
    if '\\begin{tikzpicture}' not in outputs:
        html_path = "./render_tools/" + "/content-mmd-to-html.html"

        right_num = outputs.count('\\right')
        left_num = outputs.count('\left')

        if right_num != left_num:
            outputs = outputs.replace('\left(', '(').replace('\\right)', ')').replace('\left[', '[').replace(
                '\\right]', ']').replace('\left{', '{').replace('\\right}', '}').replace('\left|', '|').replace(
                '\\right|', '|').replace('\left.', '.').replace('\\right.', '.')

        outputs = outputs.replace('"', '``').replace('$', '')

        outputs_list = outputs.split('\n')
        gt = ''
        for out in outputs_list:
            gt += '"' + out.replace('\\', '\\\\') + r'\n' + '"' + '+' + '\n'

        gt = gt[:-2]

        with open(html_path, 'r') as web_f:
            lines = web_f.read()
            lines = lines.split("const text =")
            new_web = lines[0] + 'const text =' + gt + lines[1]
    else:
        html_path = "./render_tools/" + "/tikz.html"

        outputs = outputs.translate(translation_table)
        outputs_list = outputs.split('\n')
        gt = ''
        for out in outputs_list:
            if out:
                if '\\begin{tikzpicture}' not in out and '\\end{tikzpicture}' not in out:
                    while out[-1] == ' ':
                        out = out[:-1]
                        if out is None:
                            break

                    if out:
                        if out[-1] != ';':
                            gt += out[:-1] + ';\n'
                        else:
                            gt += out + '\n'
                else:
                    gt += out + '\n'

    with open(html_path, 'r') as web_f:
        lines = web_f.read()
        lines = lines.split("const text =")
        new_web = lines[0] +"const text ="+ gt + lines[1]

    with open(html_path_2, 'w') as web_f_new:
        web_f_new.write(new_web)
    print('\t\tDone!')
    return outputs_

with gr.Blocks(title="OCR识别系统") as demo:
    with gr.Row():
        with gr.Column():
            # 新增模型选择组件
            model_selector = gr.Dropdown(
                choices=["GOT_weights/", "results/dlmp/"],
                value="results/dlmp/",
                label="选择模型"
            )
            load_btn = gr.Button("加载模型")
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="filepath", label="上传图片")
            text_input = gr.Textbox(label="附加描述（可选）")
            process_btn = gr.Button("开始处理")
        with gr.Column():
            output_text = gr.Textbox(label="识别结果", interactive=False, lines=20)
            render_btn = gr.Button("渲染结果")

    # 最后一个退出整个进程
    exit_btn = gr.Button("退出")


    # 事件处理新增加载模型逻辑
    def load_model(model_path):
        global model_name, template, tokenizer, model, image_processor, image_processor_high, conv, input_ids, stopping_criteria,stop_str

        template = "mpt"
        model_name = os.path.expanduser(model_path)

        print(f'\t\tDone!\nloading model……', end='')
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
        return "模型加载成功！"
    # 绑定新事件
    load_btn.click(
        fn=load_model,
        inputs=model_selector,
        outputs=output_text
    )

    # 绑定点击事件
    process_btn.click(
        fn=eval_model,
        inputs=img_input,
        outputs=output_text
    )
    render_btn.click(
        fn=render_result,
        inputs=output_text,
        outputs=output_text
    )
    exit_btn.click(
        fn=demo.close,
        inputs=None,
        outputs=None
    )
demo.launch()
