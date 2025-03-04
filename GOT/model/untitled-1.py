import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from GOT.model import GOTQwenForCausalLM
from torch import nn

def selective_quantize(model):
    # 量化视觉编码器的卷积和线性层
    torch.quantization.quantize_dynamic(
        model.model.vision_tower_high,
        {nn.Conv2d, nn.Linear},
        dtype=torch.qint8,
        inplace=True
    )

    # 量化投影层（单个线性层）
    torch.quantization.quantize_dynamic(
        model.model.mm_projector_vary,
        {nn.Linear},
        dtype=torch.qint8,
        inplace=True
    )

    # 量化语言模型中间层（保留首尾层精度）
    for layer in model.model.layers[4:-4]:  # 跳过前4层和后4层
        torch.quantization.quantize_dynamic(
            layer,
            {nn.Linear},
            dtype=torch.qint8,
            inplace=True
        )
    return model


# 使用流程
model = GOTQwenForCausalLM.from_pretrained('GOT_weights/', low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=151643)

quantized_model =selective_quantize(model.cpu())
# 保存量化后的模型
save_path = 'GOT_weights_quantized/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

quantized_model.save_pretrained(save_path,safetensors=True)
