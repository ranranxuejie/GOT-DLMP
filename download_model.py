
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import AutoModel
model_name = "stepfun-ai/GOT-OCR2_0"
model = AutoModel.from_pretrained(model_name,trust_remote_code=True)
model.save_pretrained("./GOT_weights")
