import json
import pandas as pd
import os
js_list = []
for file in os.listdir('./latex'):
    print(file)
    js = json.load(open('./latex/'+file,'r',encoding='utf-8'))
    js_list.append(js)
with open('./DLMP_got.json','w') as f:
    json.dump(js_list,f,ensure_ascii=True,indent=4,encoding='utf-8')

with open('DLMP_got_lora.jsonl', 'w') as f:
    for js in js_list:
        image_text = js['conversations'][1]['value']
        image_name = js['image']

        js_lora = {"query": "<image>\nOCR_DLMP:", "response": image_text,"images": [fr"\mnt\d\PycharmProjects\2024B\GOT-DLMP\datasets\DLMP_got\org_imgs\{image_name}"]}
        f.write(json.dumps(js_lora,ensure_ascii=False)+'\n')
#%%
with open('./DLMP_got.json','r') as f:
    js_list = json.load(f)
with open('DLMP_got.json', 'w',encoding='utf-8') as f:
    json.dump(js_list,f,indent=4,ensure_ascii=False)
