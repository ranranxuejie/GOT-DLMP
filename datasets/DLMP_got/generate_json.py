import json
import pandas as pd
import os
js_list = []
for file in os.listdir('./latex'):
    print(file)
    js = json.load(open('./latex/'+file,'r',encoding='utf-8'))
    js_list.append(js)
with open('./DLMP_got.json','w') as f:
    json.dump(js_list,f,ensure_ascii=True,indent=4)
