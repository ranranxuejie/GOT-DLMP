import json

import pandas as pd


def template(info):
    response,image_path = info
    return {"image": image_path,
            "conversations":[
                {"from":"human",
                 "value":"<image>\nOCR_DLMP:"},
                {"from":"gpt",
                 "value":response}
            ]}
a = template(("电力变压器\n产品型号 \tS11 - M.RL/10\n标准代号 \tGB1094.1 - 2 - 1996 GB1094.3.5 - 85\n额定容量 \t1600 KVA\n产品代号 \t1LT.710\n额定频率 \t50 Hz\n相数 \t3\n额定电压 \t10000 / 400 V\n短路阻抗 \t% \n绝缘水平 \tL1 75 AC 35 /AC 5\n联结组标号 \tYyn0\n开关 位置 \t高压侧 低压侧\n开关 电压(V)\t10500 10000 9500  400\n冷却方式 \tONAN\n使用条件 \t户外\n器身重 \t850 Kg\n油重 \t850 Kg\n总重 \t1850 Kg\n出厂序号 \t \n广西柳州特种变压器有限责任公司 "
              ,r"DLMP015.jpg"))
b = template(("Hangzhou Changchuan Technology Co.,Ltd.\n杭州长川科技股份有限公司\n设备型号 \t \n电源电压 \t \n出厂编号 \t \n最大电流 \t \n重 量 \t \n气 压 \t \n机台尺寸 \t \n出厂日期 \t \nMADE IN CHINA\nTEL \t0571 - 85096193\nFAX \t0571 - 88830180 ",
              r"DLMP031.jpg"))
c = template(("国韵工美 \tGUOART\n 建筑工地箱 \n 产品型号 \tNT\n 额定电流 \tA\n 额定电压 \t380V/220V\n 额定绝缘电压 \t380V\n 额定短时耐受电流 \t6KA\n 频 率 \t50Hz\n 防护等级 \tIP44\n 标 准 \tGB7251.4\n 生产日期 \t 年 月 \n 生产编号 \t \n 宁波瑞泰电气成套设备有限公司 \nWWW.GUORT.COM 400 - 866 - 9777",
              r"DLMP063.jpg"))
test_data = [a,b,c]
df = pd.read_csv("DLMP_got.csv",encoding="gbk")
for index, row in df.iterrows():
    test_data.append(template((row["response"],row["image_path"])))
# print(test_data)
with open("DLMP_got.json","w",encoding="utf-8") as f:
    json.dump(test_data,f,ensure_ascii=False,indent=4)
