import os
import json
import difflib
import Levenshtein
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd


# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为 SimHei
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

try:
    os.chdir('./datasets/DLMP/1-原图未裁剪未扩充数据集')
except:
    pass
new_label_path = '../../../result.json'
new_labels = json.load(open(new_label_path))
similarity_df = pd.read_csv('similarity.csv',encoding='utf-8')
similarity_df = []
for similarity_method in [0,1]:
    similarities= []
    for img_txt in os.listdir('./labels_old'):
        img_n = int(img_txt.split('.')[0].split('_')[-1])+1
        img_txt_file = open('./labels_old/'+img_txt,'r')
        img_txt_lines = [line.split(',')[-1].replace('\n','') for line in img_txt_file.readlines()]
        new_img_name = f'DLMP{int(img_n):03d}.jpg'
        new_label = new_labels[new_img_name]
        old_label = ''.join(img_txt_lines)
        # 新增相似度比较
        if similarity_method == 0:
            matcher = difflib.SequenceMatcher(None, old_label, new_label)
            similarity = matcher.ratio() * 100  # 转换为百分比
        elif similarity_method == 1:
            lev_dist = Levenshtein.distance(old_label, new_label)
            similarity = 100 - (lev_dist / max(len(old_label), len(new_label)) * 100)
        print(f"{img_txt}相似度: {similarity:.2f}%")
        similarities.append(similarity)
    similarity_df.append(similarities)
    # 直方图
    plt.subplots(nrows=1, ncols=1,figsize=(6, 4),dpi=300)
    # 绘制直方图
    plt.hist(similarities, bins=50, edgecolor='black')
    plt.xlabel('相似度')
    plt.ylabel('频率')
    # plt.title('Histogram of Similarity')
    plt.savefig(f'similarity_histogram_{similarity_method}.png')
    plt.show()
similarity_df = pd.DataFrame(similarity_df)
similarity_df.to_csv('similarity.csv',index=False)
