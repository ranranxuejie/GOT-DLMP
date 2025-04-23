import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import json
import os
try:
    os.chdir('./GOT/eval')
except:
    pass

import pandas as pd
text_result_0 = pd.read_csv('./org_result/text_results.csv',index_col=0)
text_result_1 = pd.read_csv('./train_result/text_results.csv',index_col=0)
table_result_0 = pd.read_csv('./org_result/table_results.csv',index_col=0)
table_result_1 = pd.read_csv('./train_result/table_results.csv',index_col=0)
bins = 100
for column in text_result_0.columns[:-1]:
    plt.subplots(2,2,figsize=(9,9),dpi=300)
    # 大标题
    plt.suptitle(column)
    plt.subplot(2,2,1)
    plt.hist(text_result_0[column],bins=bins,label='org')
    plt.title('文本（原模型）')
    plt.subplot(2,2,2)
    plt.hist(text_result_1[column],bins=bins,label='train')
    plt.title('文本（训练模型）')
    plt.subplot(2,2,3)
    plt.hist(table_result_0[column],bins=bins,label='org')
    plt.title('表格（原模型）')
    plt.subplot(2,2,4)
    plt.hist(table_result_1[column],bins=bins,label='train')
    plt.title('表格（训练模型）')
    plt.savefig('./plot/'+column+'.png')
    plt.close()
