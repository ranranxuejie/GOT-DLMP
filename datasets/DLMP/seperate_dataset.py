# 读取org_imgs文件
import os
import shutil
import re
org_imgs_path = '/mnt/d/PycharmProjects/2024B/FOTS.PyTorch/datasets/DLMP/org_imgs'
# org_imgs = os.listdir(org_imgs_path)
# # 筛选出以.jpg结尾的文件
# org_imgs = [img for img in org_imgs if img.endswith('.jpg')]

# 读取Label.txt文件
label_path = '/mnt/d/PycharmProjects/2024B/FOTS.PyTorch/datasets/DLMP/org_imgs/Label.txt'
label_file = open(label_path, 'r')
label_file = label_file.readlines()
img_names = [label.split('\t')[0].split('/')[1] for label in label_file]
train_rate = 0.2
test_imgs = img_names[:int(len(img_names) * train_rate)]
train_imgs = img_names[int(len(img_names) * train_rate):]

# 复制train_imgs到train文件夹
org_path = '/mnt/d/PycharmProjects/2024B/FOTS.PyTorch/datasets/DLMP'
# 删除原有图片文件夹下的文件
for img in os.listdir(os.path.join(org_path, 'train')):
    os.remove(os.path.join(org_path, 'train', img))
for img in os.listdir(os.path.join(org_path, 'test')):
    os.remove(os.path.join(org_path, 'test', img))
for img in train_imgs:
    shutil.copy(os.path.join(org_imgs_path, img), os.path.join(org_path, 'train'))
for img in test_imgs:
    shutil.copy(os.path.join(org_imgs_path, img), os.path.join(org_path, 'test'))
test_labels = label_file[:int(len(label_file) * train_rate)]
train_labels = label_file[int(len(label_file) * train_rate):]
# 复制train_labels到train文件夹
with open(os.path.join(org_path, 'train', 'Label.txt'), 'w') as f:
    for label in train_labels:
        f.write(label)
with open(os.path.join(org_path, 'test', 'Label.txt'), 'w') as f:
    for label in test_labels:
        f.write(label)
