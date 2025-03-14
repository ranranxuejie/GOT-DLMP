import json
import os
try:
    os.chdir('./datasets/DLMP/1-原图未裁剪未扩充数据集')
except:
    pass
from tqdm import tqdm
from PIL import Image
def txt_to_labelme(txt_path, json_path, image_path, image_height, image_width):
    shapes = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            coords = list(map(int, parts[:8]))
            label = parts[8]

            # 转换为LabelMe的坐标格式
            points = []
            for i in range(0, 8, 2):
                points.append([coords[i], coords[i + 1]])

            shapes.append({
                "label": label,
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            })

    # 构建LabelMe JSON结构
    labelme_data = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path,
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width
    }

    with open(json_path, 'w') as f:
        json.dump(labelme_data, f, indent=2)

#%%
# 使用示例
imgs = os.listdir('./images_old')
imgs_name = [i.split('.')[0] for i in imgs]

for i in tqdm(imgs_name):
    # 获取图像尺寸
    image_path = f"./images_old/{i}.jpg"
    image = Image.open(image_path)
    image_width, image_height = image.size
    txt_to_labelme(
        txt_path=f"./labels_old/{i}.txt",
        json_path=f"./labels_json/{i}.json",
        image_path=f"..\\images_old\\{i}.jpg",  # 需与实际图像文件名一致
        image_height=image_height,  # 需根据实际图像尺寸填写
        image_width=image_width  # 需根据实际图像尺寸填写
    )
