import os
import shutil

# 定义路径
org_imgs_path = r"D:\PycharmProjects\2024B\GOT-DLMP\datasets\DLMP\org_imgs"
new_labels_path = r"D:\PycharmProjects\2024B\GOT-DLMP\doubao_api\new_labels"
output_path = r"D:\PycharmProjects\2024B\GOT-DLMP\datasets\DLMP\val_imgs"

# 确保输出目录存在
os.makedirs(output_path, exist_ok=True)

# 获取两个目录的文件名（不含后缀）
def get_filenames(path):
    return {os.path.splitext(f)[0] for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))}

org_files = get_filenames(org_imgs_path)
label_files = get_filenames(new_labels_path)

# 找出org_imgs中有但new_labels中没有的文件
extra_files = org_files - label_files

# 复制这些文件到val_imgs目录
for filename in extra_files:
    # 查找原始文件（考虑可能有不同后缀）
    for f in os.listdir(org_imgs_path):
        if os.path.splitext(f)[0] == filename:
            src = os.path.join(org_imgs_path, f)
            dst = os.path.join(output_path, f)
            shutil.copy2(src, dst)
            print(f"已复制: {f}")
            break

print(f"完成！共复制了{len(extra_files)}个文件到{output_path}")
