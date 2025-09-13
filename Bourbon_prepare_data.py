# -*- coding:utf-8 -*-
#############首先把所有的文件提取出来
import re
import random
import shutil
import json
import os

import numpy as np
import PIL.Image
import PIL.ImageDraw
from PIL import Image


# 定义翻转操作函数
def flip_images(image_path, output_folder):
    image = Image.open(image_path)
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)

    # 保存初始图片
    image.save(os.path.join(output_folder, f"{name}_original{ext}"))

    # 向上翻转
    flipped_up = image.transpose(Image.FLIP_TOP_BOTTOM)
    flipped_up.save(os.path.join(output_folder, f"{name}_flip_up{ext}"))

    # 向右翻转
    flipped_right = image.transpose(Image.FLIP_LEFT_RIGHT)
    flipped_right.save(os.path.join(output_folder, f"{name}_flip_right{ext}"))

    # 先向上再向右翻转
    flipped_up_right = flipped_up.transpose(Image.FLIP_LEFT_RIGHT)
    flipped_up_right.save(os.path.join(output_folder, f"{name}_flip_up_right{ext}"))
label_colors = {
    "signal": (255, 0, 0)
    # 添加更多标签及其颜色
}

# 绿色的
label_colors2 = {
    "vessel": (0, 255, 0),
    "hole": (0, 0, 255)
    # 添加更多标签及其颜色
}

def json_to_png(json_path, output_path):
    with open(json_path) as f:
        data = json.load(f)

    shapes = data['shapes']
    image_height = data['imageHeight']
    image_width = data['imageWidth']

    # 创建一个空白图像
    img = PIL.Image.new('RGB', (image_width, image_height), (0, 0, 0))
    draw = PIL.ImageDraw.Draw(img)

    for shape in shapes:
        label = shape['label']
        points = shape['points']
        polygon = [(int(x), int(y)) for x, y in points]

        if label in label_colors:
            color = label_colors[label]
        else:
            # 默认颜色（黑色）
            color = (0, 0, 0)

        draw.polygon(polygon, outline=color, fill=color)

    img.save(output_path)

def json_to_png2(json_path, output_path):
    with open(json_path) as f:
        data = json.load(f)

    shapes = data['shapes']
    image_height = data['imageHeight']
    image_width = data['imageWidth']

    # 创建一个空白图像
    img = PIL.Image.new('RGB', (image_width, image_height), (0, 0, 0))
    draw = PIL.ImageDraw.Draw(img)

    for shape in shapes:
        label = shape['label']
        points = shape['points']
        polygon = [(int(x), int(y)) for x, y in points]

        if label in label_colors2:
            color = label_colors2[label]
        else:
            # 默认颜色（黑色）
            color = (0, 0, 0)

        draw.polygon(polygon, outline=color, fill=color)

    img.save(output_path)

def sanitize_filename(filename):
    # 删除文件名中的空格
    return filename.replace(' ', '')

def remove_fixed_part(file_name):
    # 使用正则表达式匹配固定部分和后面的数字
    pattern = r'-ImageExport-\d+'
    # 替换匹配到的部分为空字符串
    new_file_name = re.sub(pattern, '', file_name)
    return new_file_name

def move_and_rename_files1(src_dir, dest_dir_json, dest_dir_png):
    # 创建目标文件夹（如果不存在）
    os.makedirs(dest_dir_json, exist_ok=True)
    os.makedirs(dest_dir_png, exist_ok=True)

    # 遍历源文件夹中的所有文件和子文件夹
    for root, dirs, files in os.walk(src_dir):
        # 分割路径
        rel_path = os.path.relpath(root, src_dir)
        path_parts = rel_path.split(os.sep)


        if len(path_parts) >= 2:

            folder2 = path_parts[0]
            folder3 = path_parts[1] if len(path_parts) > 1 else ''

            for file in files:
                # 检查文件是否以.json或2.png结尾
                if file.endswith('.json') or file.endswith('_c1-2.png'):
                    print("111")
                    src_file_path = os.path.join(root, file)
                    # 构建新的文件名
                    new_file_name = f"{folder2}_{file}"
                    new_file_name = sanitize_filename(new_file_name)
                    new_file_name = remove_fixed_part(new_file_name)
                    if file.endswith('.json'):
                        dest_file_path = os.path.join(dest_dir_json, new_file_name)
                    elif file.endswith('_c1-2.png'):
                        dest_file_path = os.path.join(dest_dir_png, new_file_name)
                        print(new_file_name)
                        print(dest_file_path)

                    # 移动文件
                    shutil.copy(src_file_path, dest_file_path)
                    # print(f"Moved {src_file_path} to {dest_file_path}")

def move_and_rename_files2(src_dir, dest_dir_json, dest_dir_png):
    # 创建目标文件夹（如果不存在）
    os.makedirs(dest_dir_json, exist_ok=True)
    os.makedirs(dest_dir_png, exist_ok=True)

    # 遍历源文件夹中的所有文件和子文件夹
    for root, dirs, files in os.walk(src_dir):
        # 分割路径
        rel_path = os.path.relpath(root, src_dir)
        path_parts = rel_path.split(os.sep)


        if len(path_parts) >= 2:

            folder2 = path_parts[0]
            folder3 = path_parts[1] if len(path_parts) > 1 else ''

            for file in files:
                # 检查文件是否以.json或2.png结尾
                if file.endswith('.json') or file.endswith('_c2.png'):
                    print("111")
                    src_file_path = os.path.join(root, file)
                    # 构建新的文件名
                    new_file_name = f"{folder2}_{file}"
                    new_file_name = sanitize_filename(new_file_name)
                    new_file_name = remove_fixed_part(new_file_name)
                    if file.endswith('.json'):
                        dest_file_path = os.path.join(dest_dir_json, new_file_name)
                    elif file.endswith('_c2.png'):
                        dest_file_path = os.path.join(dest_dir_png, new_file_name)
                        print(new_file_name)
                        print(dest_file_path)

                    # 移动文件
                    shutil.copy(src_file_path, dest_file_path)
                    # print(f"Moved {src_file_path} to {dest_file_path}")
#####################################################################
src_directory1 = 'Bourbon_tiqushuju//shentou'  # 替换为你的源文件夹路径
json_directory1 = 'Bourbon_tiqushuju//json_red'  # 替换为存放.json文件的目标文件夹路径
png_directory1 = 'Bourbon_tiqushuju//merge'  # 替换为存放2.png文件的目标文件夹路径
src_directory2 = 'Bourbon_tiqushuju//xueguan'  # 替换为你的源文件夹路径
json_directory2 = 'Bourbon_tiqushuju//json_green'  # 替换为存放.json文件的目标文件夹路径
png_directory2 = 'Bourbon_tiqushuju//green'  # 替换为存放2.png文件的目标文件夹路径

move_and_rename_files1(src_directory1, json_directory1, png_directory1)
move_and_rename_files2(src_directory2, json_directory2, png_directory2)

dl1 = os.listdir("Bourbon_tiqushuju//json_red")
os.makedirs("Bourbon_tiqushuju//label_red")
os.makedirs("Bourbon_tiqushuju//label_green")
for namej in dl1:
    json_to_png("Bourbon_tiqushuju//json_red//"+namej,"Bourbon_tiqushuju//label_red//"+namej[0:len(namej)-5]+".png")

dl2 = os.listdir("Bourbon_tiqushuju//json_green")
for namej in dl2:
    json_to_png2("Bourbon_tiqushuju//json_green//"+namej,"Bourbon_tiqushuju//label_green//"+namej[0:len(namej)-5]+".png")

folder1 = png_directory1
folder2 = "Bourbon_tiqushuju//label_red"
folder3 = "Bourbon_tiqushuju//label_green"
folder4 = png_directory2



# 获取所有图片文件名
images1 = sorted([f for f in os.listdir(folder1) if f.endswith('.png')])
images2 = sorted([f for f in os.listdir(folder2) if f.endswith('.png')])
images3 = sorted([f for f in os.listdir(folder3) if f.endswith('.png')])
images4 = sorted([f for f in os.listdir(folder4) if f.endswith('.png')])

# 检查图片数量是否一致
assert len(images1) == len(images2) == len(images3) == len(images4)

# 获取基础名称（去掉"_c1"和"_c2"部分）
base_names_c1 = [f.split('_c1')[0] for f in images1]
base_names_c2 = [f.split('_c2')[0] for f in images3]

# 确认基础名称匹配
assert base_names_c1 == base_names_c2,"buyizhi"

print(base_names_c1)
# 生成随机顺序
indices = list(range(len(base_names_c1)))
random.shuffle(indices)

# 划分数据集，70%训练，30%测试
split_index = int(0.7 * len(indices))
train_indices = indices[:split_index]
test_indices = indices[split_index:]

# 创建输出目录
train_folder1 = 'Bourbon_tiqushuju//train//merge'
train_folder2 = 'Bourbon_tiqushuju//train//label_red'
train_folder3 = 'Bourbon_tiqushuju//train//label_green'
train_folder4 = 'Bourbon_tiqushuju//train//green'
test_folder1 = 'Bourbon_tiqushuju//test//merge'
test_folder2 = 'Bourbon_tiqushuju//test//label_red'
test_folder3 = 'Bourbon_tiqushuju//test//label_green'
test_folder4 = 'Bourbon_tiqushuju//test//green'
os.makedirs(train_folder1, exist_ok=True)
os.makedirs(train_folder2, exist_ok=True)
os.makedirs(train_folder3, exist_ok=True)
os.makedirs(train_folder4, exist_ok=True)
os.makedirs(test_folder1, exist_ok=True)
os.makedirs(test_folder2, exist_ok=True)
os.makedirs(test_folder3, exist_ok=True)
os.makedirs(test_folder4, exist_ok=True)

# 移动图片到对应的文件夹
for idx in train_indices:
    shutil.copy(os.path.join(folder1, images1[idx]), train_folder1)
    shutil.copy(os.path.join(folder2, images2[idx]), train_folder2)
    shutil.copy(os.path.join(folder3, images3[idx]), train_folder3)
    shutil.copy(os.path.join(folder4, images4[idx]), train_folder4)

for idx in test_indices:
    shutil.copy(os.path.join(folder1, images1[idx]), test_folder1)
    shutil.copy(os.path.join(folder2, images2[idx]), test_folder2)
    shutil.copy(os.path.join(folder3, images3[idx]), test_folder3)
    shutil.copy(os.path.join(folder4, images4[idx]), test_folder4)

##################翻转
# 输出文件夹路径
augmented_folder1 = 'Bourbon_tiqushuju//split//merge'
augmented_folder2 = 'Bourbon_tiqushuju//split//label_red'
augmented_folder3 = 'Bourbon_tiqushuju//split//label_green'
augmented_folder4 = 'Bourbon_tiqushuju//split//green'

# 创建输出文件夹
os.makedirs(augmented_folder1, exist_ok=True)
os.makedirs(augmented_folder2, exist_ok=True)
os.makedirs(augmented_folder3, exist_ok=True)
os.makedirs(augmented_folder4, exist_ok=True)

# 对train_folder1的所有图片进行翻转
for image_name in os.listdir(train_folder1):
    image_path = os.path.join(train_folder1, image_name)
    flip_images(image_path, augmented_folder1)

# 对train_folder2的所有图片进行翻转
for image_name in os.listdir(train_folder2):
    image_path = os.path.join(train_folder2, image_name)
    flip_images(image_path, augmented_folder2)

# 对train_folder3的所有图片进行翻转
for image_name in os.listdir(train_folder3):
    image_path = os.path.join(train_folder3, image_name)
    flip_images(image_path, augmented_folder3)

# 对train_folder4的所有图片进行翻转
for image_name in os.listdir(train_folder4):
    image_path = os.path.join(train_folder4, image_name)
    flip_images(image_path, augmented_folder4)