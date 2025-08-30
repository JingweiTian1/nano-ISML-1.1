# -*- coding:utf-8 -*-
import re
import shutil
import os


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

src_directory1 = 'Bourbon_tiqushuju//tian2-merge'
json_directory1 = 'Bourbon_tiqushuju//predict_merge'
png_directory1 = 'Bourbon_tiqushuju//predict_merge'
src_directory2 = 'Bourbon_tiqushuju//tian2-CD31'
json_directory2 = 'Bourbon_tiqushuju//predict_green'
png_directory2 = 'Bourbon_tiqushuju//predict_green'

move_and_rename_files1(src_directory1, json_directory1, png_directory1)
move_and_rename_files2(src_directory2, json_directory2, png_directory2)
