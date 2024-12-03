import os
import shutil
from pathlib import Path

def combine_images(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # 检查文件是否为图片
            if file.lower().endswith(image_extensions):
                # 获取源文件的完整路径
                source_file = os.path.join(root, file)
                # 获取目标文件的完整路径
                target_file = os.path.join(target_dir, file)
                
                # 如果目标文件已存在，添加数字后缀
                if os.path.exists(target_file):
                    filename, extension = os.path.splitext(file)
                    counter = 1
                    while os.path.exists(target_file):
                        new_filename = f"{filename}_{counter}{extension}"
                        target_file = os.path.join(target_dir, new_filename)
                        counter += 1
                
                # 移动文件
                shutil.move(source_file, target_file)
                print(f"已移动: {file} -> {target_file}")

def train_val_split(target_dir):
    train_dir = os.path.join(target_dir, "train/all")
    val_dir = os.path.join(target_dir, "val/all")
    
    # 创建训练集和验证集目录
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # 获取目标目录中的所有图片
    image_files = [f for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f)) and 
                   f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
    
    # 打乱图片顺序
    import random
    random.shuffle(image_files)
    
    # 计算验证集的大小（1/6的总数）
    val_size = len(image_files) // 10
    
    # 分割文件列表
    val_files = image_files[:val_size]
    train_files = image_files[val_size:]
    
    # 移动文件到相应目录
    for file in train_files:
        src = os.path.join(target_dir, file)
        dst = os.path.join(train_dir, file)
        shutil.move(src, dst)
        print(f"移动到训练集: {file}")
        
    for file in val_files:
        src = os.path.join(target_dir, file)
        dst = os.path.join(val_dir, file)
        shutil.move(src, dst)
        print(f"移动到验证集: {file}")
    
    print(f"总共处理了 {len(image_files)} 张图片")
    print(f"训练集: {len(train_files)} 张图片")
    print(f"验证集: {len(val_files)} 张图片")

source_directory = "Data" 
target_directory = "Data/combined_images"  

combine_images(source_directory, target_directory)
train_val_split(target_directory)
