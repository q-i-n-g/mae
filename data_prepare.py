import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
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


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    # Create a new black image with size 2 * image_size
    new_size = 2 * image_size
    new_image = Image.new('RGB', (new_size, new_size), (0, 0, 0))

    # Calculate the position to paste the original image at the center
    left = (new_size - pil_image.width) // 2
    top = (new_size - pil_image.height) // 2
    new_image.paste(pil_image, (left, top))

    pil_image = new_image
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def resize_images(source_dir, target_dir, size):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    for root, dirs, files in os.walk(source_dir):
        # 获取相对路径，用于在目标目录中创建相同的目录结构
        relative_path = os.path.relpath(root, source_dir)
        target_path = os.path.join(target_dir, relative_path)
        
        # 确保目标子目录存在
        if not os.path.exists(target_path):
            os.makedirs(target_path)
            
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                image = Image.open(os.path.join(root, file))
                resized_image = center_crop_arr(image, size)
                # 保存到对应的子目录中
                resized_image.save(os.path.join(target_path, file))
                print(f"已处理: {os.path.join(relative_path, file)}")

source_directory = "Data" 
target_directory = "Data/combined_images"  
final_directory = "Data/final_images"

combine_images(source_directory, target_directory)
train_val_split(target_directory)
resize_images(target_directory, final_directory, 224)