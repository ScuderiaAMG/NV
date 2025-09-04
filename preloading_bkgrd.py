import os
from PIL import Image

def resize_images(input_dir, target_size=(640, 640)):
    """
    将指定目录内所有图片统一调整为指定分辨率
    
    参数:
    input_dir: 图片所在目录路径
    target_size: 目标分辨率 (宽度, 高度)
    """
    # 确保目录存在
    if not os.path.isdir(input_dir):
        raise ValueError(f"目录不存在: {input_dir}")
    
    # 支持的图片格式
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    
    # 遍历目录处理图片
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(image_extensions):
            filepath = os.path.join(input_dir, filename)
            try:
                with Image.open(filepath) as img:
                    # 直接调整尺寸（不保持比例）
                    resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                    
                    # 保存覆盖原文件（保持原始格式）
                    resized_img.save(filepath, quality=95)
                    print(f"已处理: {filename} -> {target_size}")
                    
            except Exception as e:
                print(f"处理失败 {filename}: {str(e)}")

# 使用示例
if __name__ == "__main__":
    resize_images("/home/legion/dataset/trash/empty")