import os
import cv2
import torch
import numpy as np
import random
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import tensorflow as tf
from ultralytics import YOLO
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from datetime import datetime
import concurrent.futures
import math
from tqdm import tqdm
import json
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import time
import warnings

# Ignore TensorRT warnings
warnings.filterwarnings("ignore", message="TF-TRT Warning: Could not find TensorRT")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def setup_environment():
    """
    Set up environment and GPU warm-up
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Output hardware information
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # GPU warm-up
    if torch.cuda.is_available():
        warm_up_tensor = torch.randn(1000, 1000).to(device)
        for _ in range(10):
            _ = torch.matmul(warm_up_tensor, warm_up_tensor)
        print("GPU warm-up completed")

    return device

def load_background_images(background_path):
    """
    Load background images
    """
    background_images = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    for filename in os.listdir(background_path):
        if filename.lower().endswith(valid_extensions):
            img_path = os.path.join(background_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                background_images.append(img)
                print(f"Loaded background image: {filename}")

    print(f"Total background images loaded: {len(background_images)}")
    return background_images

def detect_target_center_hough(image):
    """
    Detect the center of a bullseye target using Hough Circle Transform
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=min(image.shape[0], image.shape[1]) // 4
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        best_circle = max(circles, key=lambda x: x[2])
        x, y, r = best_circle
        confidence = calculate_circle_confidence(gray, x, y, r)
        return (x, y, r), confidence

    return None, 0

def detect_target_center_hough_parallel_wrapper(args):
    """
    Wrapper function for parallel Hough circle detection
    """
    image, idx = args
    try:
        target_center, confidence = detect_target_center_hough(image)
        return idx, target_center, confidence
    except Exception as e:
        print(f"Error processing image {idx}: {e}")
        return idx, None, 0

def precompute_background_targets_parallel(background_images, num_processes=None):
    """
    Precompute target centers for background images using parallel processing
    """
    if num_processes is None:
        num_processes = min(8, len(background_images), cpu_count())

    print(f"Using {num_processes} processes for Hough circle detection")

    bg_target_cache = {}
    args = [(bg_img, i) for i, bg_img in enumerate(background_images)]

    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(detect_target_center_hough_parallel_wrapper, args), 
                           total=len(args), desc="Detecting circles in backgrounds"))

    for idx, target_center, confidence in results:
        bg_target_cache[idx] = (target_center, confidence)

        if target_center is not None and idx % 10 == 0:
            x, y, r = target_center
            print(f"Background {idx+1}: Found target at ({x}, {y}) with radius {r}, confidence {confidence:.2f}")
        elif idx % 10 == 0:
            print(f"Background {idx+1}: No target found")

    return bg_target_cache

def calculate_circle_confidence(gray_img, x, y, radius):
    """
    Calculate circle confidence score
    """
    height, width = gray_img.shape

    x_min = max(0, int(x) - int(radius))
    x_max = min(width, int(x) + int(radius))
    y_min = max(0, int(y) - int(radius))
    y_max = min(height, int(y) + int(radius))

    if x_min >= x_max or y_min >= y_max:
        return 0

    roi = gray_img[y_min:y_max, x_min:x_max]

    if roi.size == 0:
        return 0

    edges = cv2.Canny(roi, 50, 150)
    edge_strength = np.mean(edges) if edges.size > 0 else 0
    contrast = np.std(roi) if roi.size > 0 else 0

    confidence = 0.6 * edge_strength + 0.4 * contrast
    return confidence

def augment_dataset(raw_dataset_path, output_path, target_per_class=900):
    """
    Augment dataset with enhanced regularization techniques
    """
    os.makedirs(output_path, exist_ok=True)

    # Enhanced augmentation pipeline with regularization techniques
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=45, p=0.8),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.Blur(blur_limit=3, p=0.3),
        A.CLAHE(p=0.3),
        A.RandomGamma(p=0.3),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        
        # Add regularization augmentations
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),  # Cutout regularization
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),  # Noise injection
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.2),  # Weather variations
        A.RandomShadow(p=0.2),  # Shadow variations
        
        # More aggressive spatial transforms
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.8),
        A.OpticalDistortion(distort_limit=0.2, shift_limit=0.1, p=0.3),
        
        ToTensorV2()
    ])

    class_folders = sorted([f for f in os.listdir(raw_dataset_path) 
                          if os.path.isdir(os.path.join(raw_dataset_path, f))])

    print(f"Found {len(class_folders)} classes to augment")

    for class_folder in class_folders:
        class_path = os.path.join(raw_dataset_path, class_folder)
        output_class_path = os.path.join(output_path, class_folder)
        os.makedirs(output_class_path, exist_ok=True)

        class_images = []
        for filename in os.listdir(class_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    class_images.append(img)

        if not class_images:
            print(f"No images found for class {class_folder}, skipping")
            continue

        print(f"Augmenting class {class_folder} with {len(class_images)} base images")

        existing_count = len(class_images)
        needed_count = target_per_class - existing_count

        if needed_count <= 0:
            print(f"Class {class_folder} already has {existing_count} images, skipping augmentation")
            continue

        augmented_count = 0
        while augmented_count < needed_count:
            for base_img in class_images:
                if augmented_count >= needed_count:
                    break

                try:
                    # Apply multiple augmentation variations for each base image
                    for variation in range(3):  # Generate 3 variations per base image
                        if augmented_count >= needed_count:
                            break
                            
                        augmented = transform(image=base_img)
                        augmented_img = augmented['image']

                        if isinstance(augmented_img, torch.Tensor):
                            augmented_img = augmented_img.permute(1, 2, 0).numpy()
                            augmented_img = (augmented_img * 255).astype(np.uint8)

                        output_filename = f"{class_folder}_aug_{augmented_count:04d}.jpg"
                        output_filepath = os.path.join(output_class_path, output_filename)
                        cv2.imwrite(output_filepath, augmented_img)

                        augmented_count += 1
                        if augmented_count % 100 == 0:
                            print(f"Generated {augmented_count} augmented images for {class_folder}")

                except Exception as e:
                    print(f"Error augmenting image: {e}")
                    continue

        print(f"Completed augmentation for {class_folder}. Total images: {existing_count + augmented_count}")

def process_single_image(class_path, img_filename, class_idx, class_name, 
                        background_images, bg_target_cache, output_path, 
                        img_counter, target_scale):
    """
    Process single image with enhanced randomization for regularization
    """
    # Enhanced randomization for background selection
    bg_idx = random.randint(0, len(background_images) - 1)
    bg_img = background_images[bg_idx].copy()
    bg_height, bg_width = bg_img.shape[:2]

    target_center, confidence = bg_target_cache[bg_idx]

    # Load augmented image
    aug_img_path = os.path.join(class_path, img_filename)
    aug_img = cv2.imread(aug_img_path, cv2.IMREAD_UNCHANGED)

    if aug_img is None:
        return False

    # Enhanced randomization for target placement
    if target_center is None or random.random() < 0.3:  # 30% chance for random placement
        # Random placement for regularization
        target_x = random.randint(aug_img.shape[1] // 2, bg_width - aug_img.shape[1] // 2)
        target_y = random.randint(aug_img.shape[0] // 2, bg_height - aug_img.shape[0] // 2)
        target_radius = min(bg_width, bg_height) // 10
    else:
        target_x, target_y, target_radius = target_center
        # Add small random offset to prevent perfect centering
        offset_range = min(target_radius // 4, 20)
        target_x += random.randint(-offset_range, offset_range)
        target_y += random.randint(-offset_range, offset_range)

    # Enhanced scaling randomization
    aug_height, aug_width = aug_img.shape[:2]
    
    # Variable scaling for regularization
    scale_variation = random.uniform(0.8, 1.2)  # ±20% scale variation
    base_scale_factor = target_scale * scale_variation
    
    # Calculate dynamic scaling based on background size
    bg_min_dim = min(bg_width, bg_height)
    max_scale_factor = bg_min_dim / max(aug_width, aug_height)
    actual_scale_factor = min(base_scale_factor, max_scale_factor * 0.8)  # Conservative upper bound
    
    new_width = max(10, int(aug_width * actual_scale_factor))
    new_height = max(10, int(aug_height * actual_scale_factor))

    if new_width <= 0 or new_height <= 0:
        return False

    aug_img_resized = cv2.resize(aug_img, (new_width, new_height))

    # Enhanced placement randomization
    placement_strategy = random.choice(['center', 'random', 'edge'])
    
    if placement_strategy == 'center':
        start_x = max(0, target_x - new_width // 2)
        start_y = max(0, target_y - new_height // 2)
    elif placement_strategy == 'random':
        start_x = random.randint(0, max(0, bg_width - new_width))
        start_y = random.randint(0, max(0, bg_height - new_height))
    else:  # edge
        edge = random.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top':
            start_x = random.randint(0, bg_width - new_width)
            start_y = 0
        elif edge == 'bottom':
            start_x = random.randint(0, bg_width - new_width)
            start_y = bg_height - new_height
        elif edge == 'left':
            start_x = 0
            start_y = random.randint(0, bg_height - new_height)
        else:  # right
            start_x = bg_width - new_width
            start_y = random.randint(0, bg_height - new_height)

    # Ensure within bounds
    start_x = max(0, min(start_x, bg_width - new_width))
    start_y = max(0, min(start_y, bg_height - new_height))
    
    end_x = start_x + new_width
    end_y = start_y + new_height

    # Apply random image degradation for regularization
    if random.random() < 0.2:  # 20% chance for degradation
        degradation_type = random.choice(['blur', 'noise', 'brightness'])
        if degradation_type == 'blur':
            kernel_size = random.choice([3, 5])
            aug_img_resized = cv2.GaussianBlur(aug_img_resized, (kernel_size, kernel_size), 0)
        elif degradation_type == 'noise':
            noise = np.random.normal(0, random.randint(5, 20), aug_img_resized.shape).astype(np.uint8)
            aug_img_resized = cv2.add(aug_img_resized, noise)
        elif degradation_type == 'brightness':
            brightness_factor = random.uniform(0.7, 1.3)
            aug_img_resized = cv2.convertScaleAbs(aug_img_resized, alpha=brightness_factor, beta=0)

    # Alpha blending
    if aug_img_resized.shape[2] == 4:
        alpha = aug_img_resized[:, :, 3] / 255.0
        alpha = alpha[:, :, np.newaxis]

        bg_region = bg_img[start_y:end_y, start_x:end_x]

        # Ensure dimensions match
        if bg_region.shape[:2] == aug_img_resized.shape[:2]:
            for c in range(3):
                bg_region[:, :, c] = (1 - alpha) * bg_region[:, :, c] + alpha * aug_img_resized[:, :, c]
            bg_img[start_y:end_y, start_x:end_x] = bg_region
    else:
        if bg_img[start_y:end_y, start_x:end_x].shape == aug_img_resized.shape:
            bg_img[start_y:end_y, start_x:end_x] = aug_img_resized

    # Save synthetic image
    image_filename = f"synthetic_{img_counter:06d}.jpg"
    image_path = os.path.join(output_path, "images", image_filename)
    cv2.imwrite(image_path, bg_img)

    # Create YOLO format annotation file
    label_filename = f"synthetic_{img_counter:06d}.txt"
    label_path = os.path.join(output_path, "labels", label_filename)

    # Calculate bounding box with small randomization
    bbox_noise = random.uniform(0.95, 1.05)  # ±5% bbox size variation
    x_center = (start_x + (end_x - start_x) / 2) / bg_width
    y_center = (start_y + (end_y - start_y) / 2) / bg_height
    width = ((end_x - start_x) / bg_width) * bbox_noise
    height = ((end_y - start_y) / bg_height) * bbox_noise

    # Ensure normalized coordinates are valid
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.01, min(1.0, width))
    height = max(0.01, min(1.0, height))

    with open(label_path, 'w') as f:
        f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    return True

def process_single_image_wrapper(args):
    """
    Wrapper function for parallel image processing
    """
    try:
        return process_single_image(*args)
    except Exception as e:
        print(f"Error in process_single_image: {e}")
        return False

def create_synthetic_dataset_parallel(augmented_path, background_images, output_path, target_scale=5.0, 
                                     batch_size=50, max_workers=None):
    """
    Create synthetic dataset with enhanced regularization
    """
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "labels"), exist_ok=True)

    classes = sorted([f for f in os.listdir(augmented_path) 
                     if os.path.isdir(os.path.join(augmented_path, f))])

    print(f"Creating synthetic dataset with {len(classes)} classes")

    # Background preprocessing
    print("Precomputing target centers for background images using parallel processing...")
    start_time = time.time()

    bg_target_cache = precompute_background_targets_parallel(background_images, num_processes=min(8, cpu_count()))

    end_time = time.time()
    print(f"Background preprocessing completed in {end_time - start_time:.2f} seconds")

    # Count total images to process
    total_images = 0
    for class_name in classes:
        class_path = os.path.join(augmented_path, class_name)
        if os.path.exists(class_path):
            class_images = [f for f in os.listdir(class_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            total_images += len(class_images)

    print(f"Total images to process: {total_images}")

    image_counter = 0

    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(augmented_path, class_name)
        if not os.path.exists(class_path):
            print(f"Class path {class_path} does not exist, skipping")
            continue

        class_images = [f for f in os.listdir(class_path) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not class_images:
            print(f"No images found for class {class_name}, skipping")
            continue

        print(f"Processing class {class_name} with {len(class_images)} images")

        args_list = []
        for img_filename in class_images:
            args = (
                class_path, img_filename, class_idx, class_name, 
                background_images, bg_target_cache, output_path, 
                image_counter, target_scale
            )
            args_list.append(args)
            image_counter += 1

        print(f"Processing {len(class_images)} images for class {class_name} using parallel processing")
        start_time = time.time()

        if max_workers is None:
            max_workers = min(8, cpu_count(), len(class_images))

        batch_size = min(100, len(args_list))
        successful_count = 0

        for i in range(0, len(args_list), batch_size):
            batch_args = args_list[i:i+batch_size]

            with Pool(processes=max_workers) as pool:
                batch_results = pool.map(process_single_image_wrapper, batch_args)

            successful_count += sum(batch_results)

            print(f"Processed batch {i//batch_size + 1}/{(len(args_list)-1)//batch_size + 1}, "
                  f"successful: {sum(batch_results)}/{len(batch_args)}")

        end_time = time.time()
        print(f"Processed {successful_count}/{len(class_images)} images for class {class_name} in {end_time - start_time:.2f} seconds")

    print(f"Completed synthetic dataset creation. Total images: {image_counter}")

    # >>>>>>>>>>>>>>>>>> 新增代码开始 <<<<<<<<<<<<<<<<<<
    # 添加纯背景图像以增强模型对整张图片的判别能力
    print("正在向数据集中添加纯背景图像...")
    added_bg_count = add_pure_background_images(
        background_images=background_images,
        output_path=output_path,
        pure_bg_ratio=0.1  # 可根据需要调整，例如0.05表示5%
    )
    image_counter += added_bg_count
    print(f"数据集创建完成。总图像数（含纯背景）: {image_counter}")
    # >>>>>>>>>>>>>>>>>> 新增代码结束 <<<<<<<<<<<<<<<<<<

    # Create classes.txt
    with open(os.path.join(output_path, "classes.txt"), 'w') as f:
        for class_name in classes:
            f.write(f"{class_name}\n")

    return classes, image_counter

def add_pure_background_images(background_images, output_path, pure_bg_ratio=0.1):
    """
    向合成数据集中添加纯背景图像（无任何目标）。
    
    Args:
        background_images (list): 已加载的背景图像列表。
        output_path (str): 合成数据集的根目录路径。
        pure_bg_ratio (float): 纯背景图像占总合成图像数量的比例。例如0.1表示10%。
    """
    import random
    from pathlib import Path

    # 确保输出目录存在
    images_dir = Path(output_path) / "images"
    labels_dir = Path(output_path) / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # 计算当前已有的合成图像总数
    existing_image_count = len(list(images_dir.glob("*.jpg"))) + len(list(images_dir.glob("*.png")))
    
    # 计算需要添加的纯背景图像数量
    num_pure_bg = int(existing_image_count * pure_bg_ratio)
    # 为了安全，确保不会超过可用的背景图数量
    num_pure_bg = min(num_pure_bg, len(background_images))
    
    if num_pure_bg <= 0:
        print("警告: 未添加纯背景图像。请检查合成数据集是否已生成或比例是否过小。")
        return 0

    # 随机选择背景图像索引，避免重复
    selected_bg_indices = random.sample(range(len(background_images)), num_pure_bg)
    
    added_count = 0
    for i, bg_idx in enumerate(tqdm(selected_bg_indices, desc="Adding pure background images")):
        bg_img = background_images[bg_idx]
        img_filename = f"pure_bg_{i:06d}.jpg"
        label_filename = f"pure_bg_{i:06d}.txt"
        
        img_path = images_dir / img_filename
        label_path = labels_dir / label_filename
        
        # 保存图像
        cv2.imwrite(str(img_path), bg_img)
        # 创建空的标签文件（这是关键！）
        with open(label_path, 'w') as f:
            pass  # 创建一个空文件
        
        added_count += 1

    print(f"✅ 成功添加 {added_count} 张纯背景图像到数据集。")
    return added_count

def setup_yolo_dataset_structure(synthetic_path, output_yolo_path, train_ratio=0.7, val_ratio=0.2):
    """
    Set up YOLO format dataset structure with proper separation
    """
    os.makedirs(output_yolo_path, exist_ok=True)

    images_dir = os.path.join(synthetic_path, "images")
    labels_dir = os.path.join(synthetic_path, "labels")

    # Get all image files
    all_images = sorted([f for f in os.listdir(images_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    # Enhanced dataset splitting with background awareness
    # Group images by background patterns to prevent data leakage
    background_groups = {}
    for img_name in all_images:
        # Extract background identifier from filename or use hash-based grouping
        bg_group = hash(img_name) % 10  # Simple grouping for demonstration
        if bg_group not in background_groups:
            background_groups[bg_group] = []
        background_groups[bg_group].append(img_name)

    # Split groups to ensure background diversity across splits
    group_keys = list(background_groups.keys())
    random.shuffle(group_keys)
    
    train_cutoff = int(len(group_keys) * train_ratio)
    val_cutoff = train_cutoff + int(len(group_keys) * val_ratio)
    
    train_groups = group_keys[:train_cutoff]
    val_groups = group_keys[train_cutoff:val_cutoff]
    test_groups = group_keys[val_cutoff:]

    # Collect images for each split
    train_images = []
    val_images = []
    test_images = []
    
    for group in train_groups:
        train_images.extend(background_groups[group])
    for group in val_groups:
        val_images.extend(background_groups[group])
    for group in test_groups:
        test_images.extend(background_groups[group])

    print(f"Dataset split - Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")

    # Create YOLO dataset directory structure
    yolo_dirs = {
        'train': os.path.join(output_yolo_path, 'train'),
        'val': os.path.join(output_yolo_path, 'val'),
        'test': os.path.join(output_yolo_path, 'test')
    }

    for dir_type, dir_path in yolo_dirs.items():
        os.makedirs(os.path.join(dir_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dir_path, 'labels'), exist_ok=True)

    # Copy files to appropriate directories
    for img_name in train_images:
        src_img = os.path.join(images_dir, img_name)
        dst_img = os.path.join(yolo_dirs['train'], 'images', img_name)
        if not os.path.exists(dst_img):
            os.symlink(src_img, dst_img)

        label_name = os.path.splitext(img_name)[0] + '.txt'
        src_label = os.path.join(labels_dir, label_name)
        dst_label = os.path.join(yolo_dirs['train'], 'labels', label_name)
        if os.path.exists(src_label) and not os.path.exists(dst_label):
            os.symlink(src_label, dst_label)

    for img_name in val_images:
        src_img = os.path.join(images_dir, img_name)
        dst_img = os.path.join(yolo_dirs['val'], 'images', img_name)
        if not os.path.exists(dst_img):
            os.symlink(src_img, dst_img)

        label_name = os.path.splitext(img_name)[0] + '.txt'
        src_label = os.path.join(labels_dir, label_name)
        dst_label = os.path.join(yolo_dirs['val'], 'labels', label_name)
        if os.path.exists(src_label) and not os.path.exists(dst_label):
            os.symlink(src_label, dst_label)

    for img_name in test_images:
        src_img = os.path.join(images_dir, img_name)
        dst_img = os.path.join(yolo_dirs['test'], 'images', img_name)
        if not os.path.exists(dst_img):
            os.symlink(src_img, dst_img)

        label_name = os.path.splitext(img_name)[0] + '.txt'
        src_label = os.path.join(labels_dir, label_name)
        dst_label = os.path.join(yolo_dirs['test'], 'labels', label_name)
        if os.path.exists(src_label) and not os.path.exists(dst_label):
            os.symlink(src_label, dst_label)

    # Create dataset.yaml file
    classes_file = os.path.join(synthetic_path, "classes.txt")
    with open(classes_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    yaml_content = f"""
# YOLOv11 dataset configuration
path: {output_yolo_path}
train: train/images
val: val/images
test: test/images

# Number of classes
nc: {len(class_names)}

# Class names
names: {class_names}
"""

    yaml_path = os.path.join(output_yolo_path, "dataset.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"YOLO dataset structure created at {output_yolo_path}")
    print(f"Dataset includes {len(class_names)} classes: {class_names}")

    return yaml_path, class_names

def train_yolov11s(dataset_yaml, output_dir="./runs/train"):
    """
    Train YOLOv11s model with enhanced regularization
    """
    print("Loading YOLOv11s model...")
    model = YOLO('yolo11s.pt')

    # Enhanced training parameters with regularization
    training_params = {
        'data': dataset_yaml,
        'epochs': 100,
        'imgsz': 640,
        'batch': 8,
        'workers': 4,
        'device': 0,
        'optimizer': 'AdamW',
        
        # Enhanced learning rate with regularization
        'lr0': 0.0005,  # Reduced initial learning rate
        'lrf': 0.005,   # Reduced final learning rate
        'momentum': 0.9,  # Reduced momentum
        'weight_decay': 0.001,  # Increased weight decay
        
        # Enhanced regularization techniques
        'warmup_epochs': 5.0,  # Longer warmup
        'warmup_momentum': 0.8,
        'box': 5.0,  # Balanced loss weights
        'cls': 1.0,  # Increased classification weight
        'dfl': 1.5,
        
        # Advanced regularization
        'close_mosaic': 15,  # Later mosaic disabling
        'amp': True,
        'project': output_dir,
        'name': 'yolov11s_target_detection_regularized',
        'exist_ok': True,
        'patience': 30,  # More aggressive early stopping
        'save_period': 5,  # More frequent checkpointing
        
        # Additional regularization parameters
        'dropout': 0.1,  # Add dropout regularization
        'label_smoothing': 0.1,  # Label smoothing
        'cos_lr': True,  # Cosine learning rate scheduler
        'fliplr': 0.3,   # Reduced horizontal flip probability
        'flipud': 0.1,   # Reduced vertical flip probability
        
        'single_cls': False,
        'overlap_mask': True,
        'mask_ratio': 4,
        'val': True,
        'split': 'val',
        'save_json': False,
        'save_hybrid': False,
        'conf': 0.001,
        'iou': 0.5,  # Reduced IoU threshold for more strict matching
        'max_det': 300,
        'half': False,
        'plots': True
    }

    print("Starting YOLOv11s training with enhanced regularization...")
    results = model.train(**training_params)

    print("Evaluating model on validation set...")
    metrics = model.val()

    print(f"Training completed. Results saved to {output_dir}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")

    return model, results

# def optimize_model(model, dataset_yaml):
#     """
#     Model optimization with focus on generalization
#     """
#     print("Starting model optimization with generalization focus...")

#     # Conservative hyperparameter optimization
#     try:
#         result_grid = model.tune(
#             data=dataset_yaml,
#             use_ray=True,
#             grace_period=5,  # Shorter grace period
#             gpu_per_trial=1,
#             iterations=15,   # Reduced iterations
#             # Conservative search space
#             space={
#                 'lr0': (1e-5, 5e-2),  # Narrower range
#                 'lrf': (0.005, 0.5),  # Narrower range
#                 'momentum': (0.8, 0.95),
#                 'weight_decay': (0.0005, 0.005),  # Focus on regularization
#                 'warmup_epochs': (3.0, 8.0),
#                 'warmup_momentum': (0.7, 0.9),
#                 'box': (3.0, 8.0),  # Narrower range
#                 'cls': (0.5, 2.0),  # Balanced range
#             }
#         )
#         print("Conservative hyperparameter tuning completed")
#     except Exception as e:
#         print(f"Hyperparameter tuning failed: {e}")

#     # Export with optimization
#     try:
#         model.export(format="pb")
#         print("Model exported to TensorFlow format")
#     except Exception as e:
#         print(f"Model export failed: {e}")

#     return model
def optimize_model(model, dataset_yaml):
    """
    Model optimization with focus on generalization
    """
    print("Starting model optimization with generalization focus...")

    # 1. 尝试超参调优（确保已安装 ray[tune]）
    try:
        result_grid = model.tune(
            data=dataset_yaml,
            use_ray=True,
            grace_period=5,
            gpu_per_trial=1,
            iterations=15,
            space={
                'lr0': (1e-5, 5e-2),
                'lrf': (0.005, 0.5),
                'momentum': (0.8, 0.95),
                'weight_decay': (0.0005, 0.005),
                'warmup_epochs': (3.0, 8.0),
                'box': (3.0, 8.0),
                'cls': (0.5, 2.0),
            }
        )
        print("Conservative hyperparameter tuning completed")
    except Exception as e:
        print(f"Hyperparameter tuning skipped or failed: {e}")

    # 2. 【重要】我们不需要导出为其他格式，.pt 文件已在训练时生成
    # 因此，此处不执行任何 export 操作

    # 直接返回原始模型对象，它已经包含了训练好的权重和 save_dir 属性
    return model

def analyze_results(model, dataset_path):
    """
    Analyze training results with focus on generalization metrics
    """
    print("Analyzing training results and generalization patterns...")

    # Load best model
    best_model_path = os.path.join(model.save_dir, "weights", "best.pt")
    if os.path.exists(best_model_path):
        model = YOLO(best_model_path)

    # Comprehensive evaluation on all splits
    splits = ['train', 'val', 'test']
    metrics_summary = {}

    for split in splits:
        try:
            split_results = model.val(
                data=os.path.join(dataset_path, "dataset.yaml"),
                split=split,
                conf=0.25,
                iou=0.45
            )
            metrics_summary[split] = {
                'map50': split_results.box.map50,
                'map': split_results.box.map
            }
            print(f"{split.upper()} - mAP50: {split_results.box.map50:.4f}, mAP50-95: {split_results.box.map:.4f}")
        except Exception as e:
            print(f"Error evaluating on {split} split: {e}")

    # Calculate generalization gap
    if 'train' in metrics_summary and 'val' in metrics_summary:
        generalization_gap = metrics_summary['train']['map'] - metrics_summary['val']['map']
        print(f"Generalization gap (train_map - val_map): {generalization_gap:.4f}")

    # Enhanced error analysis
    error_analysis = {
        "generalization_gap": generalization_gap if 'generalization_gap' in locals() else None,
        "overfitting_indicator": generalization_gap > 0.1 if 'generalization_gap' in locals() else True,
        "train_val_consistency": abs(metrics_summary.get('train', {}).get('map50', 0) - 
                                   metrics_summary.get('val', {}).get('map50', 0)) < 0.15
    }

    print("Enhanced error analysis completed")
    return error_analysis

def main():
    """
    Main function: Execute complete training pipeline with regularization
    """
    print("Starting YOLOv11s target detection training pipeline with enhanced regularization")
    start_time = datetime.now()

    # Set paths
    base_path = "/home/legion/dataset"
    raw_dataset_path = os.path.join(base_path, "raw_dataset")
    background_path = os.path.join(base_path, "trash", "empty")
    augmented_path = os.path.join(base_path, "augmented_dataset")
    synthetic_path = os.path.join(base_path, "synthetic_dataset")
    yolo_dataset_path = os.path.join(base_path, "yolo_dataset")

    # 1. Set up environment and GPU warm-up
    device = setup_environment()

    # 2. Load background images
    background_images = load_background_images(background_path)

    # 3. Enhanced data augmentation with regularization
    augment_dataset(raw_dataset_path, augmented_path, target_per_class=900)

    # 4. Create synthetic dataset with enhanced randomization
    classes, num_images = create_synthetic_dataset_parallel(
        augmented_path, 
        background_images, 
        synthetic_path,
        target_scale=4.0,  # Reduced scaling for more realistic sizes
        batch_size=50,
        max_workers=8
    )

    # 5. Set up YOLO dataset structure with proper separation
    dataset_yaml, class_names = setup_yolo_dataset_structure(synthetic_path, yolo_dataset_path)

    # 6. Train YOLOv11s model with enhanced regularization
    model, results = train_yolov11s(dataset_yaml)

    # 7. Conservative model optimization
    optimized_model = optimize_model(model, dataset_yaml)

    # 8. Comprehensive result analysis
    error_analysis = analyze_results(optimized_model, yolo_dataset_path)

    # Output training summary
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "="*50)
    print("REGULARIZED TRAINING SUMMARY")
    print("="*50)
    print(f"Start time: {start_time}")
    print(f"End time: {end_time}")
    print(f"Total duration: {duration}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Total synthetic images: {num_images}")
    print(f"Final model saved at: {model.save_dir}")
    print(f"Generalization analysis: {error_analysis}")
    print("Regularized training pipeline completed successfully!")

if __name__ == "__main__":
    main()