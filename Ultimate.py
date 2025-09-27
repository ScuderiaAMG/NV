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

# 忽略TensorRT警告
warnings.filterwarnings("ignore", message="TF-TRT Warning: Could not find TensorRT")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少TensorFlow日志输出

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
        # Perform simple matrix operations to warm up GPU
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
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Use Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,  # Minimum distance between circles
        param1=50,   # Upper threshold for edge detection
        param2=30,   # Threshold for center detection
        minRadius=10,
        maxRadius=min(image.shape[0], image.shape[1]) // 4
    )
    
    if circles is not None:
        # Convert coordinates and radius to integers
        circles = np.round(circles[0, :]).astype("int")
        
        # Find the circle with highest confidence (largest radius typically)
        best_circle = max(circles, key=lambda x: x[2])
        x, y, r = best_circle
        
        # Calculate confidence score
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
        # 使用较少的进程以避免内存问题
        num_processes = min(8, len(background_images), cpu_count())
    
    print(f"Using {num_processes} processes for Hough circle detection")
    
    bg_target_cache = {}
    
    # Prepare arguments for parallel processing
    args = [(bg_img, i) for i, bg_img in enumerate(background_images)]
    
    # Use multiprocessing Pool
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(detect_target_center_hough_parallel_wrapper, args), 
                           total=len(args), desc="Detecting circles in backgrounds"))
    
    # Process results
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
    Calculate circle confidence score (fixed numerical overflow issue)
    """
    height, width = gray_img.shape
    
    # Ensure circle is within image bounds (using signed integer calculations)
    x_min = max(0, int(x) - int(radius))
    x_max = min(width, int(x) + int(radius))
    y_min = max(0, int(y) - int(radius))
    y_max = min(height, int(y) + int(radius))
    
    if x_min >= x_max or y_min >= y_max:
        return 0
    
    # Extract ROI region
    roi = gray_img[y_min:y_max, x_min:x_max]
    
    if roi.size == 0:
        return 0
    
    # Calculate edge strength and contrast as confidence
    edges = cv2.Canny(roi, 50, 150)
    edge_strength = np.mean(edges) if edges.size > 0 else 0
    contrast = np.std(roi) if roi.size > 0 else 0
    
    confidence = 0.6 * edge_strength + 0.4 * contrast
    return confidence

def augment_dataset(raw_dataset_path, output_path, target_per_class=900):
    """
    Augment dataset: Apply various data augmentation techniques
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Albumentations augmentation pipeline - 移除有问题的参数
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=45, p=0.8),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.Blur(blur_limit=3, p=0.3),
        A.CLAHE(p=0.3),
        A.RandomGamma(p=0.3),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),  # 移除了alpha_affine参数
        ToTensorV2()
    ])
    
    class_folders = sorted([f for f in os.listdir(raw_dataset_path) 
                          if os.path.isdir(os.path.join(raw_dataset_path, f))])
    
    print(f"Found {len(class_folders)} classes to augment")
    
    for class_folder in class_folders:
        class_path = os.path.join(raw_dataset_path, class_folder)
        output_class_path = os.path.join(output_path, class_folder)
        os.makedirs(output_class_path, exist_ok=True)
        
        # Get class images
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
        
        # Calculate number of images to generate
        existing_count = len(class_images)
        needed_count = target_per_class - existing_count
        
        if needed_count <= 0:
            print(f"Class {class_folder} already has {existing_count} images, skipping augmentation")
            continue
        
        # Apply augmentations
        augmented_count = 0
        while augmented_count < needed_count:
            for base_img in class_images:
                if augmented_count >= needed_count:
                    break
                
                try:
                    # Apply various augmentation techniques
                    augmented = transform(image=base_img)
                    augmented_img = augmented['image']
                    
                    if isinstance(augmented_img, torch.Tensor):
                        augmented_img = augmented_img.permute(1, 2, 0).numpy()
                        augmented_img = (augmented_img * 255).astype(np.uint8)
                    
                    # Save augmented image
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
    Process single image (thread-safe)
    """
    # Randomly select background image
    bg_idx = random.randint(0, len(background_images) - 1)
    bg_img = background_images[bg_idx].copy()
    bg_height, bg_width = bg_img.shape[:2]
    
    # Get target center from cache
    target_center, confidence = bg_target_cache[bg_idx]
    
    # Load augmented image
    aug_img_path = os.path.join(class_path, img_filename)
    aug_img = cv2.imread(aug_img_path, cv2.IMREAD_UNCHANGED)
    
    if aug_img is None:
        return False
    
    if target_center is None:
        # If no target detected, use image center
        target_x, target_y = bg_width // 2, bg_height // 2
        target_radius = min(bg_width, bg_height) // 10  # Default radius
    else:
        target_x, target_y, target_radius = target_center
    
    # Resize augmented image: allow scaling based on target_scale parameter
    aug_height, aug_width = aug_img.shape[:2]
    # Calculate scaling factor: scale based on target_scale, but not exceeding 1/3 of background
    scale_factor = target_scale  # Direct scaling factor
    new_width = int(aug_width * scale_factor)
    new_height = int(aug_height * scale_factor)
    
    # Ensure new dimensions don't exceed 1/3 of background
    max_allowed_size = min(bg_width, bg_height) // 3
    if new_width > max_allowed_size or new_height > max_allowed_size:
        # If too large, scale to maximum allowed size
        scale_factor = max_allowed_size / max(new_width, new_height)
        new_width = int(aug_width * scale_factor)
        new_height = int(aug_height * scale_factor)
    
    if new_width <= 0 or new_height <= 0:
        return False
        
    aug_img_resized = cv2.resize(aug_img, (new_width, new_height))
    
    # Calculate placement position (centered on target)
    start_x = max(0, target_x - new_width // 2)
    start_y = max(0, target_y - new_height // 2)
    end_x = min(bg_width, start_x + new_width)
    end_y = min(bg_height, start_y + new_height)
    
    # Adjust image size to fit boundaries
    if end_x - start_x != new_width or end_y - start_y != new_height:
        aug_img_resized = cv2.resize(aug_img_resized, 
                                   (end_x - start_x, end_y - start_y))
    
    # Alpha blending (if augmented image has alpha channel)
    if aug_img_resized.shape[2] == 4:
        alpha = aug_img_resized[:, :, 3] / 255.0
        alpha = alpha[:, :, np.newaxis]  # Add channel dimension for broadcasting
        
        # Ensure the background region has the same dimensions as the augmented image
        bg_region = bg_img[start_y:end_y, start_x:end_x]
        
        # Blend using alpha channel
        for c in range(3):
            bg_region[:, :, c] = (1 - alpha) * bg_region[:, :, c] + alpha * aug_img_resized[:, :, c]
        
        # Update the background image with the blended region
        bg_img[start_y:end_y, start_x:end_x] = bg_region
    else:
        # Direct overlay
        bg_img[start_y:end_y, start_x:end_x] = aug_img_resized
    
    # Save synthetic image
    image_filename = f"synthetic_{img_counter:06d}.jpg"
    image_path = os.path.join(output_path, "images", image_filename)
    cv2.imwrite(image_path, bg_img)
    
    # Create YOLO format annotation file
    label_filename = f"synthetic_{img_counter:06d}.txt"
    label_path = os.path.join(output_path, "labels", label_filename)
    
    # Calculate bounding box (normalized coordinates)
    x_center = (start_x + (end_x - start_x) / 2) / bg_width
    y_center = (start_y + (end_y - start_y) / 2) / bg_height
    width = (end_x - start_x) / bg_width
    height = (end_y - start_y) / bg_height
    
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
    Create synthetic dataset: Fuse augmented images onto backgrounds with parallel processing
    """
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "labels"), exist_ok=True)
    
    # Get all classes
    classes = sorted([f for f in os.listdir(augmented_path) 
                     if os.path.isdir(os.path.join(augmented_path, f))])
    
    print(f"Creating synthetic dataset with {len(classes)} classes")
    
    # Precompute target centers for background images using parallel processing
    print("Precomputing target centers for background images using parallel processing...")
    start_time = time.time()
    
    # Use all available CPU cores for Hough circle detection
    # 减少进程数以避免内存问题
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
    
    # Create images and annotations for each class
    image_counter = 0
    
    # Process each class
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
        
        # Prepare arguments for parallel processing
        args_list = []
        for img_filename in class_images:
            args = (
                class_path, img_filename, class_idx, class_name, 
                background_images, bg_target_cache, output_path, 
                image_counter, target_scale
            )
            args_list.append(args)
            image_counter += 1
        
        # Use multiprocessing Pool for image processing
        print(f"Processing {len(class_images)} images for class {class_name} using parallel processing")
        start_time = time.time()
        
        # Determine number of processes to use
        if max_workers is None:
            max_workers = min(8, cpu_count(), len(class_images))  # 限制最大进程数
        
        # 分批处理以避免内存问题
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
    
    # Create classes.txt
    with open(os.path.join(output_path, "classes.txt"), 'w') as f:
        for class_name in classes:
            f.write(f"{class_name}\n")
    
    return classes, image_counter

def setup_yolo_dataset_structure(synthetic_path, output_yolo_path, train_ratio=0.8):
    """
    Set up YOLO format dataset structure
    """
    os.makedirs(output_yolo_path, exist_ok=True)
    
    images_dir = os.path.join(synthetic_path, "images")
    labels_dir = os.path.join(synthetic_path, "labels")
    
    # Get all image files
    all_images = sorted([f for f in os.listdir(images_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # Split into training and validation sets
    train_images, val_images = train_test_split(
        all_images, train_size=train_ratio, random_state=42
    )
    
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
        # Copy image
        src_img = os.path.join(images_dir, img_name)
        dst_img = os.path.join(yolo_dirs['train'], 'images', img_name)
        os.symlink(src_img, dst_img)  # Use symbolic links to save space
        
        # Copy annotation
        label_name = os.path.splitext(img_name)[0] + '.txt'
        src_label = os.path.join(labels_dir, label_name)
        dst_label = os.path.join(yolo_dirs['train'], 'labels', label_name)
        if os.path.exists(src_label):
            os.symlink(src_label, dst_label)
    
    for img_name in val_images:
        src_img = os.path.join(images_dir, img_name)
        dst_img = os.path.join(yolo_dirs['val'], 'images', img_name)
        os.symlink(src_img, dst_img)
        
        label_name = os.path.splitext(img_name)[0] + '.txt'
        src_label = os.path.join(labels_dir, label_name)
        dst_label = os.path.join(yolo_dirs['val'], 'labels', label_name)
        if os.path.exists(src_label):
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
    Train YOLOv11s model
    """
    # Load pretrained model
    print("Loading YOLOv11s model...")
    model = YOLO('yolo11s.pt')  # Use pretrained YOLOv11s model
    
    # Training parameters
    training_params = {
        'data': dataset_yaml,
        'epochs': 100,
        'imgsz': 640,
        'batch': 8,  # Adjust based on GPU memory
        'workers': 4,
        'device': 0,  # Use GPU 0
        'optimizer': 'AdamW',  # Use AdamW optimizer
        'lr0': 0.001,  # Initial learning rate
        'lrf': 0.01,  # Final learning rate
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'box': 7.5,  # Bounding box loss weight
        'cls': 0.5,  # Classification loss weight
        'dfl': 1.5,  # Distribution Focal loss weight
        'close_mosaic': 10,
        'amp': True,  # Automatic Mixed Precision training
        'project': output_dir,
        'name': 'yolov11s_target_detection',
        'exist_ok': True,
        'patience': 50,  # Early stopping patience
        'save_period': 10,  # Save checkpoint every 10 epochs
        'single_cls': False,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,  # Enable validation
        'split': 'val',
        'save_json': False,
        'save_hybrid': False,
        'conf': 0.001,  # Validation confidence threshold
        'iou': 0.6,  # NMS IoU threshold
        'max_det': 300,  # Maximum detections per image
        'half': False,  # Use half-precision inference
        'plots': True  # Save training plots
    }
    
    # Start training
    print("Starting YOLOv11s training...")
    results = model.train(**training_params)
    
    # Evaluate model on validation set
    print("Evaluating model on validation set...")
    metrics = model.val()
    
    print(f"Training completed. Results saved to {output_dir}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    
    return model, results

def optimize_model(model, dataset_yaml):
    """
    Model optimization: Hyperparameter tuning, post-processing optimization, etc.
    """
    print("Starting model optimization...")
    
    # Hyperparameter optimization (using Ray Tune)
    try:
        result_grid = model.tune(
            data=dataset_yaml,
            use_ray=True,
            grace_period=10,
            gpu_per_trial=1,
            iterations=20,
            # Custom search space
            space={
                'lr0': (1e-5, 1e-1),
                'lrf': (0.01, 1.0),
                'momentum': (0.6, 0.98),
                'weight_decay': (0.0, 0.001),
                'warmup_epochs': (0.0, 5.0),
                'warmup_momentum': (0.0, 0.95),
                'box': (0.02, 0.2),
                'cls': (0.2, 4.0),
            }
        )
        print("Hyperparameter tuning completed")
    except Exception as e:
        print(f"Hyperparameter tuning failed: {e}")
    
    # Export to TensorFlow format for faster inference
    try:
        model.export(format="pb")  # Export as TF GraphDef
        print("Model exported to TensorFlow format")
    except Exception as e:
        print(f"Model export failed: {e}")
    
    return model

def analyze_results(model, dataset_path):
    """
    Analyze training results and error patterns
    """
    print("Analyzing training results and error patterns...")
    
    # Load best model
    best_model_path = os.path.join(model.save_dir, "weights", "best.pt")
    if os.path.exists(best_model_path):
        model = YOLO(best_model_path)
    
    # Evaluate on test set
    test_results = model.val(
        data=os.path.join(dataset_path, "dataset.yaml"),
        split="test",
        conf=0.25,
        iou=0.45
    )
    
    # Analyze error patterns
    error_analysis = {
        "false_positives": 0,
        "false_negatives": 0,
        "localization_errors": 0,
        "classification_errors": 0
    }
    
    # Add more detailed error analysis logic here
    # For example, use model predictions and analyze confusion matrix
    
    print("Error analysis completed")
    print(f"Test results - mAP50-95: {test_results.box.map:.4f}")
    
    return error_analysis

def main():
    """
    Main function: Execute complete training pipeline
    """
    print("Starting YOLOv11s target detection training pipeline")
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
    
    # 3. Detect bullseye in background (optional, for verification)
    if background_images:
        sample_bg = background_images[0]
        target_center, confidence = detect_target_center_hough(sample_bg)
        if target_center:
            x, y, r = target_center
            print(f"Sample target detection - Center: ({x}, {y}), Radius: {r}, Confidence: {confidence:.2f}")
        else:
            print("No target detected in sample background image")
    
    # 4. Data augmentation
    augment_dataset(raw_dataset_path, augmented_path, target_per_class=900)
    
    # 5. Create synthetic dataset (using parallel processing)
    classes, num_images = create_synthetic_dataset_parallel(
        augmented_path, 
        background_images, 
        synthetic_path,
        target_scale=5.0,  # Scale up by 5x
        batch_size=50,     # Process 50 images per batch
        max_workers=8      # Use 8 processes for CPU-intensive tasks (减少进程数)
    )
    
    # 6. Set up YOLO dataset structure
    dataset_yaml, class_names = setup_yolo_dataset_structure(synthetic_path, yolo_dataset_path)
    
    # 7. Train YOLOv11s model
    model, results = train_yolov11s(dataset_yaml)
    
    # 8. Model optimization
    optimized_model = optimize_model(model, dataset_yaml)
    
    # 9. Result analysis
    error_analysis = analyze_results(optimized_model, yolo_dataset_path)
    
    # Output training summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Start time: {start_time}")
    print(f"End time: {end_time}")
    print(f"Total duration: {duration}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Total synthetic images: {num_images}")
    print(f"Final model saved at: {model.save_dir}")
    print(f"Error analysis: {error_analysis}")
    print("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()