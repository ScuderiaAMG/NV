import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import time
import shutil
import albumentations as A
from albumentations.pytorch import ToTensorV2
import xml.etree.ElementTree as ET
from tqdm import tqdm
from ultralytics import YOLO
import pynvml
import psutil
import concurrent.futures
import math
from ultralytics import __version__ as ultralytics_version
from packaging import version
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random
import yaml
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
from collections import defaultdict
import gc

os.environ['ULTRALYTICS_CHECK'] = 'False'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

TRAIN_HOURS = 96
CHECKPOINT_INTERVAL = 4 * 3600
BATCH_SIZE = 4
IMG_SIZE = 512
NUM_WORKERS = 4
PIN_MEMORY = True
LEARNING_RATE = 0.001
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0001
WARMUP_EPOCHS = 5
BOX_GAIN = 0.05
CLS_GAIN = 0.5
OBJ_GAIN = 1.0
NUM_BACKGROUND_IMAGES = 8
BACKGROUND_SCENE_DIR = "/home/legion/dataset/background_scene"
MAX_VRAM_USAGE = 6.5

MIN_IMAGES_PER_CLASS = 800
MAX_IMAGES_PER_CLASS = 1000

test_mode = False #True
max_test_images = 5

VAL_SPLIT_RATIO = 0.2

def get_hardware_config():
    config = {
        "GPU_NAME": "Unknown",
        "GPU_MEMORY": 0,
        "CPU_CORES": os.cpu_count() or 4
    }
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        config["GPU_NAME"] = pynvml.nvmlDeviceGetName(handle)
        config["GPU_MEMORY"] = pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**3)
    except:
        pass
    print(f"Hardware configuration:")
    print(f"GPU: {config['GPU_NAME']} | VRAM: {config['GPU_MEMORY']:.1f}GB")
    print(f"CPU: {config['CPU_CORES']} cores")
    if config['GPU_MEMORY'] < MAX_VRAM_USAGE + 1:
        print(f"Warning: VRAM usage ({config['GPU_MEMORY']:.1f}GB) close to limit ({MAX_VRAM_USAGE}GB)")
    return config

config = get_hardware_config()

RAW_DATA_DIR = "/home/legion/dataset/raw_dataset"
AUG_DATA_DIR = "/home/legion/dataset/augmented_dataset"
YOLO_DATA_DIR = "/home/legion/dataset/yolo_dataset"
TRAIN_DIR = os.path.join(YOLO_DATA_DIR, "train")
VAL_DIR = os.path.join(YOLO_DATA_DIR, "val")
CHECKPOINT_DIR = "/home/legion/dataset"
MODEL_ANALYSIS_DIR = "/home/legion/dataset/model_analysis"

PRETRAINED_WEIGHTS = "/home/legion/dataset/yolov8x.pt"

for path in [AUG_DATA_DIR, TRAIN_DIR, VAL_DIR, CHECKPOINT_DIR, BACKGROUND_SCENE_DIR, MODEL_ANALYSIS_DIR]:
    os.makedirs(path, exist_ok=True)

print(f"Directory structure:")
print(f"- Raw data: {os.path.abspath(RAW_DATA_DIR)}")
print(f"- Backgrounds: {os.path.abspath(BACKGROUND_SCENE_DIR)}")
print(f"- Augmented data: {os.path.abspath(AUG_DATA_DIR)}")
print(f"- YOLO dataset: {os.path.abspath(YOLO_DATA_DIR)}")
print(f"- Checkpoints: {os.path.abspath(CHECKPOINT_DIR)}")
print(f"- Model analysis: {os.path.abspath(MODEL_ANALYSIS_DIR)}")
print(f"- Pretrained weights: {os.path.abspath(PRETRAINED_WEIGHTS)}")

def analyze_background():
    print("Analyzing background scenes and detecting ROIs...")
    background_images = []
    valid_files = [f for f in os.listdir(BACKGROUND_SCENE_DIR) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not valid_files:
        print("Error: No background images found")
        exit()
    
    for img_file in tqdm(valid_files[:NUM_BACKGROUND_IMAGES], desc="Loading backgrounds"):
        img_path = os.path.join(BACKGROUND_SCENE_DIR, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to read background image: {img_path}")
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        background_images.append(img)
    
    if not background_images:
        print("Error: No valid background images")
        exit()
    
    median_background = np.median(background_images, axis=0).astype(np.uint8)
    cv2.imwrite("median_background.jpg", median_background)
    
    diff_img = np.zeros_like(median_background, dtype=np.float32)
    for img in background_images:
        diff = cv2.absdiff(img, median_background).astype(np.float32)
        diff_img = np.maximum(diff_img, diff)
    
    diff_img = np.clip(diff_img, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY)
    
    thresh = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.dilate(thresh, kernel, iterations=3)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = []
    min_area = IMG_SIZE * IMG_SIZE * 0.01
    
    for c in contours:
        area = cv2.contourArea(c)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(c)
            x = max(0, min(x, IMG_SIZE - 10))
            y = max(0, min(y, IMG_SIZE - 10))
            w = max(10, min(w, IMG_SIZE - x))
            h = max(10, min(h, IMG_SIZE - y))
            rois.append((x, y, x+w, y+h))
    
    if not rois:
        rois.append((0, 0, IMG_SIZE, IMG_SIZE))
    
    if np.std(median_background) < 10:
        print("CRITICAL ERROR: Background images are too uniform! Cannot detect ROIs.")
        exit(1)
    
    if len(rois) < 3:
        print("WARNING: Very few ROIs detected. Check background scene quality.")
    
    roi_vis = median_background.copy()
    for (x1, y1, x2, y2) in rois:
        cv2.rectangle(roi_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite("background_rois.jpg", roi_vis)
    
    print(f"Detected {len(rois)} regions of interest")
    return background_images, rois

background_images, rois = analyze_background()

def visualize_roi_and_background():
    print("Visualizing ROI and background...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    median_img = cv2.imread("median_background.jpg")
    if median_img is not None:
        median_img = cv2.cvtColor(median_img, cv2.COLOR_BGR2RGB)
        axes[0].imshow(median_img)
        axes[0].set_title("Median Background")
        axes[0].axis('off')
    
    roi_img = cv2.imread("background_rois.jpg")
    if roi_img is not None:
        roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
        axes[1].imshow(roi_img)
        axes[1].set_title(f"Detected ROIs: {len(rois)}")
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig("background_analysis_visualization.jpg")
    plt.close()
    print("Background visualization saved")

visualize_roi_and_background()

def remove_white_background(target, img_path=""):
    if target.shape[2] == 3:
        target = cv2.cvtColor(target, cv2.COLOR_BGR2BGRA)
    
    lab = cv2.cvtColor(target[:, :, :3], cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    white_mask = cv2.inRange(
        lab, 
        np.array([200, 120, 120]),
        np.array([255, 135, 135])
    )
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    
    alpha = np.where(white_mask > 0, 0, 255).astype(np.uint8)
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    target[:, :, 3] = alpha
    
    return target

def place_target_in_roi(background, target, roi, img_path=""):
    target = remove_white_background(target, img_path)
    
    if len(target.shape) == 2 or target.shape[2] == 1:
        target = cv2.cvtColor(target, cv2.COLOR_GRAY2BGRA)
    elif target.shape[2] == 3:
        target = remove_white_background(target, img_path)
    
    alpha_channel = target[:, :, 3]
    if np.mean(alpha_channel) > 250:
        h, w = alpha_channel.shape[:2]
        center_x, center_y = w // 2, h // 2
        max_dist = min(w, h) // 2
        mask = np.zeros((h, w), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                alpha_val = max(0, min(255, 255 * (1 - dist / max_dist)))
                mask[y, x] = alpha_val
        target[:, :, 3] = mask
    
    if np.count_nonzero(alpha_channel < 10) < alpha_channel.size * 0.1:
        edges = cv2.Canny(alpha_channel, 100, 200)
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        alpha_channel[edges > 0] = 0
        target[:, :, 3] = alpha_channel
    
    x1, y1, x2, y2 = roi
    roi_width = x2 - x1
    roi_height = y2 - y1
    
    scale_factor = min(
        roi_width / target.shape[1], 
        roi_height / target.shape[0]
    ) * np.random.uniform(0.5, 0.8)
    
    new_width = max(10, int(target.shape[1] * scale_factor))
    new_height = max(10, int(target.shape[0] * scale_factor))
    resized_target = cv2.resize(target, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    max_x = max(1, roi_width - new_width)
    max_y = max(1, roi_height - new_height)
    pos_x = x1 + np.random.randint(0, max_x)
    pos_y = y1 + np.random.randint(0, max_y)
    
    alpha = resized_target[:, :, 3] / 255.0
    alpha = np.expand_dims(alpha, axis=2)
    alpha = np.repeat(alpha, 3, axis=2)
    
    bg_region = background[pos_y:pos_y+new_height, pos_x:pos_x+new_width]
    
    if bg_region.shape[:2] != resized_target.shape[:2]:
        new_height, new_width = bg_region.shape[:2]
        resized_target = cv2.resize(resized_target, (new_width, new_height))
        alpha = cv2.resize(alpha, (new_width, new_height))
    
    foreground = resized_target[:, :, :3].astype(float)
    blended = (foreground * alpha + bg_region.astype(float) * (1 - alpha)).clip(0, 255).astype(np.uint8)
    
    background[pos_y:pos_y+new_height, pos_x:pos_x+new_width] = blended
    
    xmin = max(0, pos_x)
    ymin = max(0, pos_y)
    xmax = min(background.shape[1], pos_x + new_width)
    ymax = min(background.shape[0], pos_y + new_height)
    
    return background, (xmin, ymin, xmax, ymax)

def visualize_target_placement():
    print("Visualizing target placement...")
    if not background_images:
        print("Error: No background images available")
        return
    
    bg_img = background_images[0].copy()
    class_names = [d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))]
    
    if not class_names:
        print("Error: No class directories found")
        return
    
    class_name = random.choice(class_names)
    class_dir = os.path.join(RAW_DATA_DIR, class_name)
    img_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not img_files:
        print(f"Warning: No images in class {class_name}")
        return
    
    img_file = random.choice(img_files)
    img_path = os.path.join(class_dir, img_file)
    target_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    
    if target_img is None:
        print(f"Error: Failed to read image {img_path}")
        return
    
    if not rois:
        print("Warning: No ROIs available")
        return
    
    roi = rois[random.randint(0, len(rois)-1)]
    composite_img, bbox = place_target_in_roi(bg_img, target_img, roi, img_path)
    
    if composite_img is None:
        print("Error: Target placement failed")
        return
    
    xmin, ymin, xmax, ymax = bbox
    cv2.rectangle(composite_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
    cv2.imwrite("target_placement_visualization.jpg", composite_img)
    print("Target placement visualization saved")

visualize_target_placement()

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2, brightness_limit=0.1, contrast_limit=0.1),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.1))

def process_image(class_name, img_file, aug_idx):
    try:
        class_dir = os.path.join(RAW_DATA_DIR, class_name)
        aug_class_dir = os.path.join(AUG_DATA_DIR, class_name)
        os.makedirs(aug_class_dir, exist_ok=True)
        
        img_path = os.path.join(class_dir, img_file)
        target_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
        if target_img is None:
            print(f"Error: Failed to read image {img_path}")
            return
        
        if not background_images or not rois:
            print("Error: Missing background or ROI data")
            return
        
        bg_idx = np.random.randint(0, len(background_images))
        bg_img = background_images[bg_idx].copy()
        roi_idx = np.random.randint(0, len(rois))
        roi = rois[roi_idx]
        
        composite_img, bbox = place_target_in_roi(bg_img, target_img, roi, img_path)
        
        if composite_img is None:
            print(f"Error: Target placement failed {img_path}")
            return
        
        xmin, ymin, xmax, ymax = bbox
        if xmax - xmin < 5 or ymax - ymin < 5:
            print(f"Warning: Invalid bbox size {bbox} for {img_path}")
            return
        
        try:
            transformed = transform(
                image=composite_img,
                bboxes=[bbox],
                class_labels=[class_name]
            )
        except Exception as e:
            print(f"Transform error: {e} for {img_path}")
            return
        
        aug_img = transformed['image']
        aug_bboxes = transformed['bboxes']
        
        if not aug_bboxes:
            print(f"Warning: No bbox after transform {img_path}")
            return
        
        xmin, ymin, xmax, ymax = aug_bboxes[0]
        xmin = max(0, min(xmin, IMG_SIZE-1))
        ymin = max(0, min(ymin, IMG_SIZE-1))
        xmax = max(1, min(xmax, IMG_SIZE))
        ymax = max(1, min(ymax, IMG_SIZE))
        
        if xmax <= xmin or ymax <= ymin:
            print(f"Warning: Invalid transformed bbox {xmin},{ymin},{xmax},{ymax} for {img_path}")
            return
        
        aug_img_path = os.path.join(aug_class_dir, f"{os.path.splitext(img_file)[0]}_aug_{aug_idx}.jpg")
        aug_xml_path = os.path.join(aug_class_dir, f"{os.path.splitext(img_file)[0]}_aug_{aug_idx}.xml")
        
        try:
            aug_img_np = aug_img.permute(1, 2, 0).numpy()
            if aug_img_np.max() <= 1.0:
                aug_img_np = (aug_img_np * 255).astype(np.uint8)
            else:
                aug_img_np = aug_img_np.astype(np.uint8)
            
            if np.mean(aug_img_np) > 240:
                print(f"CRITICAL: Generated image is almost white! Mean value: {np.mean(aug_img_np):.1f}")
                return
            
            cv2.imwrite(aug_img_path, aug_img_np)
        except Exception as e:
            print(f"Image save error: {e} for {aug_img_path}")
            return
        
        try:
            aug_root = ET.Element("annotation")
            ET.SubElement(aug_root, "folder").text = class_name
            ET.SubElement(aug_root, "filename").text = os.path.basename(aug_img_path)
            size = ET.SubElement(aug_root, "size")
            ET.SubElement(size, "width").text = str(IMG_SIZE)
            ET.SubElement(size, "height").text = str(IMG_SIZE)
            ET.SubElement(size, "depth").text = "3"
            obj = ET.SubElement(aug_root, "object")
            ET.SubElement(obj, "name").text = class_name
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(int(xmin))
            ET.SubElement(bndbox, "ymin").text = str(int(ymin))
            ET.SubElement(bndbox, "xmax").text = str(int(xmax))
            ET.SubElement(bndbox, "ymax").text = str(int(ymax))
            tree = ET.ElementTree(aug_root)
            tree.write(aug_xml_path)
        except Exception as e:
            print(f"XML save error: {e} for {aug_xml_path}")
    except Exception as e:
        print(f"CRITICAL ERROR in {img_file}: {str(e)}")

print("Starting target placement and labeling...")

class_stats = {}
for class_name in os.listdir(RAW_DATA_DIR):
    class_dir = os.path.join(RAW_DATA_DIR, class_name)
    if os.path.isdir(class_dir):
        img_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if img_files:
            class_stats[class_name] = len(img_files)

print("Class statistics:")
for class_name, count in class_stats.items():
    print(f"- {class_name}: {count} images")

aug_per_class = {}
for class_name, count in class_stats.items():
    if test_mode:
        target_count = min(max_test_images, MIN_IMAGES_PER_CLASS)
    else:
        if class_name in ['chimpanzee', 'seal', 'shrew', 'otter']:
            target_count = random.randint(1000, 1200)
        else:
            target_count = random.randint(MIN_IMAGES_PER_CLASS, MAX_IMAGES_PER_CLASS)
    if count > 0:
        aug_per_image = max(1, int(target_count / count))
        aug_per_class[class_name] = aug_per_image
        print(f"{class_name}: {count} images -> generating {count * aug_per_image} augmented images (target: {target_count})")

with concurrent.futures.ThreadPoolExecutor(max_workers=min(config["CPU_CORES"], 8)) as executor:
    futures = []
    total_tasks = 0
    
    for class_name, count in class_stats.items():
        if count == 0:
            continue
            
        class_dir = os.path.join(RAW_DATA_DIR, class_name)
        img_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        aug_count = aug_per_class.get(class_name, 1)
        
        for img_file in img_files:
            for aug_idx in range(aug_count):
                futures.append(executor.submit(process_image, class_name, img_file, aug_idx))
                total_tasks += 1
    
    progress_bar = tqdm(total=total_tasks, desc="Generating augmented data")
    for future in concurrent.futures.as_completed(futures):
        progress_bar.update(1)
    progress_bar.close()

print(f"Target placement completed. Generated {total_tasks} augmented images.")

print("Converting to YOLO format and splitting dataset...")

def validate_labels(labels_dir, num_classes):
    print(f"Validating labels in {labels_dir}...")
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    
    for label_file in tqdm(label_files[:1000], desc="Checking labels"):
        with open(os.path.join(labels_dir, label_file), 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) != 5:
                    continue
                
                class_id = int(parts[0])
                if class_id >= num_classes or class_id < 0:
                    print(f"ERROR: Invalid class ID {class_id} in {label_file}")
                    return False
                elif class_id >= num_classes:
                    print(f"ERROR: Class ID {class_id} exceeds max class {num_classes-1} in {label_file}")
                    return False
    
    print("Label validation passed!")
    return True

def convert_and_split_dataset(data_dir, train_dir, val_dir, val_ratio=0.2):
    train_images_dir = os.path.join(train_dir, "images")
    train_labels_dir = os.path.join(train_dir, "labels")
    val_images_dir = os.path.join(val_dir, "images")
    val_labels_dir = os.path.join(val_dir, "labels")
    
    for path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        os.makedirs(path, exist_ok=True)
    
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    
    all_images = []
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for file in os.listdir(class_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, file)
                xml_path = os.path.splitext(img_path)[0] + ".xml"
                if os.path.exists(xml_path):
                    all_images.append((img_path, xml_path, class_name))
    
    train_images, val_images = train_test_split(
        all_images, test_size=val_ratio, random_state=42, stratify=[img[2] for img in all_images]
    )
    
    print(f"Dataset split: {len(train_images)} train, {len(val_images)} validation")
    
    for img_path, xml_path, class_name in tqdm(train_images, desc="Processing train set"):
        img_file = os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(train_images_dir, img_file))
        create_yolo_label(xml_path, train_labels_dir, class_to_id, img_file)
    
    for img_path, xml_path, class_name in tqdm(val_images, desc="Processing validation set"):
        img_file = os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(val_images_dir, img_file))
        create_yolo_label(xml_path, val_labels_dir, class_to_id, img_file)
    
    num_classes = len(class_names)
    if not validate_labels(train_labels_dir, num_classes) or not validate_labels(val_labels_dir, num_classes):
        print("Critical label errors found! Stopping.")
        exit(1)
    
    dataset_yaml = {
        'train': os.path.abspath(os.path.join(train_dir, "images")),
        'val': os.path.abspath(os.path.join(val_dir, "images")),
        'nc': num_classes,
        'names': class_names
    }
    
    dataset_yaml_path = os.path.join(YOLO_DATA_DIR, "dataset.yaml")
    with open(dataset_yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)
    
    print(f"Saved dataset.yaml with {num_classes} classes")
    return class_names, len(train_images), len(val_images)

def create_yolo_label(xml_path, labels_dir, class_to_id, img_file):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        label_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + ".txt")
        
        with open(label_path, 'w') as f:
            for obj in root.findall('object'):
                cls_name = obj.find('name').text
                if cls_name not in class_to_id:
                    print(f"Warning: Class {cls_name} not in class mapping")
                    continue
                
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                
                size = root.find('size')
                width = float(size.find('width').text)
                height = float(size.find('height').text)
                
                if width <= 0 or height <= 0:
                    print(f"Error: Invalid image size {width}x{height} in {xml_path}")
                    continue
                
                x_center = (xmin + xmax) / (2 * width)
                y_center = (ymin + ymax) / (2 * height)
                bbox_width = (xmax - xmin) / width
                bbox_height = (ymax - ymin) / height
                
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                bbox_width = max(0.01, min(1.0, bbox_width))
                bbox_height = max(0.01, min(1.0, bbox_height))
                
                f.write(f"{class_to_id[cls_name]} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
    except Exception as e:
        print(f"Error processing {xml_path}: {e}")

class_names, train_count, val_count = convert_and_split_dataset(AUG_DATA_DIR, TRAIN_DIR, VAL_DIR, VAL_SPLIT_RATIO)

def visualize_yolo_labels(dataset_dir, title="YOLO Label Visualization"):
    print(f"Visualizing YOLO labels in {dataset_dir}...")
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not image_files:
        print("No images found for visualization")
        return
    
    img_file = random.choice(image_files)
    img_path = os.path.join(images_dir, img_file)
    label_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + ".txt")
    
    if not os.path.exists(label_path):
        print(f"Label file not found: {label_path}")
        return
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read image: {img_path}")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w = img.shape[:2]
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) != 5:
                continue
                
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            x = int((x_center - width/2) * img_w)
            y = int((y_center - height/2) * img_h)
            w = int(width * img_w)
            h = int(height * img_h)
            
            x = max(0, min(x, img_w-1))
            y = max(0, min(y, img_h-1))
            w = max(1, min(w, img_w - x))
            h = max(1, min(h, img_h - y))
            
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, f"Class {class_id}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    
    if "train" in dataset_dir.lower():
        plt.savefig("yolo_train_label_visualization.jpg")
        print("Train set visualization saved")
    else:
        plt.savefig("yolo_val_label_visualization.jpg")
        print("Validation set visualization saved")
    
    plt.close()

visualize_yolo_labels(TRAIN_DIR, "Train Set Labels")
visualize_yolo_labels(VAL_DIR, "Validation Set Labels")

print(f"YOLO dataset prepared with {len(class_names)} classes. Train: {train_count} images, Val: {val_count} images")

def train_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Creating YOLOv8x model...")
    
    try:
        model = YOLO(PRETRAINED_WEIGHTS)
        model.model.nc = len(class_names)
        model.model.names = class_names
        print(f"Model configured for {len(class_names)} classes")
        print(f"Class names: {class_names}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    except Exception as e:
        print(f"Error creating model: {e}")
        return None
    
    freeze_layer_names = [
        'model.0', 'model.1', 'model.2', 'model.3', 'model.4', 'model.5', 
        'model.6', 'model.7', 'model.8', 'model.9', 'model.10', 'model.11', 
        'model.12', 'model.13', 'model.14', 'model.15', 'model.16', 'model.17'
    ]
    
    for name, param in model.model.named_parameters():
        if any(freeze_name in name for freeze_name in freeze_layer_names):
            param.requires_grad = False
            print(f"Frozen layer: {name}")
    
    model.to(device)
    
    train_params = {
        'data': os.path.join(YOLO_DATA_DIR, "dataset.yaml"),
        'epochs': 1000,
        'batch': BATCH_SIZE,
        'imgsz': IMG_SIZE,
        'workers': min(NUM_WORKERS, 4),
        'device': 0,
        'project': CHECKPOINT_DIR,
        'name': 'yolov8x_training',
        'exist_ok': True,
        'optimizer': 'AdamW',
        'lr0': LEARNING_RATE,
        'momentum': MOMENTUM,
        'weight_decay': WEIGHT_DECAY,
        'warmup_epochs': WARMUP_EPOCHS,
        'box': BOX_GAIN,
        'cls': CLS_GAIN,
        'hsv_h': 0.01,
        'hsv_s': 0.5,
        'hsv_v': 0.3,
        'degrees': 45.0,
        'translate': 0.15,
        'scale': 0.6,
        'shear': 0.2,
        'perspective': 0.001,
        'flipud': 0.4,
        'fliplr': 0.4,
        'amp': True,
        'half': False,
        'plots': True,
        'save_period': 10,
        'cos_lr': True,
        'label_smoothing': 0.1,
        'dropout': 0.0,
        'pretrained': True,
        'close_mosaic': 10,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'rect': True
    }
    
    if version.parse(ultralytics_version) < version.parse("8.0.0"):
        train_params['obj'] = OBJ_GAIN
    else:
        train_params['dfl'] = 1.5
    
    start_time = time.time()
    last_checkpoint_time = start_time
    class_performance = defaultdict(list)
    trained_epochs = 0
    
    print("Starting training...")
    print(f"Training parameters: {train_params}")
    
    while time.time() - start_time < TRAIN_HOURS * 3600:
        try:
            results = model.train(**dict(train_params, epochs=trained_epochs+1))
            trained_epochs += 1
            
            if results:
                metrics = getattr(results, 'metrics', None)
                if metrics is None:
                    metrics = getattr(results, 'results_dict', {})
                
                print(f"Epoch {trained_epochs} results:")
                print(f"- Box loss: {metrics.get('train/box_loss', 0):.4f}")
                print(f"- Class loss: {metrics.get('train/cls_loss', 0):.4f}")
                print(f"- DFL loss: {metrics.get('train/df_loss', 0):.4f}")
                
                if math.isnan(metrics.get('train/box_loss', 0)) or \
                   math.isnan(metrics.get('train/cls_loss', 0)) or \
                   math.isnan(metrics.get('train/df_loss', 0)):
                    print("NaN loss detected! Stopping training.")
                    break
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print("GPU OOM detected, clearing cache...")
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                print(f"Training error: {e}")
                break
        except Exception as e:
            print(f"Training error: {e}")
            break
        
        current_time = time.time()
        
        if current_time - last_checkpoint_time >= CHECKPOINT_INTERVAL:
            try:
                print("Running validation...")
                val_results = model.val()
                
                if val_results:
                    print(f"Epoch {trained_epochs} validation results:")
                    print(f"- mAP50: {val_results.box.map50:.4f}")
                    print(f"- mAP50-95: {val_results.box.map:.4f}")
                    print(f"- Precision: {val_results.box.p.mean():.4f}")
                    print(f"- Recall: {val_results.box.r.mean():.4f}")
                    
                    for i, class_idx in enumerate(val_results.ap_class_index):
                        class_name = val_results.names[class_idx]
                        ap50 = val_results.box.ap50[i]
                        class_performance[class_name].append(ap50)
                    
                    save_class_performance(class_performance, trained_epochs)
                    
                    print("Per-class AP50:")
                    for i, class_idx in enumerate(val_results.ap_class_index):
                        class_name = val_results.names[class_idx]
                        ap50 = val_results.box.ap50[i]
                        print(f"  - {class_name}: {ap50:.4f}")
            except Exception as e:
                print(f"Validation error: {e}")
            
            checkpoint_time = time.strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"checkpoint_{checkpoint_time}_epoch_{trained_epochs}"
            checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
            weights_dir = os.path.join(checkpoint_path, "weights")
            os.makedirs(weights_dir, exist_ok=True)
            
            try:
                model.save(os.path.join(weights_dir, "last.pt"))
                print(f"Checkpoint saved to: {checkpoint_path}")
            except Exception as e:
                print(f"Error saving checkpoint: {e}")
            
            last_checkpoint_time = current_time
            
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                used_mem = mem_info.used / (1024 ** 3)
                print(f"Current VRAM usage: {used_mem:.2f}GB / {MAX_VRAM_USAGE}GB")
                if used_mem > MAX_VRAM_USAGE:
                    print("Warning: VRAM usage exceeds limit!")
            except Exception as e:
                print(f"VRAM monitoring error: {e}")
    
    analyze_model_performance(class_performance)
    
    final_path = "yolov8x_final.pt"
    try:
        training_dir = os.path.join(CHECKPOINT_DIR, "yolov8x_training")
        best_model_path = os.path.join(training_dir, "weights", "best.pt")
        
        if os.path.exists(best_model_path):
            shutil.copy(best_model_path, final_path)
            print(f"Final model saved to: {final_path}")
        else:
            last_model_path = os.path.join(training_dir, "weights", "last.pt")
            if os.path.exists(last_model_path):
                shutil.copy(last_model_path, final_path)
                print(f"Using last.pt as final model: {final_path}")
            else:
                print("Error: No trained model found!")
                final_path = None
    except Exception as e:
        print(f"Error saving final model: {e}")
        final_path = None
    
    total_time = (time.time() - start_time) / 3600
    print(f"Training completed. Total training time: {total_time:.2f} hours")
    
    return final_path

def save_class_performance(class_performance, epoch):
    performance_file = os.path.join(MODEL_ANALYSIS_DIR, f"class_performance_epoch_{epoch}.csv")
    df = pd.DataFrame(columns=['Class', 'AP50'])
    
    for class_name, ap_values in class_performance.items():
        if ap_values:
            df = pd.concat([df, pd.DataFrame({
                'Class': [class_name],
                'AP50': [ap_values[-1]]
            })], ignore_index=True)
    
    df.to_csv(performance_file, index=False)
    print(f"Class performance data saved to: {performance_file}")

def analyze_model_performance(class_performance):
    if not class_performance:
        print("No performance data to analyze")
        return
    
    performance_data = []
    
    for class_name, ap_values in class_performance.items():
        if ap_values:
            mean_ap = np.mean(ap_values)
            std_ap = np.std(ap_values)
            final_ap = ap_values[-1]
            improvement = final_ap - ap_values[0] if len(ap_values) > 1 else 0
            
            performance_data.append({
                'Class': class_name,
                'Mean AP50': mean_ap,
                'Std AP50': std_ap,
                'Final AP50': final_ap,
                'Improvement': improvement
            })
    
    if not performance_data:
        return
    
    df = pd.DataFrame(performance_data)
    full_performance_file = os.path.join(MODEL_ANALYSIS_DIR, "final_class_performance.csv")
    df.to_csv(full_performance_file, index=False)
    
    plt.figure(figsize=(15, 8))
    sns.barplot(x='Class', y='Final AP50', data=df.sort_values('Final AP50', ascending=False))
    plt.title('Final AP50 by Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_ANALYSIS_DIR, "final_ap50_by_class.png"))
    plt.close()
    
    plt.figure(figsize=(15, 8))
    sns.barplot(x='Class', y='Improvement', data=df.sort_values('Improvement', ascending=False))
    plt.title('AP50 Improvement During Training')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_ANALYSIS_DIR, "ap50_improvement.png"))
    plt.close()
    
    worst_classes = df[df['Final AP50'] < 0.7].sort_values('Final AP50')['Class'].tolist()
    if worst_classes:
        print(f"Classes needing improvement (AP50 < 0.7): {', '.join(worst_classes)}")
        
        with open(os.path.join(MODEL_ANALYSIS_DIR, "improvement_recommendations.txt"), 'w') as f:
            f.write("Recommendations for classes with AP50 < 0.7:\n")
            f.write("============================================\n\n")
            
            for class_name in worst_classes:
                row = df[df['Class'] == class_name].iloc[0]
                f.write(f"Class: {class_name}\n")
                f.write(f"- Final AP50: {row['Final AP50']:.4f}\n")
                f.write(f"- Improvement: {row['Improvement']:.4f}\n")
                f.write("Recommendations:\n")
                f.write("1. Increase training samples for this class\n")
                f.write("2. Add diverse augmentations specific to this class\n")
                f.write("3. Review annotation quality for this class\n")
                f.write("4. Consider hard negative mining for this class\n\n")
    
    print(f"Model performance analysis saved to: {MODEL_ANALYSIS_DIR}")

def verify_dataset():
    print("Verifying dataset integrity...")
    issues_found = False
    
    train_images = os.path.join(TRAIN_DIR, "images")
    train_labels = os.path.join(TRAIN_DIR, "labels")
    
    for img_file in os.listdir(train_images):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            base_name = os.path.splitext(img_file)[0]
            txt_path = os.path.join(train_labels, f"{base_name}.txt")
            
            if not os.path.exists(txt_path):
                print(f"Error: Missing label file {txt_path}")
                issues_found = True
            else:
                with open(txt_path, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        print(f"Warning: Empty label file {txt_path}")
    
    val_images = os.path.join(VAL_DIR, "images")
    val_labels = os.path.join(VAL_DIR, "labels")
    
    for img_file in os.listdir(val_images):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            base_name = os.path.splitext(img_file)[0]
            txt_path = os.path.join(val_labels, f"{base_name}.txt")
            
            if not os.path.exists(txt_path):
                print(f"Error: Missing label file {txt_path}")
                issues_found = True
            else:
                with open(txt_path, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        print(f"Warning: Empty label file {txt_path}")
    
    if not issues_found:
        print("Dataset integrity verified!")
    else:
        print("Warning: Dataset integrity issues found!")

verify_dataset()

if os.path.exists(AUG_DATA_DIR) and not test_mode:
    print("Cleaning previous augmented dataset...")
    shutil.rmtree(AUG_DATA_DIR)
    os.makedirs(AUG_DATA_DIR)
    print(f"Cleaned {AUG_DATA_DIR}")

if not test_mode:
    print("Starting full training...")
    final_model_path = train_model()
    if final_model_path:
        print(f"Training complete! Final model path: {final_model_path}")
    else:
        print("Training failed! Check error logs.")
else:
    print("Test mode complete. Set test_mode=False for full training.")