import os
import random
import time
import numpy as np
import cv2
import torch
import platform
import psutil
import shutil
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================ PATH CONFIGURATION ================
TARGET_DIR = "/home/legion/dataset/raw_data"  

BACKGROUND_DIR = "/home/legion/dataset/background_data"  

OUTPUT_DIR = "/home/legion/dataset/aug_data"  

PRETRAINED_MODEL = "/home/legion/dataset/yolov8x.pt"  

FINAL_MODEL_PATH = "/home/legion/dataset/trained_model.pt"  
# ===================================================

NUM_CLASSES = 32
BACKGROUNDS = 17
MIN_IMAGES_PER_CLASS = 900
TRAIN_HOURS = 100
GPU_BATCH_SIZE = 18
CPU_THREADS = 12
TRAIN_VAL_SPLIT = 0.8  

def get_hardware_info():
    print("===== HARDWARE INFORMATION =====")
    print(f"CPU: {platform.processor()}")
    print(f"CPU Cores: {os.cpu_count()}")
    print(f"System Memory: {psutil.virtual_memory().total // (1024**3)} GB")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory // (1024**3)} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        #print(f"GPU Driver: {torch.cuda.driver_version}")
    else:
        print("No GPU available")
    
    print("===============================")

def apply_color_filter(target, background, bg_x, bg_y, bg_roi_x, bg_roi_y):
    target_hsv = cv2.cvtColor(target, cv2.COLOR_RGB2HSV)
    bg_roi = background[bg_roi_y:bg_roi_y+target.shape[0], bg_roi_x:bg_roi_x+target.shape[1]]
    bg_roi_hsv = cv2.cvtColor(bg_roi, cv2.COLOR_RGB2HSV)
    
    h_mean = np.mean(bg_roi_hsv[:, :, 0])
    s_mean = np.mean(bg_roi_hsv[:, :, 1])
    v_mean = np.mean(bg_roi_hsv[:, :, 2])
    
    target_hsv[:, :, 0] = (target_hsv[:, :, 0] + h_mean) / 2
    target_hsv[:, :, 1] = (target_hsv[:, :, 1] + s_mean) / 2
    target_hsv[:, :, 2] = (target_hsv[:, :, 2] + v_mean) / 2
    
    return cv2.cvtColor(target_hsv, cv2.COLOR_HSV2RGB)

def generate_class_images(class_id, target_path, background_paths, output_dir):
    target_img = cv2.imread(target_path)
    if target_img is None:
        print(f"Error loading target image: {target_path}")
        return 0
    
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    
    generated = 0
    for i in range(MIN_IMAGES_PER_CLASS):
        bg_path = random.choice(background_paths)
        bg_img = cv2.imread(bg_path)
        if bg_img is None:
            print(f"Error loading background image: {bg_path}")
            continue
            
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
        bg_height, bg_width = bg_img.shape[:2]
        
        scale = random.uniform(0.05, 0.15)
        max_size = min(bg_height, bg_width) * np.sqrt(scale)
        aspect_ratio = target_img.shape[1] / target_img.shape[0]
        
        new_width = int(max_size * np.sqrt(aspect_ratio))
        new_height = int(max_size / np.sqrt(aspect_ratio))
        
        resized_target = cv2.resize(target_img, (new_width, new_height))
        
        angle = random.randint(0, 360)
        M = cv2.getRotationMatrix2D((new_width/2, new_height/2), angle, 1)
        rotated_target = cv2.warpAffine(resized_target, M, (new_width, new_height))
        
        x = random.randint(0, bg_width - new_width - 1)
        y = random.randint(0, bg_height - new_height - 1)
        
        bg_roi_x = random.randint(0, bg_width - new_width - 1)
        bg_roi_y = random.randint(0, bg_height - new_height - 1)
        
        filtered_target = apply_color_filter(
            rotated_target, 
            bg_img, 
            y, x,
            bg_roi_x, bg_roi_y
        )
        
        composite = bg_img.copy()
        composite[y:y+new_height, x:x+new_width] = filtered_target
        
        is_train = random.random() < TRAIN_VAL_SPLIT
        set_dir = "train" if is_train else "val"
        
        img_name = f"class_{class_id}_img_{i}.jpg"
        img_dir = os.path.join(output_dir, set_dir, "images")
        os.makedirs(img_dir, exist_ok=True)
        img_path = os.path.join(img_dir, img_name)
        cv2.imwrite(img_path, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
        
        label_dir = os.path.join(output_dir, set_dir, "labels")
        os.makedirs(label_dir, exist_ok=True)
        label_path = os.path.join(label_dir, img_name.replace(".jpg", ".txt"))
        center_x = (x + new_width / 2) / bg_width
        center_y = (y + new_height / 2) / bg_height
        width = new_width / bg_width
        height = new_height / bg_height
        
        with open(label_path, "w") as f:
            f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        
        generated += 1
    
    return generated

def generate_dataset():
    for set_name in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_DIR, set_name, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, set_name, "labels"), exist_ok=True)
    
    target_paths = [os.path.join(TARGET_DIR, f) for f in sorted(os.listdir(TARGET_DIR)) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    background_paths = [os.path.join(BACKGROUND_DIR, f) for f in sorted(os.listdir(BACKGROUND_DIR)) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not target_paths:
        print(f"Error: No valid target images found in {TARGET_DIR}")
        return 0
        
    if not background_paths:
        print(f"Error: No valid background images found in {BACKGROUND_DIR}")
        return 0
    
    print(f"Starting dataset generation for {NUM_CLASSES} classes...")
    start_time = time.time()
    
    total_generated = 0
    with ThreadPoolExecutor(max_workers=CPU_THREADS) as executor:
        futures = []
        for class_id, target_path in enumerate(target_paths[:NUM_CLASSES]):
            futures.append(executor.submit(
                generate_class_images, 
                class_id, 
                target_path, 
                background_paths, 
                OUTPUT_DIR
            ))
        
        for idx, future in enumerate(as_completed(futures)):
            try:
                generated = future.result()
                total_generated += generated
                print(f"Class {idx}: Generated {generated} images")
            except Exception as e:
                print(f"Error generating images for class {idx}: {str(e)}")
    
    total_time = time.time() - start_time
    print(f"Dataset generation completed in {total_time:.2f} seconds")
    print(f"Total images generated: {total_generated}")
    
    train_count = len(os.listdir(os.path.join(OUTPUT_DIR, "train", "images")))
    val_count = len(os.listdir(os.path.join(OUTPUT_DIR, "val", "images")))
    print(f"Training images: {train_count}, Validation images: {val_count}")
    
    return total_generated

def train_yolov8():
    print("Initializing YOLOv8 training...")
    
    yaml_content = f"path: {OUTPUT_DIR}\n"
    yaml_content += "train: train/images\n"
    yaml_content += "val: val/images\n"
    yaml_content += "names:\n"
    for i in range(NUM_CLASSES):
        yaml_content += f"  {i}: class_{i}\n"
    
    yaml_path = os.path.join(OUTPUT_DIR, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    
    try:
        model = YOLO(PRETRAINED_MODEL)
        print("Successfully loaded pretrained model")
    except Exception as e:
        print(f"Error loading pretrained model: {str(e)}")
        return
    
    train_time = TRAIN_HOURS * 3600
    start_time = time.time()
    
    try:
        model.train(
            data=yaml_path,
            epochs=300,
            imgsz=416,
            batch=GPU_BATCH_SIZE,
            device=0,
            patience=50,     
            amp=True,          
            workers=8,
            cache="ram",
            optimizer="AdamW",
            lr0=0.001,         
            lrf=0.01,         
            momentum=0.9,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            box=7.5,           
            cls=0.5,           
            dfl=1.5,           
            hsv_h=0.015,       
            hsv_s=0.7,        
            hsv_v=0.4,         
            degrees=10.0,      
            translate=0.1,     
            scale=0.5,         
            shear=2.0,         
            perspective=0.001,  
            flipud=0.5,        
            fliplr=0.5       
        )
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return
    
    elapsed = time.time() - start_time
    while elapsed < train_time:
        time.sleep(60)
        elapsed = time.time() - start_time
        progress = elapsed / train_time * 100
        print(f"Training progress: {progress:.1f}% - Elapsed: {elapsed/3600:.1f}h / {TRAIN_HOURS}h")
        break
    
    try:
        model.save(FINAL_MODEL_PATH)
        print(f"Training completed. Model saved as {FINAL_MODEL_PATH}")
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        return
    
    try:
        metrics = model.val()
        print(f"Validation results:")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        print(f"  mAP50: {metrics.box.map50:.4f}")
        print(f"  Precision: {metrics.box.precision:.4f}")
        print(f"  Recall: {metrics.box.recall:.4f}")
    except Exception as e:
        print(f"Error during validation: {str(e)}")

def main():
    get_hardware_info()
    
    total_images = generate_dataset()
    
    if total_images < NUM_CLASSES * MIN_IMAGES_PER_CLASS * 0.9:
        print("Error: Insufficient images generated")
        return
    
    train_yolov8()

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()