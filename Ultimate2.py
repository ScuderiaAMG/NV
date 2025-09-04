# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# YOLOv11s训练脚本 - 靶心目标检测
# 硬件: CPU Intel i7-14700HX, GPU RTX 4060 8GB
# 软件: Ubuntu 20.04, CUDA 12.8, Python 3.8.10
# 作者: Legion
# 日期: 2025-09-03
# """

# import os
# import cv2
# import torch
# import numpy as np
# import random
# from PIL import Image
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import tensorflow as tf
# from ultralytics import YOLO
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# import xml.etree.ElementTree as ET
# from datetime import datetime

# def setup_environment():
#     """
#     设置环境和GPU预热
#     """
#     # 设置随机种子确保可重现性
#     torch.manual_seed(42)
#     np.random.seed(42)
#     random.seed(42)
    
#     # 检查GPU可用性
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
    
#     # 输出硬件信息
#     if torch.cuda.is_available():
#         print(f"GPU: {torch.cuda.get_device_name(0)}")
#         print(f"CUDA version: {torch.version.cuda}")
#         print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
#     # GPU预热
#     if torch.cuda.is_available():
#         # 进行简单的矩阵运算预热GPU
#         warm_up_tensor = torch.randn(1000, 1000).to(device)
#         for _ in range(10):
#             _ = torch.matmul(warm_up_tensor, warm_up_tensor)
#         print("GPU warm-up completed")
    
#     return device

# def load_background_images(background_path):
#     """
#     加载背景图片
#     """
#     background_images = []
#     valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
#     for filename in os.listdir(background_path):
#         if filename.lower().endswith(valid_extensions):
#             img_path = os.path.join(background_path, filename)
#             img = cv2.imread(img_path)
#             if img is not None:
#                 background_images.append(img)
#                 print(f"Loaded background image: {filename}")
    
#     print(f"Total background images loaded: {len(background_images)}")
#     return background_images

# # def detect_target_center_hough(image, max_detections=10):
# #     """
# #     使用霍夫圆变换检测靶心圆心
# #     """
# #     # 转换为灰度图
# #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
# #     # 应用高斯模糊减少噪声
# #     blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
# #     # 霍夫圆检测参数
# #     dp = 1  # 累加器分辨率与图像分辨率的反比
# #     min_dist = 50  # 检测到的圆心之间的最小距离
# #     param1 = 100  # Canny边缘检测的高阈值
# #     param2 = 20   # 累加器阈值
# #     min_radius = 10  # 最小圆半径
# #     max_radius = 1536  # 最大圆半径
    
# #     # 如果背景很大，max_radius应该更大，比如设置为背景宽度的一半
# #     height, width = image.shape[:2]
# #     max_radius = max(height, width) // 2  # 动态设置max_radius

# #     best_circle = None
# #     max_confidence = 0
    
# #     # 多次检测取最佳结果
# #     for _ in range(max_detections):
# #         circles = cv2.HoughCircles(
# #             blurred, 
# #             cv2.HOUGH_GRADIENT, 
# #             dp=dp,
# #             minDist=min_dist,
# #             param1=param1,
# #             param2=param2,
# #             minRadius=min_radius,
# #             maxRadius=max_radius
# #         )
        
# #         if circles is not None:
# #             circles = np.uint16(np.around(circles))
# #             for circle in circles[0, :]:
# #                 x, y, radius = circle
# #                 # 计算置信度（基于圆的完整性和对比度）
# #                 confidence = calculate_circle_confidence(gray, x, y, radius)
# #                 if confidence > max_confidence:
# #                     max_confidence = confidence
# #                     best_circle = (x, y, radius)
    
# #     return best_circle, max_confidence
# # def detect_target_center_hough(image, max_detections=10):
# #     """
# #     使用霍夫圆变换检测靶心圆心（改进版）
# #     """
# #     # 转换为灰度图
# #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
# #     # 应用高斯模糊减少噪声 - 根据图像尺寸调整核大小
# #     height, width = image.shape[:2]
# #     kernel_size = int(max(height, width) / 300)  # 动态核大小
# #     kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1  # 确保为奇数
# #     blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 2)
    
# #     # 霍夫圆检测参数 - 根据图像尺寸动态调整
# #     dp = 1  # 累加器分辨率与图像分辨率的反比
# #     min_dist = int(min(height, width) / 10)  # 最小距离基于图像尺寸
# #     param1 = 100  # Canny边缘检测的高阈值
# #     param2 = 20   # 降低累加器阈值，使检测更敏感（原为30）
# #     min_radius = int(min(height, width) / 50)  # 最小半径基于图像尺寸
# #     max_radius = int(min(height, width) / 2)   # 最大半径基于图像尺寸
    
# #     best_circle = None
# #     max_confidence = 0
    
# #     # 多次检测取最佳结果
# #     for _ in range(max_detections):
# #         circles = cv2.HoughCircles(
# #             blurred, 
# #             cv2.HOUGH_GRADIENT, 
# #             dp=dp,
# #             minDist=min_dist,
# #             param1=param1,
# #             param2=param2,
# #             minRadius=min_radius,
# #             maxRadius=max_radius
# #         )
        
# #         if circles is not None:
# #             circles = np.uint16(np.around(circles))
# #             for circle in circles[0, :]:
# #                 x, y, radius = circle
# #                 # 计算置信度（基于圆的完整性和对比度）
# #                 confidence = calculate_circle_confidence(gray, x, y, radius)
# #                 if confidence > max_confidence:
# #                     max_confidence = confidence
# #                     best_circle = (x, y, radius)
    
# #     return best_circle, max_confidence
# def detect_target_center_hough(image, max_detections=10):
#     """
#     使用霍夫圆变换检测靶心圆心（进一步优化版）
#     """
#     # 首先缩小图像以加快处理速度
#     height, width = image.shape[:2]
#     scale_factor = 0.5  # 缩小一半
#     small_image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
    
#     # 转换为灰度图
#     gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
    
#     # 应用高斯模糊减少噪声
#     kernel_size = 9
#     blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 2)
    
#     # 使用自适应阈值增强对比度
#     thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                                   cv2.THRESH_BINARY, 11, 2)
    
#     # 霍夫圆检测参数
#     dp = 1
#     min_dist = int(min(height, width) * scale_factor / 10)
#     param1 = 100
#     param2 = 15  # 进一步降低阈值以提高检测灵敏度
#     min_radius = int(min(height, width) * scale_factor / 50)
#     max_radius = int(min(height, width) * scale_factor / 2)
    
#     best_circle = None
#     max_confidence = 0
    
#     # 多次检测取最佳结果
#     for _ in range(max_detections):
#         circles = cv2.HoughCircles(
#             thresh,  # 使用阈值化图像
#             cv2.HOUGH_GRADIENT, 
#             dp=dp,
#             minDist=min_dist,
#             param1=param1,
#             param2=param2,
#             minRadius=min_radius,
#             maxRadius=max_radius
#         )
        
#         if circles is not None:
#             circles = np.uint16(np.around(circles))
#             for circle in circles[0, :]:
#                 x, y, radius = circle
#                 # 将坐标缩放回原始尺寸
#                 x = int(x / scale_factor)
#                 y = int(y / scale_factor)
#                 radius = int(radius / scale_factor)
                
#                 # 计算置信度
#                 confidence = calculate_circle_confidence(gray, x, y, radius)
#                 if confidence > max_confidence:
#                     max_confidence = confidence
#                     best_circle = (x, y, radius)
    
#     return best_circle, max_confidence

# # def calculate_circle_confidence(gray_img, x, y, radius):
# #     """
# #     计算圆的置信度评分
# #     """
# #     height, width = gray_img.shape
    
# #     # 确保圆在图像范围内
# #     x_min = max(0, x - radius)
# #     x_max = min(width, x + radius)
# #     y_min = max(0, y - radius)
# #     y_max = min(height, y + radius)
    
# #     if x_min >= x_max or y_min >= y_max:
# #         return 0
    
# #     # 提取ROI区域
# #     roi = gray_img[y_min:y_max, x_min:x_max]
    
# #     if roi.size == 0:
# #         return 0
    
# #     # 计算边缘强度和对比度作为置信度
# #     edges = cv2.Canny(roi, 50, 150)
# #     edge_strength = np.mean(edges) if edges.size > 0 else 0
# #     contrast = np.std(roi) if roi.size > 0 else 0
    
# #     confidence = 0.6 * edge_strength + 0.4 * contrast
# #     return confidence
# def calculate_circle_confidence(gray_img, x, y, radius):
#     """
#     计算圆的置信度评分（修复数值溢出问题）
#     """
#     height, width = gray_img.shape
    
#     # 确保圆在图像范围内（使用有符号整数计算）
#     x_min = max(0, int(x) - int(radius))
#     x_max = min(width, int(x) + int(radius))
#     y_min = max(0, int(y) - int(radius))
#     y_max = min(height, int(y) + int(radius))
    
#     if x_min >= x_max or y_min >= y_max:
#         return 0
    
#     # 提取ROI区域
#     roi = gray_img[y_min:y_max, x_min:x_max]
    
#     if roi.size == 0:
#         return 0
    
#     # 计算边缘强度和对比度作为置信度
#     edges = cv2.Canny(roi, 50, 150)
#     edge_strength = np.mean(edges) if edges.size > 0 else 0
#     contrast = np.std(roi) if roi.size > 0 else 0
    
#     confidence = 0.6 * edge_strength + 0.4 * contrast
#     return confidence

# def augment_dataset(raw_dataset_path, output_path, target_per_class=900):
#     """
#     增强数据集：应用多种数据增强技术
#     """
#     os.makedirs(output_path, exist_ok=True)
    
#     # Albumentations增强管道
#     transform = A.Compose([
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         A.Rotate(limit=45, p=0.8),
#         A.RandomBrightnessContrast(p=0.5),
#         A.HueSaturationValue(p=0.5),
#         A.Blur(blur_limit=3, p=0.3),
#         A.CLAHE(p=0.3),
#         A.RandomGamma(p=0.3),
#         A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
#         ToTensorV2()
#     ])
    
#     class_folders = sorted([f for f in os.listdir(raw_dataset_path) 
#                           if os.path.isdir(os.path.join(raw_dataset_path, f))])
    
#     print(f"Found {len(class_folders)} classes to augment")
    
#     for class_folder in class_folders:
#         class_path = os.path.join(raw_dataset_path, class_folder)
#         output_class_path = os.path.join(output_path, class_folder)
#         os.makedirs(output_class_path, exist_ok=True)
        
#         # 获取类别图像
#         class_images = []
#         for filename in os.listdir(class_path):
#             if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 img_path = os.path.join(class_path, filename)
#                 img = cv2.imread(img_path)
#                 if img is not None:
#                     class_images.append(img)
        
#         if not class_images:
#             print(f"No images found for class {class_folder}, skipping")
#             continue
        
#         print(f"Augmenting class {class_folder} with {len(class_images)} base images")
        
#         # 计算需要生成的图像数量
#         existing_count = len(class_images)
#         needed_count = target_per_class - existing_count
        
#         if needed_count <= 0:
#             print(f"Class {class_folder} already has {existing_count} images, skipping augmentation")
#             continue
        
#         # 应用增强
#         augmented_count = 0
#         while augmented_count < needed_count:
#             for base_img in class_images:
#                 if augmented_count >= needed_count:
#                     break
                
#                 try:
#                     # 应用多种增强技术
#                     augmented = transform(image=base_img)
#                     augmented_img = augmented['image']
                    
#                     if isinstance(augmented_img, torch.Tensor):
#                         augmented_img = augmented_img.permute(1, 2, 0).numpy()
#                         augmented_img = (augmented_img * 255).astype(np.uint8)
                    
#                     # 保存增强后的图像
#                     output_filename = f"{class_folder}_aug_{augmented_count:04d}.jpg"
#                     output_filepath = os.path.join(output_class_path, output_filename)
#                     cv2.imwrite(output_filepath, augmented_img)
                    
#                     augmented_count += 1
#                     if augmented_count % 100 == 0:
#                         print(f"Generated {augmented_count} augmented images for {class_folder}")
                        
#                 except Exception as e:
#                     print(f"Error augmenting image: {e}")
#                     continue
        
#         print(f"Completed augmentation for {class_folder}. Total images: {existing_count + augmented_count}")

# # def create_synthetic_dataset(augmented_path, background_images, output_path,target_scale=5.0):
# #     """
# #     创建合成数据集：将增强图像融合到背景上
# #     """
# #     os.makedirs(output_path, exist_ok=True)
# #     os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
# #     os.makedirs(os.path.join(output_path, "labels"), exist_ok=True)
    
# #     # 获取所有类别
# #     classes = sorted([f for f in os.listdir(augmented_path) 
# #                      if os.path.isdir(os.path.join(augmented_path, f))])
    
# #     print(f"Creating synthetic dataset with {len(classes)} classes")
    
# #     # 为每个类别创建图像和标注
# #     image_counter = 0
    
# #     for class_idx, class_name in enumerate(classes):
# #         class_path = os.path.join(augmented_path, class_name)
# #         class_images = [f for f in os.listdir(class_path) 
# #                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
# #         print(f"Processing class {class_name} with {len(class_images)} images")
# #         # ##############################################################
# #         #  # 遍历类别图像
# #         # for img_filename in class_images:
# #         #     # 随机选择背景图像
# #         #     bg_img = random.choice(background_images).copy()
# #         #     bg_height, bg_width = bg_img.shape[:2]
            
# #         #     # 加载增强图像
# #         #     aug_img_path = os.path.join(class_path, img_filename)
# #         #     aug_img = cv2.imread(aug_img_path, cv2.IMREAD_UNCHANGED)
            
# #         #     if aug_img is None:
# #         #         continue
            
# #         #     # 检测背景图像中的靶心
# #         #     target_center, confidence = detect_target_center_hough(bg_img)
            
# #         #     if target_center is None:
# #         #         # 如果没有检测到靶心，使用图像中心
# #         #         target_x, target_y = bg_width // 2, bg_height // 2
# #         #         print(f"No target detected in background, using center: ({target_x}, {target_y})")
# #         #     else:
# #         #         target_x, target_y, target_radius = target_center
            
# #         #     # 调整增强图像大小（根据靶心大小或默认大小）
# #         #     aug_height, aug_width = aug_img.shape[:2]
# #         #     max_size = min(bg_width, bg_height) // 3
# #         #     scale_factor = min(max_size / max(aug_width, aug_height), 1.0)
# #         #     new_width = int(aug_width * scale_factor)
# #         #     new_height = int(aug_height * scale_factor)
            
# #         #     if new_width <= 0 or new_height <= 0:
# #         #         continue
                
# #         #     aug_img_resized = cv2.resize(aug_img, (new_width, new_height))############################
# #         for img_filename in class_images:
# #             # 随机选择背景图像
# #             bg_img = random.choice(background_images).copy()
# #             bg_height, bg_width = bg_img.shape[:2]
            
# #             # 加载增强图像
# #             aug_img_path = os.path.join(class_path, img_filename)
# #             aug_img = cv2.imread(aug_img_path, cv2.IMREAD_UNCHANGED)
            
# #             if aug_img is None:
# #                 continue
            
# #             # 检测背景图像中的靶心
# #             target_center, confidence = detect_target_center_hough(bg_img)
            
# #             if target_center is None:
# #                 # 如果没有检测到靶心，使用图像中心
# #                 target_x, target_y = bg_width // 2, bg_height // 2
# #                 print(f"No target detected in background, using center: ({target_x}, {target_y})")
# #             else:
# #                 target_x, target_y, target_radius = target_center
            
# #             # 调整增强图像大小：允许放大，基于target_scale参数
# #             aug_height, aug_width = aug_img.shape[:2]
# #             # 计算缩放因子：根据target_scale放大，但不超过背景的1/3（避免过大）
# #             scale_factor = target_scale  # 直接使用缩放倍数
# #             new_width = int(aug_width * scale_factor)
# #             new_height = int(aug_height * scale_factor)
            
# #             # 确保新尺寸不超过背景的1/3
# #             max_allowed_size = min(bg_width, bg_height) // 3
# #             if new_width > max_allowed_size or new_height > max_allowed_size:
# #                 # 如果太大，则缩放到最大允许尺寸
# #                 scale_factor = max_allowed_size / max(new_width, new_height)
# #                 new_width = int(aug_width * scale_factor)
# #                 new_height = int(aug_height * scale_factor)
            
# #             if new_width <= 0 or new_height <= 0:
# #                 continue
                
# #             aug_img_resized = cv2.resize(aug_img, (new_width, new_height))
            
# #             # 计算放置位置（以靶心为中心）
# #             start_x = max(0, target_x - new_width // 2)
# #             start_y = max(0, target_y - new_height // 2)
# #             end_x = min(bg_width, start_x + new_width)
# #             end_y = min(bg_height, start_y + new_height)
            
# #             # 调整图像大小以适应边界
# #             if end_x - start_x != new_width or end_y - start_y != new_height:
# #                 aug_img_resized = cv2.resize(aug_img_resized, 
# #                                            (end_x - start_x, end_y - start_y))
            
# #             # Alpha融合（如果增强图像有alpha通道）
# #             if aug_img_resized.shape[2] == 4:
# #                 alpha = aug_img_resized[:, :, 3] / 255.0
# #                 for c in range(3):
# #                     bg_img[start_y:end_y, start_x:end_x, c] = \
# #                         (1 - alpha) * bg_img[start_y:end_y, start_x:end_x, c] + \
# #                         alpha * aug_img_resized[:, :, c]
# #             else:
# #                 # 直接覆盖
# #                 bg_img[start_y:end_y, start_x:end_x] = aug_img_resized
            
# #             # 保存合成图像
# #             image_filename = f"synthetic_{image_counter:06d}.jpg"
# #             image_path = os.path.join(output_path, "images", image_filename)
# #             cv2.imwrite(image_path, bg_img)
            
# #             # 创建YOLO格式的标注文件
# #             label_filename = f"synthetic_{image_counter:06d}.txt"
# #             label_path = os.path.join(output_path, "labels", label_filename)
            
# #             # 计算边界框（归一化坐标）
# #             x_center = (start_x + (end_x - start_x) / 2) / bg_width
# #             y_center = (start_y + (end_y - start_y) / 2) / bg_height
# #             width = (end_x - start_x) / bg_width
# #             height = (end_y - start_y) / bg_height
            
# #             with open(label_path, 'w') as f:
# #                 f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
# #             image_counter += 1
            
# #             if image_counter % 100 == 0:
# #                 print(f"Created {image_counter} synthetic images")
    
# #     print(f"Completed synthetic dataset creation. Total images: {image_counter}")
    
# #     # 创建classes.txt
# #     with open(os.path.join(output_path, "classes.txt"), 'w') as f:
# #         for class_name in classes:
# #             f.write(f"{class_name}\n")
    
# #     return classes, image_counter
# def create_synthetic_dataset(augmented_path, background_images, output_path, target_scale=5.0):
#     """
#     创建合成数据集：将增强图像融合到背景上（添加性能优化）
#     """
#     os.makedirs(output_path, exist_ok=True)
#     os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
#     os.makedirs(os.path.join(output_path, "labels"), exist_ok=True)
    
#     # 获取所有类别
#     classes = sorted([f for f in os.listdir(augmented_path) 
#                      if os.path.isdir(os.path.join(augmented_path, f))])
    
#     print(f"Creating synthetic dataset with {len(classes)} classes")
    
#     # 为每个类别创建图像和标注
#     image_counter = 0
    
#     # 预计算背景图像的靶心位置（避免重复计算）
#     bg_target_cache = {}
#     for i, bg_img in enumerate(background_images):
#         target_center, confidence = detect_target_center_hough(bg_img)
#         bg_target_cache[i] = (target_center, confidence)
    
#     for class_idx, class_name in enumerate(classes):
#         class_path = os.path.join(augmented_path, class_name)
#         class_images = [f for f in os.listdir(class_path) 
#                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
#         print(f"Processing class {class_name} with {len(class_images)} images")
        
#         for img_idx, img_filename in enumerate(class_images):
#             # 显示进度
#             if img_idx % 100 == 0:
#                 print(f"Processing image {img_idx}/{len(class_images)} for class {class_name}")
            
#             # 随机选择背景图像
#             bg_idx = random.randint(0, len(background_images) - 1)
#             bg_img = background_images[bg_idx].copy()
#             bg_height, bg_width = bg_img.shape[:2]
            
#             # 从缓存获取靶心位置
#             target_center, confidence = bg_target_cache[bg_idx]
            
#             # 加载增强图像
#             aug_img_path = os.path.join(class_path, img_filename)
#             aug_img = cv2.imread(aug_img_path, cv2.IMREAD_UNCHANGED)
            
#             if aug_img is None:
#                 continue
            
#             if target_center is None:
#                 # 如果没有检测到靶心，使用图像中心
#                 target_x, target_y = bg_width // 2, bg_height // 2
#             else:
#                 target_x, target_y, target_radius = target_center
            
#             # 调整增强图像大小：允许放大，基于target_scale参数
#             aug_height, aug_width = aug_img.shape[:2]
#             # 计算缩放因子：根据target_scale放大，但不超过背景的1/3（避免过大）
#             scale_factor = target_scale  # 直接使用缩放倍数
#             new_width = int(aug_width * scale_factor)
#             new_height = int(aug_height * scale_factor)
            
#             # 确保新尺寸不超过背景的1/3
#             max_allowed_size = min(bg_width, bg_height) // 3
#             if new_width > max_allowed_size or new_height > max_allowed_size:
#                 # 如果太大，则缩放到最大允许尺寸
#                 scale_factor = max_allowed_size / max(new_width, new_height)
#                 new_width = int(aug_width * scale_factor)
#                 new_height = int(aug_height * scale_factor)
            
#             if new_width <= 0 or new_height <= 0:
#                 continue
                
#             aug_img_resized = cv2.resize(aug_img, (new_width, new_height))
            
#             # 计算放置位置（以靶心为中心）
#             start_x = max(0, target_x - new_width // 2)
#             start_y = max(0, target_y - new_height // 2)
#             end_x = min(bg_width, start_x + new_width)
#             end_y = min(bg_height, start_y + new_height)
            
#             # 调整图像大小以适应边界
#             if end_x - start_x != new_width or end_y - start_y != new_height:
#                 aug_img_resized = cv2.resize(aug_img_resized, 
#                                            (end_x - start_x, end_y - start_y))
            
#             # Alpha融合（如果增强图像有alpha通道）
#             if aug_img_resized.shape[2] == 4:
#                 alpha = aug_img_resized[:, :, 3] / 255.0
#                 for c in range(3):
#                     bg_img[start_y:end_y, start_x:end_x, c] = \
#                         (1 - alpha) * bg_img[start_y:end_y, start_x:end_x, c] + \
#                         alpha * aug_img_resized[:, :, c]
#             else:
#                 # 直接覆盖
#                 bg_img[start_y:end_y, start_x:end_x] = aug_img_resized
            
#             # 保存合成图像
#             image_filename = f"synthetic_{image_counter:06d}.jpg"
#             image_path = os.path.join(output_path, "images", image_filename)
#             cv2.imwrite(image_path, bg_img)
            
#             # 创建YOLO格式的标注文件
#             label_filename = f"synthetic_{image_counter:06d}.txt"
#             label_path = os.path.join(output_path, "labels", label_filename)
            
#             # 计算边界框（归一化坐标）
#             x_center = (start_x + (end_x - start_x) / 2) / bg_width
#             y_center = (start_y + (end_y - start_y) / 2) / bg_height
#             width = (end_x - start_x) / bg_width
#             height = (end_y - start_y) / bg_height
            
#             with open(label_path, 'w') as f:
#                 f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
#             image_counter += 1
            
#             if image_counter % 100 == 0:
#                 print(f"Created {image_counter} synthetic images")
    
#     print(f"Completed synthetic dataset creation. Total images: {image_counter}")
    
#     # 创建classes.txt
#     with open(os.path.join(output_path, "classes.txt"), 'w') as f:
#         for class_name in classes:
#             f.write(f"{class_name}\n")
    
#     return classes, image_counter

# def setup_yolo_dataset_structure(synthetic_path, output_yolo_path, train_ratio=0.8):
#     """
#     设置YOLO格式的数据集结构
#     """
#     os.makedirs(output_yolo_path, exist_ok=True)
    
#     images_dir = os.path.join(synthetic_path, "images")
#     labels_dir = os.path.join(synthetic_path, "labels")
    
#     # 获取所有图像文件
#     all_images = sorted([f for f in os.listdir(images_dir) 
#                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
#     # 分割训练集和验证集
#     train_images, val_images = train_test_split(
#         all_images, train_size=train_ratio, random_state=42
#     )
    
#     # 创建YOLO数据集目录结构
#     yolo_dirs = {
#         'train': os.path.join(output_yolo_path, 'train'),
#         'val': os.path.join(output_yolo_path, 'val'),
#         'test': os.path.join(output_yolo_path, 'test')
#     }
    
#     for dir_type, dir_path in yolo_dirs.items():
#         os.makedirs(os.path.join(dir_path, 'images'), exist_ok=True)
#         os.makedirs(os.path.join(dir_path, 'labels'), exist_ok=True)
    
#     # 复制文件到相应目录
#     for img_name in train_images:
#         # 复制图像
#         src_img = os.path.join(images_dir, img_name)
#         dst_img = os.path.join(yolo_dirs['train'], 'images', img_name)
#         os.symlink(src_img, dst_img)  # 使用符号链接节省空间
        
#         # 复制标注
#         label_name = os.path.splitext(img_name)[0] + '.txt'
#         src_label = os.path.join(labels_dir, label_name)
#         dst_label = os.path.join(yolo_dirs['train'], 'labels', label_name)
#         if os.path.exists(src_label):
#             os.symlink(src_label, dst_label)
    
#     for img_name in val_images:
#         src_img = os.path.join(images_dir, img_name)
#         dst_img = os.path.join(yolo_dirs['val'], 'images', img_name)
#         os.symlink(src_img, dst_img)
        
#         label_name = os.path.splitext(img_name)[0] + '.txt'
#         src_label = os.path.join(labels_dir, label_name)
#         dst_label = os.path.join(yolo_dirs['val'], 'labels', label_name)
#         if os.path.exists(src_label):
#             os.symlink(src_label, dst_label)
    
#     # 创建dataset.yaml文件
#     classes_file = os.path.join(synthetic_path, "classes.txt")
#     with open(classes_file, 'r') as f:
#         class_names = [line.strip() for line in f.readlines()]
    
#     yaml_content = f"""
# # YOLOv11 dataset configuration
# path: {output_yolo_path}
# train: train/images
# val: val/images
# test: test/images

# # Number of classes
# nc: {len(class_names)}

# # Class names
# names: {class_names}
# """
    
#     yaml_path = os.path.join(output_yolo_path, "dataset.yaml")
#     with open(yaml_path, 'w') as f:
#         f.write(yaml_content)
    
#     print(f"YOLO dataset structure created at {output_yolo_path}")
#     print(f"Dataset includes {len(class_names)} classes: {class_names}")
    
#     return yaml_path, class_names

# def train_yolov11s(dataset_yaml, output_dir="./runs/train"):
#     """
#     训练YOLOv11s模型
#     """
#     # 加载预训练模型
#     print("Loading YOLOv11s model...")
#     model = YOLO('yolo11s.pt')  # 使用预训练的YOLOv11s模型
    
#     # 训练参数
#     training_params = {
#         'data': dataset_yaml,
#         'epochs': 100,
#         'imgsz': 640,
#         'batch': 8,  # 根据GPU内存调整
#         'workers': 4,
#         'device': 0,  # 使用GPU 0
#         'optimizer': 'AdamW',  # 使用AdamW优化器
#         'lr0': 0.001,  # 初始学习率
#         'lrf': 0.01,  # 最终学习率
#         'momentum': 0.937,
#         'weight_decay': 0.0005,
#         'warmup_epochs': 3.0,
#         'warmup_momentum': 0.8,
#         'box': 7.5,  # 边界框损失权重
#         'cls': 0.5,  # 分类损失权重
#         'dfl': 1.5,  # 分布焦点损失权重
#         'close_mosaic': 10,
#         'amp': True,  # 自动混合精度训练
#         'project': output_dir,
#         'name': 'yolov11s_target_detection',
#         'exist_ok': True,
#         'patience': 50,  # 早停耐心值
#         'save_period': 10,  # 每10个epoch保存一次检查点
#         'single_cls': False,
#         'overlap_mask': True,
#         'mask_ratio': 4,
#         'dropout': 0.0,
#         'val': True,  # 开启验证
#         'split': 'val',
#         'save_json': False,
#         'save_hybrid': False,
#         'conf': 0.001,  # 验证置信度阈值
#         'iou': 0.6,  # NMS IoU阈值
#         'max_det': 300,  # 每张图像最大检测数
#         'half': False,  # 使用半精度推理
#         'plots': True  # 保存训练曲线图
#     }
    
#     # 开始训练
#     print("Starting YOLOv11s training...")
#     results = model.train(**training_params)
    
#     # 在验证集上评估模型
#     print("Evaluating model on validation set...")
#     metrics = model.val()
    
#     print(f"Training completed. Results saved to {output_dir}")
#     print(f"mAP50-95: {metrics.box.map:.4f}")
#     print(f"mAP50: {metrics.box.map50:.4f}")
    
#     return model, results

# def optimize_model(model, dataset_yaml):
#     """
#     模型优化：超参数调优、后处理优化等
#     """
#     print("Starting model optimization...")
    
#     # 超参数优化（使用Ray Tune）
#     try:
#         result_grid = model.tune(
#             data=dataset_yaml,
#             use_ray=True,
#             grace_period=10,
#             gpu_per_trial=1,
#             iterations=20,
#             # 自定义搜索空间
#             space={
#                 'lr0': (1e-5, 1e-1),
#                 'lrf': (0.01, 1.0),
#                 'momentum': (0.6, 0.98),
#                 'weight_decay': (0.0, 0.001),
#                 'warmup_epochs': (0.0, 5.0),
#                 'warmup_momentum': (0.0, 0.95),
#                 'box': (0.02, 0.2),
#                 'cls': (0.2, 4.0),
#             }
#         )
#         print("Hyperparameter tuning completed")
#     except Exception as e:
#         print(f"Hyperparameter tuning failed: {e}")
    
#     # 导出为TensorFlow格式以加速推理
#     try:
#         model.export(format="pb")  # 导出为TF GraphDef
#         print("Model exported to TensorFlow format")
#     except Exception as e:
#         print(f"Model export failed: {e}")
    
#     return model

# def analyze_results(model, dataset_path):
#     """
#     分析训练结果和错误模式
#     """
#     print("Analyzing training results and error patterns...")
    
#     # 加载最佳模型
#     best_model_path = os.path.join(model.save_dir, "weights", "best.pt")
#     if os.path.exists(best_model_path):
#         model = YOLO(best_model_path)
    
#     # 在测试集上评估
#     test_results = model.val(
#         data=os.path.join(dataset_path, "dataset.yaml"),
#         split="test",
#         conf=0.25,
#         iou=0.45
#     )
    
#     # 分析错误模式
#     error_analysis = {
#         "false_positives": 0,
#         "false_negatives": 0,
#         "localization_errors": 0,
#         "classification_errors": 0
#     }
    
#     # 这里可以添加更详细的错误分析逻辑
#     # 例如，使用模型预测并分析混淆矩阵等
    
#     print("Error analysis completed")
#     print(f"Test results - mAP50-95: {test_results.box.map:.4f}")
    
#     return error_analysis

# def main():
#     """
#     主函数：执行完整的训练流程
#     """
#     print("Starting YOLOv11s target detection training pipeline")
#     start_time = datetime.now()
    
#     # 设置路径
#     base_path = "/home/legion/dataset"
#     raw_dataset_path = os.path.join(base_path, "raw_dataset")
#     background_path = os.path.join(base_path, "trash", "empty")
#     augmented_path = os.path.join(base_path, "augmented_dataset")
#     synthetic_path = os.path.join(base_path, "synthetic_dataset")
#     yolo_dataset_path = os.path.join(base_path, "yolo_dataset")
    
#     # 1. 设置环境和GPU预热
#     device = setup_environment()
    
#     # 2. 加载背景图片
#     background_images = load_background_images(background_path)
    
#     # 3. 检测背景中的靶心（可选，用于验证）
#     if background_images:
#         sample_bg = background_images[0]
#         target_center, confidence = detect_target_center_hough(sample_bg)
#         if target_center:
#             x, y, r = target_center
#             print(f"Sample target detection - Center: ({x}, {y}), Radius: {r}, Confidence: {confidence:.2f}")
    
#     # 4. 数据增强
#     augment_dataset(raw_dataset_path, augmented_path, target_per_class=900)
    
#     # 5. 创建合成数据集
#     # classes, num_images = create_synthetic_dataset(augmented_path, background_images, synthetic_path)
#     classes, num_images = create_synthetic_dataset(augmented_path, background_images, synthetic_path, target_scale=5.0)
    
#     # 6. 设置YOLO数据集结构
#     dataset_yaml, class_names = setup_yolo_dataset_structure(synthetic_path, yolo_dataset_path)
    
#     # 7. 训练YOLOv11s模型
#     model, results = train_yolov11s(dataset_yaml)
    
#     # 8. 模型优化
#     optimized_model = optimize_model(model, dataset_yaml)
    
#     # 9. 结果分析
#     error_analysis = analyze_results(optimized_model, yolo_dataset_path)
    
#     # 输出训练总结
#     end_time = datetime.now()
#     duration = end_time - start_time
    
#     print("\n" + "="*50)
#     print("TRAINING SUMMARY")
#     print("="*50)
#     print(f"Start time: {start_time}")
#     print(f"End time: {end_time}")
#     print(f"Total duration: {duration}")
#     print(f"Number of classes: {len(class_names)}")
#     print(f"Total synthetic images: {num_images}")
#     print(f"Final model saved at: {model.save_dir}")
#     print(f"Error analysis: {error_analysis}")
#     print("Training pipeline completed successfully!")

# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLOv11s Training Script - Bullseye Target Detection
Hardware: CPU Intel i7-14700HX, GPU RTX 4060 8GB
Software: Ubuntu 20.04, CUDA 12.8, Python 3.8.10
Author: Legion
Date: 2025-09-03
"""

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
        num_processes = min(cpu_count(), len(background_images))
    
    print(f"Using {num_processes} processes for Hough circle detection")
    
    bg_target_cache = {}
    
    # Prepare arguments for parallel processing
    args = [(bg_img, i) for i, bg_img in enumerate(background_images)]
    
    # Use multiprocessing Pool
    with Pool(processes=num_processes) as pool:
        results = pool.map(detect_target_center_hough_parallel_wrapper, args)
    
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
    
    # Albumentations augmentation pipeline
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=45, p=0.8),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.Blur(blur_limit=3, p=0.3),
        A.CLAHE(p=0.3),
        A.RandomGamma(p=0.3),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
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

def create_synthetic_dataset_parallel(augmented_path, background_images, output_path, target_scale=5.0, 
                                     batch_size=50, max_workers=6):
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
    print(f"Using {max_workers} workers and batch size {batch_size}")
    
    # Precompute target centers for background images (avoid repeated calculations)
    print("Precomputing target centers for background images...")
    bg_target_cache = {}
    for i, bg_img in enumerate(tqdm(background_images, desc="Preprocessing backgrounds")):
        target_center, confidence = detect_target_center_hough(bg_img)
        bg_target_cache[i] = (target_center, confidence)
    
    # Create images and annotations for each class
    image_counter = 0
    total_images = sum([len([f for f in os.listdir(os.path.join(augmented_path, c)) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) 
                       for c in classes])
    
    print(f"Total images to process: {total_images}")
    
    # Create progress bar
    pbar = tqdm(total=total_images, desc="Creating synthetic images")
    
    # Process each class
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(augmented_path, class_name)
        class_images = [f for f in os.listdir(class_path) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Processing class {class_name} with {len(class_images)} images")
        
        # Batch processing
        num_batches = math.ceil(len(class_images) / batch_size)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(class_images))
            batch_images = class_images[start_idx:end_idx]
            
            # Use thread pool to process current batch
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_image = {
                    executor.submit(
                        process_single_image, 
                        class_path, img_filename, class_idx, class_name, 
                        background_images, bg_target_cache, output_path, 
                        image_counter + i, target_scale
                    ): img_filename 
                    for i, img_filename in enumerate(batch_images)
                }
                
                # Process completed tasks
                for future in concurrent.futures.as_completed(future_to_image):
                    img_filename = future_to_image[future]
                    try:
                        result = future.result()
                        if result:
                            image_counter += 1
                            pbar.update(1)
                            
                            if image_counter % 100 == 0:
                                pbar.set_description(f"Created {image_counter} synthetic images")
                    except Exception as e:
                        print(f"Error processing {img_filename}: {e}")
    
    pbar.close()
    print(f"Completed synthetic dataset creation. Total images: {image_counter}")
    
    # Create classes.txt
    with open(os.path.join(output_path, "classes.txt"), 'w') as f:
        for class_name in classes:
            f.write(f"{class_name}\n")
    
    return classes, image_counter

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
        for c in range(3):
            bg_img[start_y:end_y, start_x:end_x, c] = \
                (1 - alpha) * bg_img[start_y:end_y, start_x:end_x, c] + \
                alpha * aug_img_resized[:, :, c]
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
    bg_target_cache = precompute_background_targets_parallel(background_images, num_processes=28)
    
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
            max_workers = min(cpu_count(), len(class_images))
        
        with Pool(processes=max_workers) as pool:
            results = pool.map(process_single_image_wrapper, args_list)
        
        # Count successful images
        successful_count = sum(results)
        end_time = time.time()
        print(f"Processed {successful_count}/{len(class_images)} images for class {class_name} in {end_time - start_time:.2f} seconds")
    
    print(f"Completed synthetic dataset creation. Total images: {image_counter}")
    
    # Create classes.txt
    with open(os.path.join(output_path, "classes.txt"), 'w') as f:
        for class_name in classes:
            f.write(f"{class_name}\n")
    
    return classes, image_counter

def process_single_image_wrapper(args):
    """
    Wrapper function for parallel image processing
    """
    try:
        return process_single_image(*args)
    except Exception as e:
        print(f"Error in process_single_image: {e}")
        return False

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

# def main():
#     """
#     Main function: Execute complete training pipeline
#     """
#     print("Starting YOLOv11s target detection training pipeline")
#     start_time = datetime.now()
    
#     # Set paths
#     base_path = "/home/legion/dataset"
#     raw_dataset_path = os.path.join(base_path, "raw_dataset")
#     background_path = os.path.join(base_path, "trash", "empty")
#     augmented_path = os.path.join(base_path, "augmented_dataset")
#     synthetic_path = os.path.join(base_path, "synthetic_dataset")
#     yolo_dataset_path = os.path.join(base_path, "yolo_dataset")
    
#     # 1. Set up environment and GPU warm-up
#     device = setup_environment()
    
#     # 2. Load background images
#     background_images = load_background_images(background_path)
    
#     # 3. Detect bullseye in background (optional, for verification)
#     if background_images:
#         sample_bg = background_images[0]
#         target_center, confidence = detect_target_center_hough(sample_bg)
#         if target_center:
#             x, y, r = target_center
#             print(f"Sample target detection - Center: ({x}, {y}), Radius: {r}, Confidence: {confidence:.2f}")
    
#     # 4. Data augmentation
#     augment_dataset(raw_dataset_path, augmented_path, target_per_class=900)
    
#     # 5. Create synthetic dataset (using parallel processing)
#     classes, num_images = create_synthetic_dataset_parallel(
#         augmented_path, 
#         background_images, 
#         synthetic_path,
#         target_scale=5.0,  # Scale up by 5x
#         batch_size=50,     # Process 50 images per batch
#         max_workers=6      # Use 6 threads
#     )
    
#     # 6. Set up YOLO dataset structure
#     dataset_yaml, class_names = setup_yolo_dataset_structure(synthetic_path, yolo_dataset_path)
    
#     # 7. Train YOLOv11s model
#     model, results = train_yolov11s(dataset_yaml)
    
#     # 8. Model optimization
#     optimized_model = optimize_model(model, dataset_yaml)
    
#     # 9. Result analysis
#     error_analysis = analyze_results(optimized_model, yolo_dataset_path)
    
#     # Output training summary
#     end_time = datetime.now()
#     duration = end_time - start_time
    
#     print("\n" + "="*50)
#     print("TRAINING SUMMARY")
#     print("="*50)
#     print(f"Start time: {start_time}")
#     print(f"End time: {end_time}")
#     print(f"Total duration: {duration}")
#     print(f"Number of classes: {len(class_names)}")
#     print(f"Total synthetic images: {num_images}")
#     print(f"Final model saved at: {model.save_dir}")
#     print(f"Error analysis: {error_analysis}")
#     print("Training pipeline completed successfully!")

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
    
    # 4. Data augmentation
    augment_dataset(raw_dataset_path, augmented_path, target_per_class=900)
    
    # 5. Create synthetic dataset (using parallel processing)
    classes, num_images = create_synthetic_dataset_parallel(
        augmented_path, 
        background_images, 
        synthetic_path,
        target_scale=5.0,  # Scale up by 5x
        batch_size=50,     # Process 50 images per batch
        max_workers=28     # Use 28 processes for CPU-intensive tasks
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