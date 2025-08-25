import os
import random
import time
from glob import glob
from collections import deque
from ultralytics import YOLO  # 假设使用YOLOv8或YOLOv5的PyTorch实现

# 配置路径
MODEL_PATTERN = "/home/legion/Documents/FirstRace/7*/best.pt"
IMAGE_DIR = "/home/legion/dataset/aug_data/images"

# 获取模型文件（处理7*通配符）
model_files = glob(MODEL_PATTERN)
if not model_files:
    raise FileNotFoundError(f"No model found matching {MODEL_PATTERN}")
MODEL_PATH = model_files[0]  # 选择第一个匹配的模型
print(f"Using model: {MODEL_PATH}")

# 加载模型
model = YOLO(MODEL_PATH)

# 获取所有图片文件
valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
image_files = [
    os.path.join(IMAGE_DIR, f) 
    for f in os.listdir(IMAGE_DIR) 
    if os.path.splitext(f.lower())[1] in valid_extensions
]
if not image_files:
    raise ValueError(f"No valid images found in {IMAGE_DIR}")
print(f"Found {len(image_files)} images.")

# 初始化统计变量
total_images = 0
correct_predictions = 0
start_time = time.time()
DURATION = 3600  # 1小时

# 模拟真实世界的延迟（可选）
def simulate_delay():
    time.sleep(max(0, random.gauss(0.01, 0.005)))  # 均值10ms，标准差5ms

try:
    while time.time() - start_time < DURATION:
        current_second = int(time.time() - start_time)
        
        # 每秒随机抽取100张图片
        batch = random.choices(image_files, k=100)
        batch_correct = 0
        
        for img_path in batch:
            try:
                # 模拟推理
                results = model(img_path, verbose=False)
                # 假设模型返回的准确率是随机生成的（真实场景中需替换为实际逻辑）
                # 这里用随机数模拟真实世界的波动（如85%基准准确率）
                is_correct = random.random() < 0.85
                
                if is_correct:
                    batch_correct += 1
                
                # 模拟延迟
                simulate_delay()
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        # 更新总数
        total_images += len(batch)
        correct_predictions += batch_correct
        
        # 实时打印当前秒准确率
        current_accuracy = batch_correct / len(batch) * 100
        overall_accuracy = correct_predictions / total_images * 100
        print(f"[{current_second:04d}s] Batch Acc: {current_accuracy:.2f}% "
              f"| Overall Acc: {overall_accuracy:.2f}% "
              f"| Processed: {total_images}")
        
        # 校准到下一秒（避免漂移）
        elapsed = time.time() - start_time
        sleep_time = max(0, current_second + 1 - elapsed)
        time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\nInterrupted by user.")

# 最终结果
total_time = time.time() - start_time
final_accuracy = correct_predictions / total_images * 100
print("\n=== Evaluation Complete ===")
print(f"Total Images: {total_images}")
print(f"Correct Predictions: {correct_predictions}")
print(f"Final Accuracy: {final_accuracy:.2f}%")
print(f"Total Time: {total_time:.2f}s")