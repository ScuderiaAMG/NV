import os
import random
import time
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from collections import deque
import threading

MODEL_PATH = "/home/legion/Documents/FirstRace/7*/best.pt"  
IMAGE_DIR = "/home/legion/dataset/aug_data/images"  
VAL_LABEL_DIR = "/home/legion/dataset/aug_data/val/labels"  
OUTPUT_DIR = "/home/legion/dataset/validation_results" 

SAMPLE_RATE = 100 
TOTAL_SECONDS = 3600  
IOU_THRESHOLD = 0.5  
CONFIDENCE_THRESHOLD = 0.25 

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model():
    import glob
    model_paths = glob.glob(MODEL_PATH)
    if not model_paths:
        raise FileNotFoundError(f"No model found matching pattern: {MODEL_PATH}")
    
    latest_model = max(model_paths, key=os.path.getmtime)
    print(f"Loading model: {latest_model}")
    
    model = YOLO(latest_model)
    return model

def get_image_label_pairs():
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_paths = []
    label_paths = []
    
    for root, _, files in os.walk(IMAGE_DIR):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))
                
                rel_path = os.path.relpath(root, IMAGE_DIR)
                label_file = os.path.splitext(file)[0] + '.txt'
                label_path = os.path.join(VAL_LABEL_DIR, rel_path, label_file)
                
                if not os.path.exists(label_path):
                    label_path = os.path.join(root.replace('images', 'labels'), 
                                            os.path.splitext(file)[0] + '.txt')
                
                label_paths.append(label_path)
    
    return list(zip(image_paths, label_paths))

def parse_labels(label_path):
    if not os.path.exists(label_path):
        return []
    
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])
                labels.append({
                    'class_id': class_id,
                    'bbox': [x_center, y_center, width, height]
                })
    return labels

def calculate_iou(box1, box2):
    def xywh_to_xyxy(box):
        x_center, y_center, width, height = box
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        return [x1, y1, x2, y2]
    
    box1 = xywh_to_xyxy(box1)
    box2 = xywh_to_xyxy(box2)
    
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    iou = intersection / union if union > 0 else 0
    return iou

def evaluate_detection(gt_labels, pred_results, iou_threshold=0.5):
    if not gt_labels and not pred_results:
        return 1.0, 0, 0, 0  
    
    if not gt_labels or not pred_results:
        return 0.0, 0, len(gt_labels) if gt_labels else 0, len(pred_results) if pred_results else 0
    
    matched_gt = set()
    matched_pred = set()
    correct = 0
    
    for i, pred in enumerate(pred_results):
        for j, gt in enumerate(gt_labels):
            if j in matched_gt:
                continue
                
            if pred['class_id'] == gt['class_id']:
                iou = calculate_iou(pred['bbox'], gt['bbox'])
                if iou >= iou_threshold:
                    correct += 1
                    matched_gt.add(j)
                    matched_pred.add(i)
                    break
    
    precision = correct / len(pred_results) if pred_results else 0
    recall = correct / len(gt_labels) if gt_labels else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1, correct, len(gt_labels), len(pred_results)

class ValidationWorker:
    def __init__(self, model, image_label_pairs):
        self.model = model
        self.image_label_pairs = image_label_pairs
        self.results = {
            'total_samples': 0,
            'total_correct': 0,
            'total_gt_objects': 0,
            'total_pred_objects': 0,
            'f1_scores': [],
            'precision_scores': [],
            'recall_scores': []
        }
        self.lock = threading.Lock()
        self.running = True
    
    def process_batch(self, batch):
        batch_results = {
            'samples': len(batch),
            'correct': 0,
            'gt_objects': 0,
            'pred_objects': 0,
            'f1_scores': []
        }
        
        for img_path, label_path in batch:
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            gt_labels = parse_labels(label_path)
            
            with torch.no_grad():
                results = self.model(img, conf=CONFIDENCE_THRESHOLD, verbose=False)
            
            pred_results = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls.item())
                        conf = box.conf.item()
                        bbox = box.xywhn[0].cpu().numpy() if box.xywhn[0].is_cuda else box.xywhn[0].numpy()
                        pred_results.append({
                            'class_id': class_id,
                            'confidence': conf,
                            'bbox': bbox.tolist()
                        })
            
            f1, correct, gt_count, pred_count = evaluate_detection(gt_labels, pred_results, IOU_THRESHOLD)
            
            batch_results['correct'] += correct
            batch_results['gt_objects'] += gt_count
            batch_results['pred_objects'] += pred_count
            batch_results['f1_scores'].append(f1)
        
        if batch_results['samples'] > 0:
            batch_precision = batch_results['correct'] / batch_results['pred_objects'] if batch_results['pred_objects'] > 0 else 0
            batch_recall = batch_results['correct'] / batch_results['gt_objects'] if batch_results['gt_objects'] > 0 else 0
            batch_f1 = 2 * batch_precision * batch_recall / (batch_precision + batch_recall) if (batch_precision + batch_recall) > 0 else 0
            
            with self.lock:
                self.results['total_samples'] += batch_results['samples']
                self.results['total_correct'] += batch_results['correct']
                self.results['total_gt_objects'] += batch_results['gt_objects']
                self.results['total_pred_objects'] += batch_results['pred_objects']
                self.results['f1_scores'].extend(batch_results['f1_scores'])
                self.results['precision_scores'].append(batch_precision)
                self.results['recall_scores'].append(batch_recall)
            
            return batch_f1, batch_precision, batch_recall, batch_results['samples']
        
        return 0, 0, 0, 0
    
    def run_validation(self):
        start_time = time.time()
        last_second = 0
        
        print("Starting validation...")
        print("Time (s) | Samples | F1 Score | Precision | Recall | Samples/s")
        print("-" * 65)
        
        while self.running and (time.time() - start_time) < TOTAL_SECONDS:
            current_second = int(time.time() - start_time)
            
            if current_second > last_second:
                last_second = current_second
                
                batch = random.sample(self.image_label_pairs, min(SAMPLE_RATE, len(self.image_label_pairs)))
                
                f1, precision, recall, samples = self.process_batch(batch)
                
                print(f"{current_second:8d} | {samples:7d} | {f1:8.4f} | {precision:9.4f} | {recall:6.4f} | {samples:9d}")
        
        total_time = time.time() - start_time
        total_samples = self.results['total_samples']
        
        if total_samples > 0:
            avg_f1 = np.mean(self.results['f1_scores']) if self.results['f1_scores'] else 0
            avg_precision = np.mean(self.results['precision_scores']) if self.results['precision_scores'] else 0
            avg_recall = np.mean(self.results['recall_scores']) if self.results['recall_scores'] else 0
            
            print("\n" + "="*65)
            print("FINAL RESULTS:")
            print(f"Total time: {total_time:.2f} seconds")
            print(f"Total samples: {total_samples}")
            print(f"Samples per second: {total_samples / total_time:.2f}")
            print(f"Average F1 Score: {avg_f1:.4f}")
            print(f"Average Precision: {avg_precision:.4f}")
            print(f"Average Recall: {avg_recall:.4f}")
            print(f"Total GT objects: {self.results['total_gt_objects']}")
            print(f"Total predicted objects: {self.results['total_pred_objects']}")
            print(f"Total correct detections: {self.results['total_correct']}")
            
            results_file = os.path.join(OUTPUT_DIR, f"validation_results_{int(time.time())}.txt")
            with open(results_file, 'w') as f:
                f.write("FINAL VALIDATION RESULTS\n")
                f.write("="*50 + "\n")
                f.write(f"Total time: {total_time:.2f} seconds\n")
                f.write(f"Total samples: {total_samples}\n")
                f.write(f"Samples per second: {total_samples / total_time:.2f}\n")
                f.write(f"Average F1 Score: {avg_f1:.4f}\n")
                f.write(f"Average Precision: {avg_precision:.4f}\n")
                f.write(f"Average Recall: {avg_recall:.4f}\n")
                f.write(f"Total GT objects: {self.results['total_gt_objects']}\n")
                f.write(f"Total predicted objects: {self.results['total_pred_objects']}\n")
                f.write(f"Total correct detections: {self.results['total_correct']}\n")
            
            print(f"\nResults saved to: {results_file}")
        else:
            print("No samples were processed.")

def main():
    print("Loading model...")
    model = load_model()
    
    print("Loading image-label pairs...")
    image_label_pairs = get_image_label_pairs()
    
    if not image_label_pairs:
        print("No image-label pairs found!")
        return
    
    print(f"Found {len(image_label_pairs)} image-label pairs")
    
    worker = ValidationWorker(model, image_label_pairs)
    
    try:
        worker.run_validation()
    except KeyboardInterrupt:
        print("\nValidation interrupted by user")
        worker.running = False
    except Exception as e:
        print(f"Validation error: {e}")
        worker.running = False

if __name__ == "__main__":
    main()