import cv2
import torch
from ultralytics import YOLO
import time

class YOLOv8CameraDetector:
    def __init__(self, model_path, conf_threshold=0.5):
        """
        初始化YOLOv8检测器
        
        参数:
            model_path: 训练好的模型路径
            conf_threshold: 置信度阈值
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_names = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}")
    
    def detect_objects(self, frame):
        """
        对帧进行物体检测
        
        参数:
            frame: 输入图像帧
            
        返回:
            带有检测结果的图像帧
        """
        # 使用模型进行预测
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        # 绘制检测结果
        annotated_frame = results[0].plot()
        
        return annotated_frame
    
    def run_camera_detection(self, camera_index=0):
        """
        运行摄像头检测
        
        参数:
            camera_index: 摄像头索引 (默认0)
        """
        # 打开摄像头
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("无法打开摄像头")
            return
        
        print("按 'q' 键退出检测")
        
        # 用于计算FPS
        prev_time = 0
        
        try:
            while True:
                # 读取帧
                ret, frame = cap.read()
                if not ret:
                    print("无法获取帧")
                    break
                
                # 进行物体检测
                result_frame = self.detect_objects(frame)
                
                # 计算并显示FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time)
                prev_time = curr_time
                cv2.putText(result_frame, f'FPS: {int(fps)}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 显示结果
                cv2.imshow('YOLOv8 物体检测', result_frame)
                
                # 按'q'退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            # 释放资源
            cap.release()
            cv2.destroyAllWindows()

def main():
    # 模型路径 - 替换为你的实际路径
    model_path = "/home/legion/Documents/FirstRace/7*/best.pt"  # 或者使用你的best.pt路径
    
    # 创建检测器实例
    detector = YOLOv8CameraDetector(model_path, conf_threshold=0.5)
    
    # 运行摄像头检测
    # 如果默认摄像头索引0不工作，尝试其他索引(1,2等)
    detector.run_camera_detection(camera_index=0)

if __name__ == "__main__":
    main()