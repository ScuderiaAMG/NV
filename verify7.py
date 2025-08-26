import cv2
import torch
from ultralytics import YOLO
import time
import os

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
            检测结果和带有检测结果的图像帧
        """
        # 使用模型进行预测
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        # 绘制检测结果
        annotated_frame = results[0].plot()
        
        return results, annotated_frame
    
    def process_and_save_results(self, frame, frame_count, output_dir="detection_results"):
        """
        处理帧并保存结果（无GUI显示）
        
        参数:
            frame: 输入图像帧
            frame_count: 帧计数
            output_dir: 输出目录
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 进行物体检测
        results, annotated_frame = self.detect_objects(frame)
        
        # 保存带注释的帧
        output_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(output_path, annotated_frame)
        
        # 保存检测结果到文本文件
        txt_output_path = os.path.join(output_dir, f"frame_{frame_count:06d}.txt")
        with open(txt_output_path, 'w') as f:
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = self.class_names[class_id]
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].tolist()
                    f.write(f"{class_name} {confidence:.2f} {bbox}\n")
        
        print(f"已保存帧 {frame_count} 的结果")
    
    def run_camera_detection_headless(self, camera_index=0, max_frames=100, output_dir="detection_results"):
        """
        无GUI模式下运行摄像头检测
        
        参数:
            camera_index: 摄像头索引 (默认0)
            max_frames: 最大处理帧数
            output_dir: 输出目录
        """
        # 打开摄像头
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("无法打开摄像头")
            return
        
        print(f"开始处理，最多处理 {max_frames} 帧")
        print("按 Ctrl+C 停止处理")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while frame_count < max_frames:
                # 读取帧
                ret, frame = cap.read()
                if not ret:
                    print("无法获取帧")
                    break
                
                # 处理并保存结果
                self.process_and_save_results(frame, frame_count, output_dir)
                frame_count += 1
                
                # 计算并显示进度
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                print(f"已处理 {frame_count}/{max_frames} 帧, FPS: {fps:.2f}")
                
                # 短暂休眠以减少CPU使用率
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("用户中断处理")
        finally:
            # 释放资源
            cap.release()
            total_time = time.time() - start_time
            print(f"处理完成，共处理 {frame_count} 帧，总用时 {total_time:.2f} 秒，平均 FPS: {frame_count/total_time:.2f}")
    
    def run_camera_detection_with_gui(self, camera_index=0):
        """
        带GUI显示的摄像头检测（如果GUI可用）
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
                _, result_frame = self.detect_objects(frame)
                
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
    
    # 尝试使用GUI模式，如果失败则使用无头模式
    try:
        # 测试GUI功能
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imshow('Test', test_frame)
        cv2.destroyWindow('Test')
        
        # 如果测试成功，使用GUI模式
        print("GUI模式可用，使用GUI显示")
        detector.run_camera_detection_with_gui(camera_index=0)
    except:
        print("GUI模式不可用，使用无头模式")
        detector.run_camera_detection_headless(
            camera_index=0, 
            max_frames=100,  # 处理100帧后停止
            output_dir="detection_results"
        )

if __name__ == "__main__":
    main()