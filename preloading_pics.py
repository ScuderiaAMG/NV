# from deepseek R1
import cv2
import numpy as np

def extract_light_features(original_img, env_img):
    # 计算差异图并提取光照特征
    diff = cv2.absdiff(original_img, env_img)
    # 估计全局颜色偏移（简化示例）
    avg_diff = np.mean(diff, axis=(0,1))
    return avg_diff  # 返回RGB偏移量作为特征

def apply_to_other_image(target_img, light_features):
    # 应用光照特征到目标图片
    adjusted_img = target_img.copy()
    # 简单添加偏移（实际中需优化，如使用色彩空间转换）
    for i in range(3):  # 对RGB通道
        adjusted_img[:,:,i] = np.clip(adjusted_img[:,:,i] + light_features[i], 0, 255)
    return adjusted_img.astype(np.uint8)

# 示例用法：加载图片
original_A = cv2.imread('original_A.jpg')  # 原图A
env_A = cv2.imread('env_A.jpg')            # 环境图A
img_B = cv2.imread('B.jpg')                # 图片B

# 提取特征并应用到B
light_features = extract_light_features(original_A, env_A)
simulated_B = apply_to_other_image(img_B, light_features)
cv2.imwrite('simulated_B_env.jpg', simulated_B)  # 保存模拟环境下的B图片

########################################################################################################################333
# From WENXIN X1 Turbo
import cv2
import numpy as np
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import img_to_array
# 1. 提取环境光特征（以颜色直方图为例）
def extract_light_features(img):
    hist = cv2.calcHist([img], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
    cv2.normalize(hist, hist)
    return hist.flatten()
# 2. 应用环境光到目标图片（简单颜色映射）
def apply_light_transfer(src_img, target_img, light_features):
    # 假设light_features为颜色映射表
    # 此处需根据具体特征设计映射逻辑
    return cv2.LUT(src_img, light_features)
# 示例使用
original_A = cv2.imread("original_A.jpg")
env_A = cv2.imread("env_A.jpg")
light_features = extract_light_features(env_A)
# 对图片B应用环境光
img_B = cv2.imread("image_B.jpg")
simulated_B = apply_light_transfer(img_B, original_A, light_features)

