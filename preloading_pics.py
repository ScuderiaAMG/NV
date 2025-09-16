# from deepseek R1
import cv2
import numpy as np

# def extract_light_features(original_img, env_img):
#     # 计算差异图并提取光照特征
#     diff = cv2.absdiff(original_img, env_img)
#     # 估计全局颜色偏移（简化示例）
#     avg_diff = np.mean(diff, axis=(0,1))
#     return avg_diff  # 返回RGB偏移量作为特征
def extract_light_features(original_img, env_img):
    # 检查图像形状是否相同
    if original_img.shape != env_img.shape:
        # 调整env_img的大小以匹配original_img
        env_img = cv2.resize(env_img, (original_img.shape[1], original_img.shape[0]))
    diff = cv2.absdiff(original_img, env_img)
    avg_diff = np.mean(diff, axis=(0, 1))
    return avg_diff
def apply_to_other_image(target_img, light_features):
    # 应用光照特征到目标图片
    adjusted_img = target_img.copy()
    # 简单添加偏移（实际中需优化，如使用色彩空间转换）
    for i in range(3):  # 对RGB通道
        adjusted_img[:,:,i] = np.clip(adjusted_img[:,:,i] + light_features[i], 0, 255)
    return adjusted_img.astype(np.uint8)

# 示例用法：加载图片
original_A = cv2.imread('/home/legion/dataset/original_A.png')  # 原图A
env_A = cv2.imread('/home/legion/dataset/env_A.jpg')            # 环境图A
img_B = cv2.imread('/home/legion/dataset/raw_data/baby.png')                # 图片B
img_C = cv2.imread('/home/legion/dataset/raw_data/chimpanzee.png')                 # 图片C
img_D = cv2.imread('/home/legion/dataset/raw_data/cup.png')                 # 图片D
img_E = cv2.imread('/home/legion/dataset/raw_data/dolphin.png')                 # 图片E
img_F = cv2.imread('/home/legion/dataset/raw_data/elephant.png')                 # 图片F
img_G = cv2.imread('/home/legion/dataset/raw_data/flatfish.png')                 # 图片G
img_H = cv2.imread('/home/legion/dataset/raw_data/forest.png')                 # 图片H
img_I = cv2.imread('/home/legion/dataset/raw_data/girl.png')                 # 图片I
img_J = cv2.imread('/home/legion/dataset/raw_data/hamster.png')                 # 图片J
img_K = cv2.imread('/home/legion/dataset/raw_data/keyboard.png')                 # 图片K
img_L = cv2.imread('/home/legion/dataset/raw_data/lamp.png')                 # 图片L
img_M = cv2.imread('/home/legion/dataset/raw_data/lion.png')                 # 图片M
img_N = cv2.imread('/home/legion/dataset/raw_data/motorcycle.png')                 # 图片N
img_O = cv2.imread('/home/legion/dataset/raw_data/mushroom.png')                 # 图片O
img_P = cv2.imread('/home/legion/dataset/raw_data/oak_tree.png')                 # 图片P
img_Q = cv2.imread('/home/legion/dataset/raw_data/otter.png')                 # 图片Q
img_R = cv2.imread('/home/legion/dataset/raw_data/palm_tree.png')                 # 图片R
img_S = cv2.imread('/home/legion/dataset/raw_data/pickup_truck.png')                 # 图片S
img_T = cv2.imread('/home/legion/dataset/raw_data/plain.png')                 # 图片T
img_U = cv2.imread('/home/legion/dataset/raw_data/poppy.png')                 # 图片U
img_V = cv2.imread('/home/legion/dataset/raw_data/raccoon.png')                 # 图片V
img_W = cv2.imread('/home/legion/dataset/raw_data/seal.png')                 # 图片W
img_X = cv2.imread('/home/legion/dataset/raw_data/shrew.png')                 # 图片X
img_Y = cv2.imread('/home/legion/dataset/raw_data/snail.png')                 # 图片Y
img_Z = cv2.imread('/home/legion/dataset/raw_data/squirrel.png')                 # 图片Z
img_1 = cv2.imread('/home/legion/dataset/raw_data/sweet_pepper.png')                 # 图片1
img_2 = cv2.imread('/home/legion/dataset/raw_data/table.png')                 # 图片2
img_3 = cv2.imread('/home/legion/dataset/raw_data/tank.png')                 # 图片3
img_4 = cv2.imread('/home/legion/dataset/raw_data/whale.png')                 # 图片4
img_5 = cv2.imread('/home/legion/dataset/raw_data/wolf.png')                 # 图片5
img_6 = cv2.imread('/home/legion/dataset/raw_data/woman.png')                 # 图片6
img_7 = cv2.imread('/home/legion/dataset/raw_data/worm.png')                 # 图片7


# 提取特征并应用到B
light_features = extract_light_features(original_A, env_A)
simulated_baby = apply_to_other_image(img_B, light_features)
simulated_chimpanzee = apply_to_other_image(img_C, light_features)
simulated_cup = apply_to_other_image(img_D, light_features)
simulated_dolphin = apply_to_other_image(img_E, light_features)
simulated_elephant = apply_to_other_image(img_F, light_features)
simulated_flatfish = apply_to_other_image(img_G, light_features)
simulated_forest = apply_to_other_image(img_H, light_features)
simulated_girl = apply_to_other_image(img_I, light_features)
simulated_hamster = apply_to_other_image(img_J, light_features)
simulated_keyboard = apply_to_other_image(img_K, light_features)
simulated_lamp = apply_to_other_image(img_L, light_features)
simulated_lion = apply_to_other_image(img_M, light_features)
simulated_motorcycle = apply_to_other_image(img_N, light_features)
simulated_mushroom = apply_to_other_image(img_O, light_features)
simulated_oak_tree = apply_to_other_image(img_P, light_features)
simulated_otter = apply_to_other_image(img_Q, light_features)
simulated_palm_tree = apply_to_other_image(img_R, light_features)
simulated_pickup_truck = apply_to_other_image(img_S, light_features)
simulated_plain = apply_to_other_image(img_T, light_features)
simulated_poppy = apply_to_other_image(img_U, light_features)
simulated_raccoon = apply_to_other_image(img_V, light_features)
simulated_seal = apply_to_other_image(img_W, light_features)
simulated_shrew = apply_to_other_image(img_X, light_features)
simulated_snail = apply_to_other_image(img_Y, light_features)
simulated_squirrel = apply_to_other_image(img_Z, light_features)
simulated_sweet_pepper = apply_to_other_image(img_1, light_features)
simulated_table = apply_to_other_image(img_2, light_features)
simulated_tank = apply_to_other_image(img_3, light_features)
simulated_whale = apply_to_other_image(img_4, light_features)
simulated_wolf = apply_to_other_image(img_5, light_features)
simulated_woman = apply_to_other_image(img_6, light_features)
simulated_worm = apply_to_other_image(img_7, light_features)
cv2.imwrite('/home/legion/dataset/attempt_light/baby_aug.png', simulated_baby)  # 保存模拟环境下的B图片
cv2.imwrite('/home/legion/dataset/attempt_light/chimpanzee_aug.png', simulated_chimpanzee)  # 保存模拟环境下的C图片
cv2.imwrite('/home/legion/dataset/attempt_light/cup_aug.png', simulated_cup)  # 保存模拟环境下的D图片
cv2.imwrite('/home/legion/dataset/attempt_light/dolphin_aug.png', simulated_dolphin)  # 保存模拟环境下的E图片
cv2.imwrite('/home/legion/dataset/attempt_light/elephant_aug.png', simulated_elephant)  # 保存模拟环境下的F图片
cv2.imwrite('/home/legion/dataset/attempt_light/flatfish_aug.png', simulated_flatfish)  # 保存模拟环境下的G图片
cv2.imwrite('/home/legion/dataset/attempt_light/forest_aug.png', simulated_forest)  # 保存模拟环境下的H图片
cv2.imwrite('/home/legion/dataset/attempt_light/girl_aug.png', simulated_girl)  # 保存模拟环境下的I图片
cv2.imwrite('/home/legion/dataset/attempt_light/hamster_aug.png', simulated_hamster)  # 保存模拟环境下的J图片
cv2.imwrite('/home/legion/dataset/attempt_light/keyboard_aug.png', simulated_keyboard)  # 保存模拟环境下的K图片
cv2.imwrite('/home/legion/dataset/attempt_light/lamp_aug.png', simulated_lamp)  # 保存模拟环境下的L图片
cv2.imwrite('/home/legion/dataset/attempt_light/lion_aug.png', simulated_lion)  # 保存模拟环境下的M图片
cv2.imwrite('/home/legion/dataset/attempt_light/motorcycle_aug.png', simulated_motorcycle)  # 保存模拟环境下的N图片
cv2.imwrite('/home/legion/dataset/attempt_light/mushroom_aug.png', simulated_mushroom)  # 保存模拟环境下的O图片
cv2.imwrite('/home/legion/dataset/attempt_light/oak_tree_aug.png', simulated_oak_tree)  # 保存模拟环境下的P图片
cv2.imwrite('/home/legion/dataset/attempt_light/otter_aug.png', simulated_otter)  # 保存模拟环境下的Q图片
cv2.imwrite('/home/legion/dataset/attempt_light/palm_tree_aug.png', simulated_palm_tree)  # 保存模拟环境下的R图片
cv2.imwrite('/home/legion/dataset/attempt_light/pickup_truck_aug.png', simulated_pickup_truck)  # 保存模拟环境下的S图片
cv2.imwrite('/home/legion/dataset/attempt_light/plain_aug.png', simulated_plain)  # 保存模拟环境下的T图片
cv2.imwrite('/home/legion/dataset/attempt_light/poppy_aug.png', simulated_poppy)  # 保存模拟环境下的U图片
cv2.imwrite('/home/legion/dataset/attempt_light/raccoon_aug.png', simulated_raccoon)  # 保存模拟环境下的V图片
cv2.imwrite('/home/legion/dataset/attempt_light/seal_aug.png', simulated_seal)  # 保存模拟环境下的W图片
cv2.imwrite('/home/legion/dataset/attempt_light/shrew_aug.png', simulated_shrew)  # 保存模拟环境下的X图片
cv2.imwrite('/home/legion/dataset/attempt_light/snail_aug.png', simulated_snail)  # 保存模拟环境下的Y图片
cv2.imwrite('/home/legion/dataset/attempt_light/squirrel_aug.png', simulated_squirrel)  # 保存模拟环境下的Z图片
cv2.imwrite('/home/legion/dataset/attempt_light/sweet_pepper_aug.png', simulated_sweet_pepper)  # 保存模拟环境下的1图片
cv2.imwrite('/home/legion/dataset/attempt_light/table_aug.png', simulated_table)  # 保存模拟环境下的2图片
cv2.imwrite('/home/legion/dataset/attempt_light/tank_aug.png', simulated_tank)  # 保存模拟环境下的3图片
cv2.imwrite('/home/legion/dataset/attempt_light/whale_aug.png', simulated_whale)  # 保存模拟环境下的4图片
cv2.imwrite('/home/legion/dataset/attempt_light/wolf_aug.png', simulated_wolf)  # 保存模拟环境下的5图片
cv2.imwrite('/home/legion/dataset/attempt_light/woman_aug.png', simulated_woman)  # 保存模拟环境下的6图片
cv2.imwrite('/home/legion/dataset/attempt_light/worm_aug.png', simulated_worm)  # 保存模拟环境下的7图片
