# 检查训练/验证集重叠
import os
import numpy as np
train_files = set(os.listdir("dataset/train/images"))
val_files = set(os.listdir("dataset/val/images"))
print("数据泄露检查:", len(train_files & val_files))