# import os, xml.etree.ElementTree as ET
# from collections import Counter

# bad = []
# for xml_path in [...]:
#     tree = ET.parse(xml_path)
#     root = tree.getroot()
#     w = float(root.find('.//width').text)
#     h = float(root.find('.//height').text)
#     for obj in root.findall('.//object'):
#         bnd = obj.find('bndbox')
#         xmin, xmax = float(bnd.find('xmin').text), float(bnd.find('xmax').text)
#         ymin, ymax = float(bnd.find('ymin').text), float(bnd.find('ymax').text)
#         if w <= 0 or h <= 0 or xmax <= xmin or ymax <= ymin:
#             bad.append(xml_path)
# print("Bad XML:", bad[:10])
import cv2, os
YOLO_DATA_DIR = "/home/legion/dataset/yolo_dataset"
TRAIN_DIR = os.path.join(YOLO_DATA_DIR, "train")
bad_img = []
for f in os.listdir(TRAIN_DIR+"/images")[:100]:
    img = cv2.imread(os.path.join(TRAIN_DIR,"images",f))
    if img is None or img.size==0:
        bad_img.append(f)
print("Bad images:", bad_img)