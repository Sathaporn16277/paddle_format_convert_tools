import os
import cv2
import json
import math
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_json(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        labelme_data = json.load(file)
    return labelme_data

def corner2poly(image_path, resize_shape, original_boxes):
    original_w, original_h = Image.open(image_path).size
    resize_w, resize_h = resize_shape
    original_x1, original_y1, original_x2, original_y2 = original_boxes

    width_ratio = resize_w / original_w
    height_ratio = resize_h / original_h

    scaled_x1 = math.floor(original_x1 * width_ratio)
    scaled_y1 = math.floor(original_y1 * height_ratio)
    scaled_x2 = math.ceil(original_x2 * width_ratio)
    scaled_y2 = math.ceil(original_y2 * height_ratio)

    return [[scaled_x1, scaled_y1], [scaled_x1, scaled_y2], [scaled_x2, scaled_y2], [scaled_x2, scaled_y1]]

labelme_folder = r'dataset\labelme_pin_code\test'
resize_w, resize_h = 640, 640

labelme_image_folder = os.path.normpath(os.path.join(labelme_folder, 'images')).replace("\\", "/")
labelme_labels_folder = os.path.normpath(os.path.join(labelme_folder, 'labels')).replace("\\", "/")

for filename in os.listdir(labelme_labels_folder):
    if filename.endswith('.json'):
        json_file_path = os.path.normpath(os.path.join(labelme_labels_folder, filename)).replace("\\", "/")
        try:
            labelme_data = load_json(json_file_path)
            labelme_image_file_path = os.path.normpath(os.path.join(labelme_image_folder, labelme_data['imagePath'])).replace("\\", "/")
            for shape in labelme_data['shapes']:
                original_x1, original_y1 = shape['points'][0]
                original_x2, original_y2 = shape['points'][1]
                original_boxes = [original_x1, original_y1, original_x2, original_y2]
                boxes = corner2poly(labelme_image_file_path, (resize_w, resize_h), original_boxes)
                poly_boxes = np.array(boxes, dtype=np.int32).reshape((-1, 1, 2))
                image = cv2.resize(cv2.imread(labelme_image_file_path), (resize_w, resize_h))
                cv2.polylines(image, [poly_boxes], isClosed=True, color=(0, 255, 0), thickness=2)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                plt.imshow(image)
                plt.title(f'Boxes: {boxes}\nPoly Boxes: {poly_boxes.tolist}')
                plt.axis('off')
                plt.show()
        except Exception as e:
            print(f"Error processing {filename}: {e}")