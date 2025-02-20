import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json

def plot_polygons(image_path, boxes):
    img = cv2.imread(image_path)
    boxes = np.array(boxes, np.int32)
    boxes = boxes.reshape((-1, 1, 2))
    result = cv2.polylines(img, [boxes], isClosed=True, color=(0, 255, 0), thickness=2)
    img_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot polygons on image')
    parser.add_argument('--image_path', type=str, help='Path to image file')
    parser.add_argument('--boxes', type=str, help='List of polygon vertices in JSON format')
    args = parser.parse_args()

    # Parse boxes as JSON
    boxes = json.loads(args.boxes)

    plot_polygons(args.image_path, boxes)

if __name__ == '__main__':
    main()
