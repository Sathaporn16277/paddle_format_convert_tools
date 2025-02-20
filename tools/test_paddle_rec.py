import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json

def show_image(image_path):
    image = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot polygons on image')
    parser.add_argument('--image_path', type=str, help='Path to image file')
    args = parser.parse_args()

    show_image(args.image_path)

if __name__ == '__main__':
    main()
