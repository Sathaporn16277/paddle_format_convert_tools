import os
import argparse
from yolo2paddle_det import *
from yolo2paddle_rec import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_folder_path', type=str, help='Path to YOLO annotation folder', required=True)
    parser.add_argument('--output_folder_path', type=str, help='Path to save the converted annotations', required=True)
    parser.add_argument('--image_w', type=int, help='Width of the images', default=640)
    parser.add_argument('--image_h', type=int, help='Height of the images', default=640)
    parser.add_argument('--folder_item', nargs='+', help='List of items in the folder', default=['train', 'valid', 'test'])
    
    args = parser.parse_args()
    for folder in args.folder_item:
        yolo_labels_path = os.path.join(args.yolo_folder_path, folder, 'labels')
        yolo_image_path = os.path.join(args.yolo_folder_path, folder, 'images')
        yolo_yaml_path = os.path.join(args.yolo_folder_path, 'data.yaml')

        output_det_image_path = os.path.join(args.output_folder_path, folder, 'det', 'images')
        output_det_labels_path = os.path.join(args.output_folder_path, folder, 'det', 'labels.txt')

        output_rec_images_path = os.path.join(args.output_folder_path, folder, 'rec', 'images')
        output_rec_labels_path = os.path.join(args.output_folder_path, folder, 'rec', 'labels.txt')
        
        convert_format_detection_dataset(yolo_labels_path, output_det_labels_path, yolo_image_path, output_det_image_path, args.image_w, args.image_h)
        convert_format_recognition_dataset(yolo_labels_path, yolo_image_path, yolo_yaml_path, output_rec_labels_path, output_rec_images_path, args.image_w, args.image_h)

if __name__ == "__main__":
    main()