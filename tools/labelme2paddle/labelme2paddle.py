import os
import sys
import argparse

# Add the directory where yolo2paddle_det.py and yolo2paddle_rec.py are located to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
labelme2paddle_dir = os.path.join(script_dir, 'path_to_labelme2paddle_folder')  # Adjust this path as needed
sys.path.append(labelme2paddle_dir)

# Now you can import the modules
from labelme2paddle_det import *
from labelme2paddle_rec import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labelme_folder_path', type=str, help='Path to Labelme annotation folder', required=True)
    parser.add_argument('--output_folder_path', type=str, help='Path to save the converted annotations', required=True)
    parser.add_argument('--image_w', type=int, help='Width of the images', default=640)
    parser.add_argument('--image_h', type=int, help='Height of the images', default=640)
    parser.add_argument('--folder_item', nargs='+', help='List of items in the folder', default=['train', 'valid', 'test'])
    args = parser.parse_args()

    for folder in args.folder_item:
        
        labelme_folder = os.path.normpath(os.path.join(args.labelme_folder_path, folder)).replace("\\", "/")
        detection_output_folder = os.path.normpath(os.path.join(args.output_folder_path, folder, 'det')).replace("\\", "/")
        recognition_output_folder = os.path.normpath(os.path.join(args.output_folder_path, folder, 'rec')).replace("\\", "/")
    
        convert_format_detection_dataset(labelme_folder=labelme_folder, output_folder=detection_output_folder, resize_w=args.image_w, resize_h=args.image_h)
        convert_format_recognition_dataset(labelme_folder=labelme_folder, output_folder=recognition_output_folder)

if __name__ == "__main__":
    main()