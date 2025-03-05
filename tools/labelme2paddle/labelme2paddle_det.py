import os
import json
import math
import argparse
from PIL import Image

def load_json(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        labelme_data = json.load(file)
    return labelme_data

def corner2poly(image_path, resize_shape, original_boxes):
    original_w, original_h = Image.open(image_path).size
    resize_w, resize_h = resize_shape
    original_x1, original_y1, original_x2, original_y2 = original_boxes

    # Calculate scaling factors
    width_ratio = resize_w / original_w
    height_ratio = resize_h / original_h

    # Scale the original box coordinates based on the ratio
    scaled_x1 = math.floor(original_x1 * width_ratio)
    scaled_y1 = math.floor(original_y1 * height_ratio)
    scaled_x2 = math.ceil(original_x2 * width_ratio)
    scaled_y2 = math.ceil(original_y2 * height_ratio)

    # Return the scaled bounding box in the form of polygon points
    return [[scaled_x1, scaled_y1], [scaled_x1, scaled_y2], [scaled_x2, scaled_y2], [scaled_x2, scaled_y1]]


def convert_format_detection_dataset(labelme_folder, output_folder, resize_w, resize_h):
    formatted_data = []
    
    labelme_image_folder = os.path.normpath(os.path.join(labelme_folder, 'images')).replace("\\", "/")
    labelme_labels_folder = os.path.normpath(os.path.join(labelme_folder, 'labels')).replace("\\", "/")

    output_images_folder = os.path.normpath(os.path.join(output_folder, 'images')).replace("\\", "/")
    output_labels_file = os.path.normpath(os.path.join(output_folder, 'labels.txt')).replace("\\", "/")

    os.makedirs(output_images_folder, exist_ok=True)
    os.makedirs(os.path.dirname(output_labels_file), exist_ok=True)
 
    for filename in os.listdir(labelme_labels_folder):
        if filename.endswith('.json'):
            json_file_path = os.path.join(labelme_labels_folder, filename)            
            try:
                labelme_data = load_json(json_file_path)
                labelme_image_file_path = os.path.normpath(os.path.join(labelme_image_folder, labelme_data['imagePath'])).replace("\\", "/")
                output_image_file_path = os.path.normpath(os.path.join(output_images_folder, labelme_data['imagePath'])).replace("\\", "/")
                Image.open(labelme_image_file_path).resize((resize_w, resize_h)).save(output_image_file_path)
                annotations = []

                for shape in labelme_data['shapes']:
                    original_x1, original_y1 = shape['points'][0]
                    original_x2, original_y2 = shape['points'][1]
                    original_boxes = [original_x1, original_y1, original_x2, original_y2]
                    boxes = corner2poly(labelme_image_file_path, (resize_w, resize_h), original_boxes)
                    annotation = {
                        "transcription": shape['label'],
                        "points": boxes
                    }

                    annotations.append(annotation)
                formatted_annotation = json.dumps(annotations, ensure_ascii=False)
                formatted_data.append(f"{output_image_file_path}\t{formatted_annotation}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
            
        try:
            with open(output_labels_file, 'w', encoding='utf-8') as output:
                for line in formatted_data:
                    output.write(line + '\n')
                print(f"Conversion complete. Output saved to {output_labels_file}")
        except Exception as e:
            print(f"Error writing output file: {e}")

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
        output_folder = os.path.normpath(os.path.join(args.output_folder_path, folder, 'det')).replace("\\", "/")
        convert_format_detection_dataset(labelme_folder=labelme_folder, output_folder=output_folder, resize_w=args.image_w, resize_h=args.image_h)

if __name__ == "__main__":
    main()