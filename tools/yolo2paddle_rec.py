import os
import cv2
import yaml
import json
import shutil
import argparse

def load_class_labels(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']

def convert_format_recognition_dataset(yolo_labels_path, yolo_image_path, yolo_yaml_path, output_labels_path, output_images_path, image_w, image_h):
    class_id_to_label = load_class_labels(yolo_yaml_path)
    
    os.makedirs(os.path.dirname(output_labels_path), exist_ok=True)
    os.makedirs(output_images_path, exist_ok=True)

    formatted_data = []

    for filename in os.listdir(yolo_labels_path):
        if filename.endswith('.txt'):
            txt_file_path = os.path.join(yolo_labels_path, filename)
            
            image_file_name_jpg = os.path.splitext(filename)[0] + '.jpg'
            image_file_name_png = os.path.splitext(filename)[0] + '.png'

            if os.path.exists(os.path.join(yolo_image_path, image_file_name_jpg)):
                image_file_name = image_file_name_jpg
            elif os.path.exists(os.path.join(yolo_image_path, image_file_name_png)):
                image_file_name = image_file_name_png
            else:
                print(f"Warning: No image found for {filename}")
                continue

            image_path = os.path.join(yolo_image_path, image_file_name)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Warning: Could not read image {image_path}")
                continue
            
            try:
                with open(txt_file_path, 'r') as f:
                    annotations = []
                    
                    for idx, line in enumerate(f.readlines()):
                        parts = line.strip().split()
                        if len(parts) != 5:
                            print(f"Warning: Skipping invalid annotation in {filename}")
                            continue
                            
                        class_id, x_center, y_center, width, height = map(float, parts)
                        
                        x_center *= image_w
                        y_center *= image_h
                        width *= image_w
                        height *= image_h
                        
                        x_min = int(max(0, x_center - (width / 2)))
                        y_min = int(max(0, y_center - (height / 2)))
                        x_max = int(min(image_w, x_center + (width / 2)))
                        y_max = int(min(image_h, y_center + (height / 2)))

                        label = class_id_to_label[int(class_id)] if int(class_id) < len(class_id_to_label) else str(int(class_id))

                        try:
                            cropped_image = image[y_min:y_max, x_min:x_max]
                            if cropped_image.size == 0:
                                print(f"Warning: Empty crop for {filename}, bbox: {x_min},{y_min},{x_max},{y_max}")
                                continue
                                
                            # Save the cropped image
                            cropped_image_name = f"{os.path.splitext(filename)[0]}_{idx}.jpg"
                            cropped_image_path = os.path.join(output_images_path, cropped_image_name)
                            cropped_image_path = os.path.normpath(cropped_image_path).replace("\\", "/")
                            
                            cv2.imwrite(cropped_image_path, cropped_image)
                            
                            # Add to formatted data with the simple format: path "label"
                            formatted_data.append(f"{cropped_image_path}\t\"{label}\"")
                            
                        except Exception as e:
                            print(f"Error cropping image {filename}, bbox: {x_min},{y_min},{x_max},{y_max}: {e}")
                            
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    # Write output file
    try:
        with open(output_labels_path, 'w', encoding='utf-8') as output:
            for line in formatted_data:
                output.write(line + '\n')
        print(f"Conversion complete. Output saved to {output_labels_path}")
    except Exception as e:
        print(f"Error writing output file: {e}")

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
        output_labels_path = os.path.join(args.output_folder_path, folder, 'rec', 'labels.txt')
        output_images_path = os.path.join(args.output_folder_path, folder, 'rec', 'images')

        convert_format_recognition_dataset(yolo_labels_path, yolo_image_path, yolo_yaml_path, output_labels_path, output_images_path, args.image_w, args.image_h)

if __name__ == "__main__":
    main()