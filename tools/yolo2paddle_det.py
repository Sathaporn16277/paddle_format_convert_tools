import os
import json
import argparse
import shutil

def convert_format_detection_dataset(yolo_labels_path, output_labels_path, yolo_image_path, output_image_path, image_w, image_h):
    formatted_data = []
    os.makedirs(os.path.dirname(output_labels_path), exist_ok=True)
    os.makedirs(output_image_path, exist_ok=True)
    
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
            
            img_file_path = os.path.normpath(os.path.join(yolo_image_path, image_file_name)).replace("\\", "/")
            output_img_path = os.path.join(output_image_path, image_file_name)

            # Copy or move the image to the output directory
            shutil.copy(img_file_path, output_img_path)  # Change to shutil.move if you want to move instead
            
            try:
                with open(txt_file_path, 'r') as file:
                    annotations = []
                    
                    for line in file:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            print(f"Warning: Skipping invalid annotation in {filename}")
                            continue
                        class_id, x_center, y_center, width, height = map(float, parts)
                        x_center *= image_w
                        y_center *= image_h
                        width *= image_w
                        height *= image_h
                        
                        x_min = x_center - (width / 2)
                        y_min = y_center - (height / 2)
                        x_max = x_center + (width / 2)
                        y_max = y_center + (height / 2)
                        
                        transcription = str(int(class_id))
                        
                        formatted_points = [
                            [x_min, y_min],
                            [x_max, y_min],
                            [x_max, y_max],
                            [x_min, y_max]
                        ]

                        annotations.append({
                            "transcription": transcription,
                            "points": formatted_points
                        })
                    
                    formatted_annotation = json.dumps(annotations, ensure_ascii=False)
                    formatted_data.append(f"{output_img_path}\t{formatted_annotation}")
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
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
        output_image_path = os.path.join(args.output_folder_path, folder, 'det', 'images')
        output_labels_path = os.path.join(args.output_folder_path, folder, 'det', 'labels.txt')
        
       
        
        convert_format_detection_dataset(yolo_labels_path, output_labels_path, yolo_image_path, output_image_path, args.image_w, args.image_h)

if __name__ == "__main__":
    main()
