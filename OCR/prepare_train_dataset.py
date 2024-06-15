import cv2
import pandas as pd
import pybboxes as pbx
import os

def process_image(image_path, bbox_path, output_dir) :
    # Read image 
    image = cv2.imread(image_path)
    # Grayscale 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8,8))
    morph = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
    # Divide
    divide = cv2.divide(gray, morph, scale = 255)
    # Thresholding
    binary_image = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # Noise Removal
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    denoised_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations = 1)

    # Read Bounding Box
    bounding_box = pd.read_csv(bbox_path, header = None, delimiter = " ")
    bounding_box.columns = ['class', 'x_center', 'y_center', 'width', 'height']
    bounding_box.drop(['class'], axis = 1, inplace = True)

    # Get image dimensions
    H, W = denoised_image.shape

    # Function to convert YOLO bbox to VOC format
    def yolo_to_voc(row):
        yolo_bbox = (row['x_center'], row['y_center'], row['width'], row['height'])
        voc_bbox = pbx.convert_bbox(yolo_bbox, from_type='yolo', to_type='voc', image_size=(W, H))
        return voc_bbox
    
    bounding_box['normal'] = bounding_box.apply(yolo_to_voc, axis = 1)

    # Save each bounding box region as a separate image
    for index, bbox in enumerate(bounding_box['normal']):
        x_min, y_min, x_max, y_max = bbox
        # Extract each bounding box
        bbox_image = binary_image[y_min:y_max, x_min:x_max]
        # Write filename
        filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_word{index + 1}.jpg"
        # Save image
        cv2.imwrite(os.path.join(output_dir, filename), bbox_image)


def create_crnn_train_data() :
    input_images_dir = 'Dataset/Train'
    input_bboxes_dir = 'Dataset/Train Annotate'
    output_dir = 'CRNN/Train Data'

    os.makedirs(output_dir, exist_ok = True)

    for image_filename in os.listdir(input_images_dir):
        if image_filename.endswith(('.jpg', '.png')):
            image_path = os.path.join(input_images_dir, image_filename)
            bbox_filename = os.path.splitext(image_filename)[0] + ".txt"
            bbox_path = os.path.join(input_bboxes_dir, bbox_filename)
            if os.path.exists(bbox_path):
                process_image(image_path, bbox_path, output_dir)
            else:
                print(f"Bounding box file not found for {image_filename}")


if __name__ == "__main__" :
    create_crnn_train_data()








