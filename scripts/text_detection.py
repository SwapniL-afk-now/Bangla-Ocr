import cv2
import os
from ultralytics import YOLO

# Load the trained model
model = YOLO("path to model")

# Define output directory for cropped regions
output_dir = "path to output directory"
os.makedirs(output_dir, exist_ok=True)
    # Load a test image
img_folder_path = "path to directory"
for img_name in os.listdir(img_folder_path):
    img_path = os.path.join(img_folder_path, img_name)
    img = cv2.imread(img_path)

    # Get predictions
    results = model.predict(source=img_path, conf=0.5, save=False)

    # Filter and draw only class 0 boxes
    boxes = results[0].boxes
    labels = results[0].names



    # Initialize a counter for naming cropped regions
    crop_counter = 0
    print(f"Processing image: {img_name}")
    for box in boxes:
        cls_id = int(box.cls[0])
        if cls_id != 0:  # only process class 0
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        label = labels[cls_id]

        # Crop the region from the image
        cropped_img = img[y1:y2, x1:x2]

        # Save the cropped region
        crop_filename = os.path.join(output_dir, f"{img_name.split('.')[0]}_{crop_counter}.jpg")
        cv2.imwrite(crop_filename, cropped_img)
        crop_counter += 1

print(f"Cropped regions saved to: {output_dir}")