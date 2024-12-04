import os
import cv2
from ultralytics import YOLO
from datetime import datetime
import requests  # To send data to app.py
import time
import re
import easyocr

VIDEO_PATH_CONFIG = 'video_path.txt'
# Load the YOLOv8 model
model = YOLO('best.pt')  # Path to your YOLOv8 model
output_folder = 'result'  # Path to intermediate result folder
final_output_folder = 'final'  # Path for final processed images

# Paths
# video_path = '/home/chaitu/Downloads/hello.mp4'  # Path to the input video
def get_video_path():
    try:
        with open(VIDEO_PATH_CONFIG, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        # Fallback to a default path if config file doesn't exist
        return 'hiv0064.mp4'
    
video_path = get_video_path()

# Create the output folders if they don't exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(final_output_folder, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get the video frame rate (frames per second)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps * 2)  # Skip 30 seconds of video between frames

frame_count = 0
reader = easyocr.Reader(['en'])

def crop_image(frame, bounding_box):
    x, y, w, h = bounding_box
    return frame[y:y+h, x:x+w]

def extract_timestamp(frame):
    bounding_box = (70, 1160, 545, 120)  # Adjust based on your video
    timestamp_region = crop_image(frame, bounding_box)
    results = reader.readtext(timestamp_region, allowlist='0123456789-: ')
    
    timestamp = ''
    for _, text, _ in results:
        timestamp += text.strip() + ' '
    
    match = re.search(r'\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}:\d{2}', timestamp)
    if match:
        return match.group().replace(':', '-').replace(' ', '_')
    return None

def load_bounding_boxes(file_path):
    bounding_boxes = []
    with open(file_path, 'r') as f:
        for line in f:
            data = line.strip().split()
            if len(data) >= 5:
                _, x_center, y_center, width, height = map(float, data[:5])
                bounding_boxes.append((x_center, y_center, width, height))
    return bounding_boxes

def convert_to_absolute_coords(bounding_boxes, img_width, img_height):
    absolute_boxes = []
    for x_center, y_center, width, height in bounding_boxes:
        x1 = int((x_center - width / 2) * img_width)
        y1 = int((y_center - height / 2) * img_height)
        x2 = int((x_center + width / 2) * img_width)
        y2 = int((y_center + height / 2) * img_height)
        absolute_boxes.append((x1, y1, x2, y2))
    return absolute_boxes


def process_image(image, bounding_boxes, output_path, overlap_threshold=0.4):
    img_height, img_width, _ = image.shape
    absolute_boxes = convert_to_absolute_coords(bounding_boxes, img_width, img_height)

    dirty_count = 0  # Counter for dirty segments
    rows, cols = 24, 12
    segment_width = img_width // cols
    segment_height = img_height // rows
    
    for i in range(5,7):
        for j in range(2,3):
            x1 = j * segment_width
            y1 = i * segment_height
            x2 = x1 + segment_width
            y2 = y1 + segment_height
            segment_area = segment_width * segment_height

            intersection_area = 0
            for bx1, by1, bx2, by2 in absolute_boxes:
                intersect_x1 = max(x1, bx1)
                intersect_y1 = max(y1, by1)
                intersect_x2 = min(x2, bx2)
                intersect_y2 = min(y2, by2)
                if intersect_x1 < intersect_x2 and intersect_y1 < intersect_y2:
                    intersection_area += (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)
            
            if intersection_area > overlap_threshold * segment_area:
                dirty_count += 1
                color = (0,0,255)    
            else:
                color = (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    for i in range(4,7):
        for j in range(3,10):
            x1 = j * segment_width
            y1 = i * segment_height
            x2 = x1 + segment_width
            y2 = y1 + segment_height
            segment_area = segment_width * segment_height

            intersection_area = 0
            for bx1, by1, bx2, by2 in absolute_boxes:
                intersect_x1 = max(x1, bx1)
                intersect_y1 = max(y1, by1)
                intersect_x2 = min(x2, bx2)
                intersect_y2 = min(y2, by2)
                if intersect_x1 < intersect_x2 and intersect_y1 < intersect_y2:
                    intersection_area += (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)
            
            if intersection_area > overlap_threshold * segment_area:
                dirty_count += 1
                color = (0,0,255)    
            else:
                color = (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
    cv2.imwrite(output_path, image)
    return dirty_count

dirty_segments_data = []

avg_counter = 0
avg_pointer = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        frame_name = f"frame_{frame_count}.jpg"
        frame_path = os.path.join(output_folder, frame_name)
        cv2.imwrite(frame_path, frame)

        results = model.predict(source=frame, save=False)
        bounding_boxes = [box.xyxy[0].tolist() for result in results for box in result.boxes]

        txt_filename = os.path.splitext(frame_name)[0] + '.txt'
        txt_path = os.path.join(output_folder, txt_filename)
        with open(txt_path, 'w') as f:
            for box in bounding_boxes:
                x1, y1, x2, y2 = box
                x_center = (x1 + x2) / 2 / frame.shape[1]
                y_center = (y1 + y2) / 2 / frame.shape[0]
                width = (x2 - x1) / frame.shape[1]
                height = (y2 - y1) / frame.shape[0]
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        temp_name = extract_timestamp(frame)
        if temp_name is None:
            temp_name = f"frame_{frame_count}.jpg"
        else:
            temp_name += ".jpg"
        final_image_path = os.path.join(final_output_folder, temp_name)
        dirty_count = process_image(frame, load_bounding_boxes(txt_path), final_image_path, overlap_threshold=0.4)
        print(f"Frame {frame_count}: {dirty_count} dirty segments")
        
        avg_pointer += 1
        avg_counter = (avg_counter*(avg_pointer-1) + dirty_count)//avg_pointer
        if avg_pointer % 5 == 0:
            try:
                response = requests.post(
                    'http://localhost:5000/receive_dirty_data',
                    json={"dirty_segments_data": [{"frame": temp_name, "dirty_segments": avg_counter}]}
                )
                print(f"Data sent successfully: {response.status_code} - {response.text}")
            except Exception as e:
                print(f"Error sending data: {str(e)}")
                
            avg_counter = 0
            avg_pointer = 0
            
            # Small delay to prevent overwhelming the server
        time.sleep(2)

    frame_count += 1

cap.release()

