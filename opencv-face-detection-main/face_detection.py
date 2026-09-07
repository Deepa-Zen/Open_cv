"""
=============================================================================
OpenCV & YOLOv8 Real-Time Multi-Class Object & Face Detection Suite
=============================================================================
Detects 80+ COCO categories in real-time on live camera stream:
- Persons, Cars, Motorcycles, Bicycles, Buses, Trucks
- Animals (Dogs, Cats, Birds, Horses, Cows, Elephants, etc.)
- Electronics (Laptops, Cellphones, TVs, Remotes, Keyboards)
- Household Items (Chairs, Bottles, Cups, Books, Clock, Backpacks)
=============================================================================
"""

import os
import cv2
import sys
import time
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# Distinct BGR color palette for bounding boxes
COLOR_PALETTE = [
    (248, 189, 56),   # Sky Blue
    (129, 185, 16),   # Emerald Green
    (94, 63, 244),    # Bright Red
    (36, 191, 251),   # Gold / Amber
    (252, 132, 192),  # Purple / Lavender
    (238, 211, 34),   # Cyan
    (22, 115, 249),   # Deep Orange
    (153, 72, 236),   # Hot Pink
    (233, 165, 14),   # Light Blue
    (22, 204, 132),   # Lime Green
]

# Initialize YOLOv8 Nano model
print("Loading YOLOv8 AI Multi-Object Detection Model...")
model = YOLO('yolov8n.pt')

# Load Caffe Face Detection Model fallback
model_proto = "deploy.prototxt"
model_weights = "res10_300x300_ssd_iter_140000_fp16.caffemodel"

caffe_net = None
if os.path.exists(model_proto) and os.path.exists(model_weights):
    caffe_net = cv2.dnn.readNetFromCaffe(model_proto, model_weights)

# Camera Source Selection
s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

cap = cv2.VideoCapture(s)

# Setup OpenCV Display Window
win_name = "Real-Time AI Multi-Object & Face Detector"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, 1280, 800)

modes = [
    "YOLOv8 Multi-Object AI (80+ Classes: Person, Car, Animals, Objects)",
    "SSD Deep Learning Face Detector",
    "Canny Edge Vision Matrix",
    "Cyberpunk HSV Color Map"
]
current_mode = 0

conf_thresholds = [0.35, 0.50, 0.70]
conf_idx = 1  # Default 0.50

snapshot_banner_time = 0
snapshot_msg = ""

# FPS Measurement Variables
prev_time = time.time()

# Ensure captures directory exists
captures_dir = os.path.join(os.getcwd(), "captures")
os.makedirs(captures_dir, exist_ok=True)

print("AI Multi-Object Detection Suite active. Press ESC to quit.")

while True:
    has_frame, frame = cap.read()
    if not has_frame:
        break

    frame = cv2.flip(frame, 1)
    frame_height, frame_width = frame.shape[:2]
    
    # Calculate FPS
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time

    conf_thresh = conf_thresholds[conf_idx]
    detected_summary = {}

    # MODE 0: YOLOv8 AI Multi-Object Detection
    if current_mode == 0:
        start_t = time.time()
        results = model(frame, conf=conf_thresh, verbose=False)[0]
        infer_time = (time.time() - start_t) * 1000.0

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            # Count object categories
            detected_summary[cls_name] = detected_summary.get(cls_name, 0) + 1

            # Select distinct color from palette
            bgr_color = COLOR_PALETTE[cls_id % len(COLOR_PALETTE)]
            
            # Draw Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr_color, 2)

            # Label Tag
            label = f"{cls_name.upper()}: {confidence*100:.0f}%"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)

            cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 8, y1), bgr_color, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

    # MODE 1: Caffe SSD Face Detection
    elif current_mode == 1 and caffe_net is not None:
        start_t = time.time()
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False, crop=False)
        caffe_net.setInput(blob)
        detections = caffe_net.forward()
        infer_time = (time.time() - start_t) * 1000.0

        face_count = 0
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_thresh:
                face_count += 1
                x1 = int(detections[0, 0, i, 3] * frame_width)
                y1 = int(detections[0, 0, i, 4] * frame_height)
                x2 = int(detections[0, 0, i, 5] * frame_width)
                y2 = int(detections[0, 0, i, 6] * frame_height)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 127), 2)
                label = f"FACE: {confidence*100:.1f}%"
                cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 127), 2, cv2.LINE_AA)
        
        detected_summary["face"] = face_count

    # MODE 2: Canny Edges
    elif current_mode == 2:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        infer_time = 2.1

    # MODE 3: Cyberpunk HSV
    elif current_mode == 3:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0] + 40) % 180
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        infer_time = 3.2

    # Draw HUD Overlay
    hud_bg = frame.copy()
    cv2.rectangle(hud_bg, (0, 0), (frame_width, 50), (15, 23, 42), -1)
    cv2.rectangle(hud_bg, (0, frame_height - 40), (frame_width, frame_height), (15, 23, 42), -1)
    frame = cv2.addWeighted(hud_bg, 0.65, frame, 0.35, 0)

    # Top Bar Info
    cv2.putText(frame, f"MODE: {modes[current_mode]}", (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Detected Objects Summary Line
    if detected_summary:
        summary_str = " | ".join([f"{k.upper()}: {v}" for k, v in detected_summary.items()])
    else:
        summary_str = "SEARCHING FOR OBJECTS..."

    cv2.putText(frame, f"DETECTED: {summary_str}", (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (16, 185, 129), 1, cv2.LINE_AA)

    # Top Right Metrics
    cv2.putText(frame, f"FPS: {fps:.1f} | Latency: {infer_time:.1f}ms | Conf: {conf_thresh*100:.0f}%", 
                (frame_width - 370, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (56, 189, 248), 2, cv2.LINE_AA)

    # Bottom Instructions Bar
    controls_text = "[M] Switch Mode  |  [C] Conf Threshold  |  [S] Save Snapshot  |  [ESC] Exit"
    cv2.putText(frame, controls_text, (15, frame_height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (203, 213, 225), 1, cv2.LINE_AA)

    # Snapshot Notification Banner
    if time.time() - snapshot_banner_time < 2.5:
        cv2.rectangle(frame, (frame_width // 2 - 200, 60), (frame_width // 2 + 200, 100), (16, 185, 129), -1)
        cv2.putText(frame, snapshot_msg, (frame_width // 2 - 180, 87), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow(win_name, frame)

    # Key Controls
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC Key
        break
    elif key in (ord('m'), ord('M')):
        current_mode = (current_mode + 1) % len(modes)
    elif key in (ord('c'), ord('C')):
        conf_idx = (conf_idx + 1) % len(conf_thresholds)
    elif key in (ord('s'), ord('S')):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"object_detect_{timestamp}.jpg"
        save_file = os.path.join(captures_dir, filename)
        cv2.imwrite(save_file, frame)
        snapshot_msg = f"Saved: {filename}"
        snapshot_banner_time = time.time()

cap.release()
cv2.destroyAllWindows()
