"""
=============================================================================
OpenCV & YOLOv8 Real-Time AI Computer Vision Web Application Project
=============================================================================
Flask Backend with Live Video Streaming, Object Detection Analytics & Web Controls
=============================================================================
"""

import os
import cv2
import time
import json
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from ultralytics import YOLO

app = Flask(__name__, static_folder='static', template_folder='templates')

# Color Palette for YOLO Bounding Boxes
COLOR_PALETTE = [
    (248, 189, 56),   # Sky Blue
    (129, 185, 16),   # Emerald Green
    (94, 63, 244),    # Bright Red
    (36, 191, 251),   # Gold / Amber
    (252, 132, 192),  # Purple
    (238, 211, 34),   # Cyan
    (22, 115, 249),   # Deep Orange
    (153, 72, 236),   # Hot Pink
    (233, 165, 14),   # Light Blue
    (22, 204, 132),   # Lime Green
]

# Global AI Model & State
print("Loading YOLOv8 Model for Web App...")
yolo_model = YOLO('yolov8n.pt')

# Load Caffe Face Model if available
caffe_net = None
model_proto = os.path.join("opencv-face-detection-main", "deploy.prototxt")
model_weights = os.path.join("opencv-face-detection-main", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

if not os.path.exists(model_proto):
    model_proto = "deploy.prototxt"
    model_weights = "res10_300x300_ssd_iter_140000_fp16.caffemodel"

if os.path.exists(model_proto) and os.path.exists(model_weights):
    caffe_net = cv2.dnn.readNetFromCaffe(model_proto, model_weights)

# System State
app_state = {
    "mode": 0,           # 0: YOLOv8, 1: Face SSD, 2: Canny, 3: Cyberpunk
    "confidence": 0.50,  # 0.10 to 0.95
    "fps": 0.0,
    "latency": 0.0,
    "camera_active": True,
    "detected_objects": {},
    "last_snapshot": None
}

captures_dir = os.path.join(os.getcwd(), "static", "captures")
os.makedirs(captures_dir, exist_ok=True)

camera = None

def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
    return camera

def generate_frames():
    global app_state
    cam = get_camera()
    prev_time = time.time()

    while True:
        if not app_state["camera_active"]:
            time.sleep(0.1)
            continue

        success, frame = cam.read()
        if not success:
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)
        frame_h, frame_w = frame.shape[:2]
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time
        app_state["fps"] = round(fps, 1)

        conf_thresh = app_state["confidence"]
        detected_counts = {}

        # MODE 0: YOLOv8 AI Multi-Object Detection
        if app_state["mode"] == 0:
            start_t = time.time()
            results = yolo_model(frame, conf=conf_thresh, verbose=False)[0]
            app_state["latency"] = round((time.time() - start_t) * 1000.0, 1)

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = yolo_model.names[cls_id]

                detected_counts[cls_name] = detected_counts.get(cls_name, 0) + 1
                bgr_color = COLOR_PALETTE[cls_id % len(COLOR_PALETTE)]

                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr_color, 2)
                label = f"{cls_name.upper()}: {confidence*100:.0f}%"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 8, y1), bgr_color, -1)
                cv2.putText(frame, label, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

        # MODE 1: Caffe SSD Face Detector
        elif app_state["mode"] == 1 and caffe_net is not None:
            start_t = time.time()
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False, crop=False)
            caffe_net.setInput(blob)
            detections = caffe_net.forward()
            app_state["latency"] = round((time.time() - start_t) * 1000.0, 1)

            face_count = 0
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > conf_thresh:
                    face_count += 1
                    x1 = int(detections[0, 0, i, 3] * frame_w)
                    y1 = int(detections[0, 0, i, 4] * frame_h)
                    x2 = int(detections[0, 0, i, 5] * frame_w)
                    y2 = int(detections[0, 0, i, 6] * frame_h)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 127), 2)
                    label = f"FACE: {confidence*100:.1f}%"
                    cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 127), 2, cv2.LINE_AA)
            detected_counts["face"] = face_count

        # MODE 2: Canny Edges
        elif app_state["mode"] == 2:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            app_state["latency"] = 2.1

        # MODE 3: Cyberpunk HSV
        elif app_state["mode"] == 3:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv[:, :, 0] = (hsv[:, :, 0] + 40) % 180
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            app_state["latency"] = 3.2

        app_state["detected_objects"] = detected_counts

        # Encode Frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/state', methods=['GET', 'POST'])
def api_state():
    global app_state
    if request.method == 'POST':
        data = request.json
        if 'mode' in data:
            app_state['mode'] = int(data['mode'])
        if 'confidence' in data:
            app_state['confidence'] = float(data['confidence'])
        if 'camera_active' in data:
            app_state['camera_active'] = bool(data['camera_active'])
    return jsonify(app_state)

@app.route('/api/snapshot', methods=['POST'])
def api_snapshot():
    global app_state
    cam = get_camera()
    success, frame = cam.read()
    if success:
        frame = cv2.flip(frame, 1)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{timestamp}.jpg"
        filepath = os.path.join(captures_dir, filename)
        cv2.imwrite(filepath, frame)
        app_state['last_snapshot'] = filename
        return jsonify({"status": "success", "filename": filename, "url": f"/static/captures/{filename}"})
    return jsonify({"status": "error", "message": "Failed to capture frame"}), 500

@app.route('/api/gallery', methods=['GET'])
def api_gallery():
    files = sorted(os.listdir(captures_dir), reverse=True)
    images = [f"/static/captures/{f}" for f in files if f.endswith(('.jpg', '.png'))]
    return jsonify({"images": images[:12]})

if __name__ == '__main__':
    print("Starting OpenCV Web Application Server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
