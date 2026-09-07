# ⚡ Open_cv — Next-Gen AI Vision, Biometrics & Pose Command Center

An advanced, real-time Computer Vision & AI Platform built using **Python, OpenCV, YOLOv8, and Flask**. It features full-body pose estimation, facial biometrics analysis, rPPG heart rate monitoring, dress color recognition, thermal infrared visual overlays, virtual security perimeter intrusion alerts, and an interactive 10-slide master presentation deck.

---

## 🌟 Key Features

- 🦾 **Full-Body Pose & Gesture Tracking**: Real-time 17 COCO skeletal keypoint estimation (Hands, Wrists, Elbows, Knees, Legs, Feet) powered by YOLOv8.
- 🤖 **YOLOv8 Multi-Object Detection**: Identifies 80+ object classes with distance estimation & tracking.
- 👤 **Facial Biometrics & Analytics**: Head pose direction, eye gaze tracking, emotion detection, and clothing/dress color recognition.
- 💓 **rPPG Vital Signs Monitoring**: Non-contact pulse/heart-rate estimation (BPM) & Eye Aspect Ratio (EAR) drowsiness index.
- 🛡️ **Virtual Security Perimeter**: Real-time ROI intrusion alerts with visual overlays & voice alerts.
- 🖥️ **Master Presentation Deck**: Built-in 10-slide interactive presentation deck accessible at `/presentation`.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install opencv-python ultralytics flask numpy torch torchvision
```

### 2. Launch the Application Server
```bash
python server.py
```

### 3. Open in Browser
- **AI Vision Command Center**: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
- **Interactive Presentation Deck**: [http://127.0.0.1:5000/presentation](http://127.0.0.1:5000/presentation)

---

## 📁 Repository Structure

```text
.
├── presentation/                 # Master Presentation Deck (index.html, styles.css, app.js)
├── static/                       # Web App CSS, JS, and Captured Snapshots
│   ├── css/styles.css
│   └── js/app.js
├── templates/                    # Web App Templates
│   └── index.html                # Cyberpunk AI Vision Command Center UI
├── server.py                     # Main Flask Master AI Server
├── realtime_face_detection.py    # Standalone Desktop OpenCV App
├── yolov8n.pt & yolov8n-pose.pt # Pretrained YOLOv8 Weights
└── README.md
```

---

## 📜 License
MIT License. Created for AI & Computer Vision Masterclass.
