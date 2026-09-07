"""
=============================================================================
Real-Time Face Detection Application using OpenCV
=============================================================================

Dependencies:
    pip install opencv-python numpy

Usage:
    python realtime_face_detection.py

Features:
    - Real-time webcam video stream capture with error handling
    - Accurate face detection using OpenCV Haar Cascade Classifier
    - Customizable bounding box colors, thickness, and minimum face size filters
    - Live face counter overlay ("Face 1", "Face 2", etc.)
    - Real-time FPS (Frames Per Second) performance display
    - Clean console status logging per frame
    - Graceful application exit (Press 'q' or 'ESC' to quit)

Author: OpenCV Computer Vision Specialist
=============================================================================
"""

import cv2
import sys
import time
import numpy as np

# ================================ Configuration ================================ #
CAMERA_INDEX = 0                      # Default system webcam
WINDOW_TITLE = "Real-Time Face Detector - OpenCV"

# Customizable Styling
BOX_COLOR = (0, 255, 127)             # Neon Spring Green (BGR)
BOX_THICKNESS = 2                     # Line thickness in pixels
TEXT_COLOR = (0, 0, 0)                # Black text for tag
TAG_BG_COLOR = (0, 255, 127)          # Match box color for filled tag background
FPS_COLOR = (56, 189, 248)            # Cyan for FPS readout (BGR)

# Detection Parameters (Optimized for real-time performance)
SCALE_FACTOR = 1.1                    # Image pyramid scale factor (1.1 = 10% reduction per scale)
MIN_NEIGHBORS = 5                     # Higher value reduces false positives
MIN_FACE_SIZE = (60, 60)              # Minimum face dimension filter (W, H)
# ============================================================================== #


class RealtimeFaceDetector:
    """
    Production-ready Real-time Face Detector class using OpenCV.
    Handles video stream initialization, frame processing, face detection,
    bounding box rendering, FPS calculation, and graceful exit handling.
    """

    def __init__(self, camera_idx=CAMERA_INDEX):
        self.camera_idx = camera_idx
        self.cap = None

        # Load Haar Cascade Classifier
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if self.face_cascade.empty():
            print(f"[ERROR] Unable to load Haar Cascade model from: {cascade_path}")
            sys.exit(1)

        print(f"[INFO] Successfully loaded Haar Cascade model: {cascade_path}")

        # Load YOLOv8 Pose Model for Hand & Leg Detection
        self.pose_model = None
        try:
            from ultralytics import YOLO
            self.pose_model = YOLO('yolov8n-pose.pt')
            print("[INFO] Successfully loaded YOLOv8 Pose Engine for Hands & Legs.")
        except Exception as e:
            print(f"[WARNING] Could not load YOLOv8 Pose Engine: {e}")

        # FPS metrics
        self.prev_tick = cv2.getTickCount()
        self.fps = 0.0

    def initialize_camera(self):
        """Attempts to open webcam stream with DirectShow backend fallback on Windows."""
        print(f"[INFO] Connecting to camera device ID: {self.camera_idx}...")

        # Try DirectShow first on Windows, then default MSMF/ANY
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY, cv2.CAP_MSMF]
        indices = [self.camera_idx, 1, 2, 0]

        for backend in backends:
            for idx in indices:
                try:
                    self.cap = cv2.VideoCapture(idx, backend)
                    if self.cap.isOpened():
                        # Set camera parameters
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                        # Test read frame
                        success, test_frame = self.cap.read()
                        if success and test_frame is not None and test_frame.size > 0:
                            print(f"[INFO] Camera stream initialized (Device #{idx}, Backend: {backend})")
                            self.camera_idx = idx
                            return True
                        else:
                            self.cap.release()
                except Exception:
                    pass

        print(f"[ERROR] Camera device could not be opened.")
        print("[ERROR] Please check webcam connection, permissions, or close conflicting camera apps.")
        return False

    def calculate_fps(self):
        """Calculates real-time Frames Per Second (FPS)."""
        current_tick = cv2.getTickCount()
        time_diff = (current_tick - self.prev_tick) / cv2.getTickFrequency()
        self.prev_tick = current_tick

        if time_diff > 0:
            self.fps = 1.0 / time_diff
        return self.fps

    def process_frame(self, frame):
        """
        Processes a single video frame: converts to grayscale, runs multi-scale face detection,
        draws styled bounding boxes, and overlays performance HUD metrics.
        """
        # Mirror frame horizontally for intuitive camera preview
        frame = cv2.flip(frame, 1)
        frame_h, frame_w = frame.shape[:2]

        # Convert to Grayscale for Haar Cascade processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Equalize Histogram to improve detection under varying lighting
        gray = cv2.equalizeHist(gray)

        # Detect Faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBORS,
            minSize=MIN_FACE_SIZE,
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        face_count = len(faces)

        # Render Bounding Boxes and Face Labels
        for idx, (x, y, w, h) in enumerate(faces, start=1):
            # Draw Main Bounding Box
            cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR, BOX_THICKNESS)

            # Corner Accent Reticles
            corner_length = int(min(w, h) * 0.15)
            cv2.line(frame, (x, y), (x + corner_length, y), (0, 255, 255), 3)
            cv2.line(frame, (x, y), (x, y + corner_length), (0, 255, 255), 3)
            cv2.line(frame, (x + w, y + h), (x + w - corner_length, y + h), (0, 255, 255), 3)
            cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_length), (0, 255, 255), 3)

            # Label Tag ("Face 1", "Face 2", etc.)
            label = f"Face {idx}"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)

            # Filled Label Background
            cv2.rectangle(
                frame,
                (x, y - text_h - 10),
                (x + text_w + 10, y),
                TAG_BG_COLOR,
                cv2.FILLED
            )

            # Label Text
            cv2.putText(
                frame,
                label,
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                TEXT_COLOR,
                1,
                cv2.LINE_AA
            )

        # Pose Estimation for Hands, Arms, Legs & Feet
        hand_count = 0
        leg_count = 0

        if self.pose_model is not None:
            try:
                results = self.pose_model(frame, conf=0.45, verbose=False)[0]
                if results.keypoints is not None:
                    for kpts in results.keypoints.data:
                        kpts_np = kpts.cpu().numpy()
                        # Keypoint circles
                        for k_idx, (x, y, conf) in enumerate(kpts_np):
                            if conf > 0.35:
                                if k_idx in (9, 10):       # Hands / Wrists
                                    cv2.circle(frame, (int(x), int(y)), 7, (0, 255, 255), -1)
                                    cv2.putText(frame, "HAND ✋", (int(x) + 6, int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                                elif k_idx in (15, 16):    # Ankles / Feet
                                    cv2.circle(frame, (int(x), int(y)), 7, (255, 0, 255), -1)
                                    cv2.putText(frame, "FOOT 👟", (int(x) + 6, int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)
                                elif k_idx in (13, 14):    # Knees
                                    cv2.circle(frame, (int(x), int(y)), 6, (0, 165, 255), -1)
                                    cv2.putText(frame, "KNEE 🦵", (int(x) + 6, int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)
                                else:
                                    cv2.circle(frame, (int(x), int(y)), 4, (56, 189, 248), -1)

                        # Draw Limb Connection Skeleton Lines
                        connections = [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
                        for p1, p2 in connections:
                            if p1 < len(kpts_np) and p2 < len(kpts_np):
                                x1, y1, c1 = kpts_np[p1]
                                x2, y2, c2 = kpts_np[p2]
                                if c1 > 0.35 and c2 > 0.35:
                                    color = (0, 255, 127) if p1 in (5,6,7,8,9,10) or p2 in (7,8,9,10) else (255, 140, 0)
                                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                        # Bounding box for Left Hand (9) & Right Hand (10)
                        if len(kpts_np) > 9 and kpts_np[9][2] > 0.35:
                            hand_count += 1
                            hx, hy = int(kpts_np[9][0]), int(kpts_np[9][1])
                            cv2.rectangle(frame, (max(0, hx - 30), max(0, hy - 30)), (min(frame_w, hx + 30), min(frame_h, hy + 30)), (0, 255, 255), 2)
                            cv2.putText(frame, "LEFT HAND 🖐️", (max(0, hx - 30), max(15, hy - 34)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

                        if len(kpts_np) > 10 and kpts_np[10][2] > 0.35:
                            hand_count += 1
                            hx, hy = int(kpts_np[10][0]), int(kpts_np[10][1])
                            cv2.rectangle(frame, (max(0, hx - 30), max(0, hy - 30)), (min(frame_w, hx + 30), min(frame_h, hy + 30)), (0, 200, 255), 2)
                            cv2.putText(frame, "RIGHT HAND 🖐️", (max(0, hx - 30), max(15, hy - 34)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)

                        # Bounding box for Left Leg & Right Leg
                        l_leg_pts = [kpts_np[pt][:2] for pt in [11, 13, 15] if len(kpts_np) > pt and kpts_np[pt][2] > 0.35]
                        if len(l_leg_pts) >= 2:
                            leg_count += 1
                            xs, ys = [p[0] for p in l_leg_pts], [p[1] for p in l_leg_pts]
                            cv2.rectangle(frame, (max(0, int(min(xs)) - 20), max(0, int(min(ys)) - 20)),
                                          (min(frame_w, int(max(xs)) + 20), min(frame_h, int(max(ys)) + 20)), (255, 0, 255), 2)
                            cv2.putText(frame, "LEFT LEG 🦵", (max(0, int(min(xs)) - 20), max(15, int(min(ys)) - 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)

                        r_leg_pts = [kpts_np[pt][:2] for pt in [12, 14, 16] if len(kpts_np) > pt and kpts_np[pt][2] > 0.35]
                        if len(r_leg_pts) >= 2:
                            leg_count += 1
                            xs, ys = [p[0] for p in r_leg_pts], [p[1] for p in r_leg_pts]
                            cv2.rectangle(frame, (max(0, int(min(xs)) - 20), max(0, int(min(ys)) - 20)),
                                          (min(frame_w, int(max(xs)) + 20), min(frame_h, int(max(ys)) + 20)), (255, 140, 0), 2)
                            cv2.putText(frame, "RIGHT LEG 🦵", (max(0, int(min(xs)) - 20), max(15, int(min(ys)) - 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 140, 0), 1, cv2.LINE_AA)
            except Exception:
                pass

        # Draw HUD Overlay Bar (Top Bar)
        current_fps = self.calculate_fps()

        # Semi-transparent HUD background
        hud_overlay = frame.copy()
        cv2.rectangle(hud_overlay, (0, 0), (frame_w, 45), (15, 23, 42), -1)
        frame = cv2.addWeighted(hud_overlay, 0.65, frame, 0.35, 0)

        # Render HUD Text
        status_text = f"FACES: {face_count} | HANDS: {hand_count} | LEGS: {leg_count}"
        fps_text = f"FPS: {current_fps:.1f}"
        controls_text = "Press 'q' or 'ESC' to Quit"

        cv2.putText(frame, status_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, fps_text, (frame_w - 180, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, FPS_COLOR, 2, cv2.LINE_AA)

        # Console Logging
        print(f"\r[STATUS] FPS: {current_fps:.1f} | Faces: {face_count} | Hands: {hand_count} | Legs: {leg_count}   ", end="", flush=True)

        return frame

    def run(self):
        """Main application execution loop."""
        if not self.initialize_camera():
            return

        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_TITLE, 1024, 600)

        print("\n" + "=" * 60)
        print(" Real-Time Face Detector Active")
        print(" Press 'q' or 'ESC' in the video window to stop.")
        print("=" * 60 + "\n")

        empty_frame_count = 0
        try:
            while True:
                has_frame, frame = self.cap.read()

                if not has_frame or frame is None or frame.size == 0:
                    empty_frame_count += 1
                    if empty_frame_count > 15:
                        print("\n[ERROR] Persistent empty frames received from camera stream. Exiting loop.")
                        break
                    time.sleep(0.03)
                    continue

                empty_frame_count = 0

                # Process frame
                output_frame = self.process_frame(frame)

                # Display in Window
                cv2.imshow(WINDOW_TITLE, output_frame)

                # Check Keypress ('q', 'Q', or ESC key)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q') or key == 27:
                    print("\n\n[INFO] Exit command received. Stopping face detector...")
                    break

        except KeyboardInterrupt:
            print("\n\n[INFO] Keyboard interrupt detected (Ctrl+C). Exiting...")

        finally:
            self.cleanup()

    def cleanup(self):
        """Releases video capture resources and destroys OpenCV windows cleanly."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            print("[INFO] Camera device released.")

        cv2.destroyAllWindows()
        print("[INFO] OpenCV display windows closed. Program terminated cleanly.")


# ================================ Entry Point ================================ #
if __name__ == "__main__":
    detector = RealtimeFaceDetector(camera_idx=CAMERA_INDEX)
    detector.run()
