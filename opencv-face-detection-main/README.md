# Face Detection with OpenCV

This project demonstrates a real-time face detection application using OpenCV's pre-trained deep learning models. The application can detect faces in a live webcam feed or video input, drawing bounding boxes around detected faces along with the confidence score.

## Features

- **Real-time Face Detection**: Uses a pre-trained deep learning model (`res10_300x300_ssd_iter_140000_fp16.caffemodel`) for accurate face detection.
- **Confidence Display**: Shows the confidence score for detected faces.
- **Performance Metrics**: Displays inference time for the model.

## Prerequisites

Ensure the following dependencies are installed:

- Python 3.x
- OpenCV (cv2)
- urllib (Standard Library)
- zipfile (Standard Library)
- numpy (comes with OpenCV)

## Setup Instructions

### 1. Download the Assets

The required model files (`deploy.prototxt` and `res10_300x300_ssd_iter_140000_fp16.caffemodel`) will be automatically downloaded and extracted into the current directory when the script is executed.

The assets are downloaded from:
[Pre-trained Model](https://www.dropbox.com/s/efitgt363ada95a/opencv_bootcamp_assets_12.zip?dl=1)

### 2. Run the Application

Run the following command to start the application:


python face_detection.py


By default, it will use the webcam feed as input.

3. Exit the Application
  Press the Esc key to exit the application.

### File Structure
.
├── face_detection.py        # Main Python script
├── deploy.prototxt          # Model architecture file (auto-downloaded)
├── res10_300x300_ssd_iter_140000_fp16.caffemodel  # Pre-trained weights (auto-downloaded)
└── opencv_bootcamp_assets_12.zip  # Downloaded ZIP file (optional)

### How It Works

1.The script downloads the pre-trained model files if not already present.
2.A 4D blob is created from each frame and passed to the model for inference.
3.The model detects faces and outputs bounding box coordinates along with confidence scores.
4.Detected faces are highlighted with bounding boxes, and inference time is displayed.

### Dependencies Installation
Install the required Python packages using pip:
   --pip install opencv-python
(Optional: For additional functionalities, you may install opencv-contrib-python.)

### References
OpenCV Documentation: https://docs.opencv.org/
Pre-trained Caffe Model: https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector

### License
This project is open-source and available under the MIT License. You are free to use, modify, and distribute this project as per the license.
