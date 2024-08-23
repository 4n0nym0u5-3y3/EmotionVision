# EmotionVision

**EmotionVision** is a real-time emotion detection system that uses a webcam to capture video feed, detects faces in the frame, and analyzes their emotions. The system identifies emotions such as happiness, sadness, anger, and others, displaying the dominant emotion along with its confidence score. It also provides a real-time FPS counter to monitor performance.

## Features

- **Real-Time Face Detection**: Detects multiple faces in the webcam feed.
- **Emotion Analysis**: Analyzes emotions and displays the dominant emotion with confidence scores.
- **Multi-Emotion Display**: Shows probabilities for all detected emotions for each face.
- **FPS Display**: Real-time FPS counter for performance monitoring.
- **Face Labeling**: Numbers each detected face to distinguish between them.

## Installation

To run the EmotionVision project, you need to install the required Python libraries. You can do this using `pip`:

```bash
pip install opencv-python deepface
```

## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/emotionvision.git
   cd emotionvision
   ```

2. Run the main script:
   ```bash
   python emotionvision.py
   ```

3. The application will start the webcam, detect faces, and display the emotions on the screen. Press `q` to quit the application.

## Code Overview

```python
import cv2
import time
from deepface import DeepFace
```
- **cv2**: OpenCV library for face detection and video processing.
- **time**: Used for FPS calculation.
- **DeepFace**: A powerful deep learning library to analyze emotions.

### Key Functions

- **Face Detection**: Detects faces in real-time using OpenCV's `CascadeClassifier`.
- **Emotion Analysis**: Analyzes the detected face to determine the dominant emotion and its confidence score using `DeepFace`.
- **FPS Calculation**: Calculates and displays the frames per second (FPS) for performance monitoring.
- **Multi-Emotion Display**: Displays probabilities of other emotions detected in the face.
- **Face Labeling**: Numbers each detected face for easy identification.
