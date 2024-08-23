import cv2
import time
from deepface import DeepFace


cap = cv2.VideoCapture(0)

# FPS calculation
fps_start_time = 0
fps = 0



while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load OpenCV's pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # For FPS calculation
    fps_end_time = time.time()
    fps = 1 / (fps_end_time - fps_start_time)
    fps_start_time = fps_end_time

    face_id = 1  # Face numbering

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extract the face from the frame
        face = frame[y:y+h, x:x+w]

        # Analyze the emotion of the face using DeepFace
        try:
            # Analyzing emotions with probabilities
            emotion_analysis = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
            dominant_emotion = emotion_analysis.get('dominant_emotion', "Unknown")
            dominant_emotion_score = emotion_analysis.get('emotion', {}).get(dominant_emotion, 0)
            emotions = emotion_analysis.get('emotion', {})

            # Display the dominant emotion and its score
            emotion_text = f"{dominant_emotion} ({dominant_emotion_score:.2f}%)"
        except Exception as e:
            print(f"Error processing face: {e}")
            emotion_text = "Error"
            emotions = {}


        face_label = f"Face #{face_id}"
        cv2.putText(frame, face_label, (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

        # Display the dominant emotion on the screen
        cv2.putText(frame, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)



        y_offset = y + h + 20
        for emotion, score in emotions.items():
            emotion_info = f"{emotion}: {score:.2f}%"
            cv2.putText(frame, emotion_info, (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            y_offset += 20

        face_id += 1  # Increment the face number



    # Display FPS on the screen
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Emotion Detector', frame)


    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
