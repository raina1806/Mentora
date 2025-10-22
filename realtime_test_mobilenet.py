import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ---- Load trained model ----
model = load_model("model/emotion_mobilenet.h5")

# Emotion labels
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

# ---- Initialize webcam and face detector ----
cap = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ---- Real-time loop ----
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Crop and preprocess face
        face_rgb = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (96, 96))      # MobileNetV2 input
        face_norm = face_resized.astype("float32") / 255.0
        face_input = np.expand_dims(face_norm, axis=0)     # Shape: (1, 96, 96, 3)

        # Predict emotion
        preds = model.predict(face_input, verbose=0)[0]
        max_index = np.argmax(preds)
        emotion = emotion_labels[max_index]
        confidence = preds[max_index]

        # Optional: show "uncertain" if confidence < 0.4
        if confidence < 0.4:
            emotion = "Uncertain"

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
        label_text = f"{emotion} ({confidence*100:.1f}%)"
        cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # Show frame
    cv2.imshow("Emotion Detection - MobileNetV2", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
