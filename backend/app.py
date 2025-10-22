from flask import Flask, request, jsonify
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow requests from frontend

# Load model
model = load_model("../model/emotion_mobilenet.h5")
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

def preprocess_face(face_img):
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (96, 96))
    face_norm = face_resized.astype("float32") / 255.0
    face_input = np.expand_dims(face_norm, axis=0)
    return face_input

@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():
    # Receive image
    file = request.files.get('frame')
    if file is None:
        return jsonify({"error": "No frame received"}), 400

    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Predict
    face_input = preprocess_face(img)
    preds = model.predict(face_input, verbose=0)[0]
    max_index = np.argmax(preds)
    emotion = emotion_labels[max_index]
    confidence = float(preds[max_index])

    return jsonify({"emotion": emotion, "confidence": confidence})

if __name__ == '__main__':
    app.run(debug=True)
