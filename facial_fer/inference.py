# facial_fer/inference.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

model = load_model('models/emotion_model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def predict_emotion(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    face = cv2.resize(img, (48, 48))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, 0)
    prediction = model.predict(face)
    label = emotion_labels[np.argmax(prediction)]
    return label
