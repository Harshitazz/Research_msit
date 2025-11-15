import cv2
import tensorflow as tf
import numpy as np
from model_loader import get_model
from text_model_loader import get_text_model

# Map output labels to emotions (example)
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Load face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def predict_emotion(frame_bytes: bytes, text_payload: str):
    EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    img_preds = np.zeros(1, len(EMOTIONS))
    text_preds = np.zeros(1, len(EMOTIONS))
    emotion = "Neutral"

    if frame_bytes != b"":
        # Decode image
        np_img = np.frombuffer(frame_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Detect faces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Preprocess
        if len(faces) == 0:
            return "Neutral"  # No face detected

        x, y, w, h = faces[0]
        face = img[y:y+h, x:x+w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(img, (48, 48))
        face = face.astype("float32") / 255.0

        # ✅ No stacking here, since image already has 3 channels
        if len(face.shape) == 2:  # in case grayscale
            face = np.stack([img]*3, axis=-1)

        face = np.expand_dims(img, axis=0)  # shape -> (1, 48, 48, 3)

        print("Final image shape:", face.shape)

        model = get_model()
        img_preds = model.predict(img)
        emotion = EMOTIONS[np.argmax(img_preds)]
    
    if text_payload != "":
        text_model = get_text_model()
        text_input = tf.convert_to_tensor([text_payload], dtype=tf.string)
        text_preds = text_model.predict(text_input)
        emotion = EMOTIONS[np.argmax(text_preds)]
    
    return emotion