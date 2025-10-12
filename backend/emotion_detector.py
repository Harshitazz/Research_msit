import cv2
import numpy as np
from model_loader import get_model

# Map output labels to emotions (example)
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def predict_emotion(frame_bytes):
    # Decode image
    np_img = np.frombuffer(frame_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Preprocess
    img = cv2.resize(img, (48, 48)) / 255.0
    img = np.expand_dims(img, axis=0)
    print(type(img), img.shape)
    print("calling getmodel")
    model = get_model()
    print(model.summary())
    print("model loaded")
    preds = model.predict(img)
    print("preds:", preds)
    emotion = EMOTIONS[np.argmax(preds)]
    return emotion
