import cv2
import tensorflow as tf
import numpy as np
from model_loader import get_model
from text_model_loader import get_text_model

# Map output labels to emotions (example)
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def predict_emotion(frame_bytes: bytes, text_payload: str):
    EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    img_preds = np.ndarray([0 for _ in range(len(EMOTIONS))])
    text_preds = np.ndarray([0 for _ in range(len(EMOTIONS))])
    emotion = "Neutral"

    if frame_bytes != b"":
        # Decode image
        np_img = np.frombuffer(frame_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Preprocess
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (48, 48))
        img = img.astype("float32") / 255.0

        # ✅ No stacking here, since image already has 3 channels
        if len(img.shape) == 2:  # in case grayscale
            img = np.stack([img]*3, axis=-1)

        img = np.expand_dims(img, axis=0)  # shape -> (1, 48, 48, 3)

        print("Final image shape:", img.shape)

        model = get_model()
        img_preds = model.predict(img)
    
    if text_payload != "":
        text_model = get_text_model()
        text_input = tf.convert_to_tensor([text_payload], dtype=tf.string)
        text_preds = text_model.predict(text_input)
    
    emotion = EMOTIONS[np.argmax(0.8*img_preds + 0.2*text_preds)]
    return emotion