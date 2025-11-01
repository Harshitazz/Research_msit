import cv2
import numpy as np
from model_loader import get_model

# Map output labels to emotions (example)
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def predict_emotion(frame_bytes):
    import cv2
    import numpy as np
    from model_loader import get_model

    EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

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
    preds = model.predict(img)
    emotion = EMOTIONS[np.argmax(preds)]
    return emotion
