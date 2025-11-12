# model_loader.py
import tensorflow as tf
import threading

_model = None
_lock = threading.Lock()

def get_model():
    global _model
    with _lock:
        if _model is None:
            # Use Keras load_model for .h5 format
            _model = tf.keras.models.load_model("resnet_emotion_functional.h5", compile=False)
        return _model