import tensorflow as tf
import keras_nlp
import threading

_text_model = None
_lock = threading.Lock()

def get_text_model():
    global _text_model
    with _lock:
        if _text_model is None:
            num_labels = 7

            # Reload preset from KerasNLP
            preprocessor = keras_nlp.models.BertPreprocessor.from_preset("bert_base_en", sequence_length=128)
            encoder = keras_nlp.models.BertBackbone.from_preset("bert_base_en")

            # Rebuild model architecture (same as in training)
            text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
            x = preprocessor(text_input)
            x = encoder(x)['pooled_output']
            x = tf.keras.layers.Dropout(0.3)(x)
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            output = tf.keras.layers.Dense(num_labels, activation='softmax', dtype='float32')(x)
            _text_model = tf.keras.Model(inputs=text_input, outputs=output)

            #Load weights
            _text_model.load_weights("sentiment_analysis.weights.h5")
        return _text_model