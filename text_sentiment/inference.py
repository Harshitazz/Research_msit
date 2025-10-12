# text_sentiment/inference.py
from transformers import BertTokenizerFast, TFBertForSequenceClassification
import tensorflow as tf
import numpy as np

model = TFBertForSequenceClassification.from_pretrained("models/bert_emotion")
tokenizer = BertTokenizerFast.from_pretrained("models/bert_emotion")

def predict_text_emotion(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=64)
    logits = model(inputs)[0]
    probs = tf.nn.sigmoid(logits)
    pred = tf.where(probs > 0.5, 1, 0).numpy()[0]
    return np.where(pred == 1)[0].tolist()
