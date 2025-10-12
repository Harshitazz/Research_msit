# text_sentiment/train.py
from transformers import BertTokenizerFast, TFBertForSequenceClassification
import tensorflow as tf
from datasets import load_dataset

def train():
    dataset = load_dataset("go_emotions", "simplified")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=64)

    tokenized = dataset.map(tokenize, batched=True)
    tokenized.set_format(type='tensorflow', columns=['input_ids', 'attention_mask', 'labels'])

    features = {x: tokenized['train'][x] for x in ['input_ids','attention_mask']}
    tf_train = tf.data.Dataset.from_tensor_slices((features, tokenized['train']['labels'])).batch(32)

    model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=28)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(tf_train, epochs=3)

    model.save_pretrained("models/bert_emotion")
    tokenizer.save_pretrained("models/bert_emotion")

if __name__ == "__main__":
    train()
