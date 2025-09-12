import os
import tensorflow as tf
import json
from model_utils import build_model

EPOCHS = 10
BATCH_SIZE = 64
IMG_SIZE = (48, 48)
DATA_DIR = "fer2013"
SAVE_DIR = "models"

def load_datasets():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(DATA_DIR, "train"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(DATA_DIR, "test"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )
    return train_ds, val_ds

def train_model(arch, train_ds, val_ds):
    print(f"\n Training {arch}...")
    model = build_model(arch, input_shape=(48, 48, 3), num_classes=7)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    loss, acc = model.evaluate(val_ds)
    model.save(os.path.join(SAVE_DIR, f"{arch}.h5"))
    print(f" {arch} Accuracy: {acc:.4f}")
    return acc

if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    train_ds, val_ds = load_datasets()
    architectures = ['mobilenetv2', 'resnet50', 'xception']
    results = {}

    for arch in architectures:
        acc = train_model(arch, train_ds, val_ds)
        results[arch] = float(acc)

    with open('facial_fer/results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("\n Accuracy Comparison:")
    for name, acc in results.items():
        print(f"{name}: {acc:.4f}")
