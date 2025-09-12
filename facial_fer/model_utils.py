from tensorflow.keras.applications import MobileNetV2, ResNet50, Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.models import Model

def build_model(arch='mobilenetv2', input_shape=(48, 48, 3), num_classes=7):
    input_layer = Input(shape=input_shape)
    if arch == 'mobilenetv2':
        base = MobileNetV2(include_top=False, input_tensor=input_layer, weights=None)
    elif arch == 'resnet50':
        base = ResNet50(include_top=False, input_tensor=input_layer, weights=None)
    elif arch == 'xception':
        base = Xception(include_top=False, input_tensor=input_layer, weights=None)
    else:
        raise ValueError("Unsupported architecture.")

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=input_layer, outputs=output)
