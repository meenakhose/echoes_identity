from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model

def load_feature_extractor():
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    return Model(inputs=base_model.input, outputs=base_model.output)

def load_siamese_model(model_path="saved_model/siamese_model.h5"):
    return load_model(model_path, compile=False)