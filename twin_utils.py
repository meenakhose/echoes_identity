import cv2
import numpy as np

INPUT_SHAPE = (224, 224)

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, INPUT_SHAPE)
    img = img.astype("float32") / 255.0
    return img

def predict_twin_similarity(img1_path, img2_path, feature_extractor, siamese_model):
    img1 = preprocess_image(img1_path)
    img2 = preprocess_image(img2_path)

    img1_features = feature_extractor.predict(np.expand_dims(img1, axis=0))
    img2_features = feature_extractor.predict(np.expand_dims(img2, axis=0))

    similarity = siamese_model.predict([img1_features, img2_features])[0][0]
    return similarity