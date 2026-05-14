import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input


def predict_image(config, weights, image_path, class_names):

    model = tf.keras.models.load_model(
        weights,
        custom_objects={"preprocess_input": preprocess_input},
        safe_mode=False
    )

    img = image.load_img(
        image_path,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)[0]

    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]

    return {
        "predicted_class": predicted_class,
        "confidence": float(predictions[predicted_index]),
        "probabilities": {
            class_names[i]: float(predictions[i])
            for i in range(len(class_names))
        }
    }