import tensorflow as tf
import numpy as np
from PIL import Image

from src.config import (
    MODEL_PATH,
    IMAGE_SIZE,
    CLASS_NAMES
)



def predict_image(image_path):


    model = tf.keras.models.load_model(
        MODEL_PATH
    )


    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )


    image = image.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    )


    image_array = np.array(
        image
    )


    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    predictions = model.predict(
        image_array,
        verbose=0
    )


    predicted_index = np.argmax(
        predictions
    )


    confidence = np.max(
        predictions
    )


    predicted_class = CLASS_NAMES[
        predicted_index
    ]


    return (
        predicted_class,
        confidence * 100
    )



if __name__ == "__main__":


    image_path = "sample.jpg"


    result, confidence = predict_image(
        image_path
    )


    print("====================")

    print("Prediction:")

    print(result)


    print(
        f"Confidence: {confidence:.2f}%"
    )

    print("====================")
