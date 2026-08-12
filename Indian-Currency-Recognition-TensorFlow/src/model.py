import tensorflow as tf

from tensorflow.keras import layers

from src.config import (
    NUM_CLASSES,
    IMAGE_SIZE
)



def create_model():


    # Data Augmentation

    data_augmentation = tf.keras.Sequential([

        layers.RandomRotation(
            0.1
        ),

        layers.RandomZoom(
            0.1
        ),

        layers.RandomContrast(
            0.1
        )

    ])



    # EfficientNet Base

    base_model = tf.keras.applications.EfficientNetB0(

        weights="imagenet",

        include_top=False,

        input_shape=(

            IMAGE_SIZE,

            IMAGE_SIZE,

            3

        )

    )



    # Enable fine tuning

    base_model.trainable = False



    




    inputs = layers.Input(

        shape=(

            IMAGE_SIZE,

            IMAGE_SIZE,

            3

        )

    )



    x = data_augmentation(inputs)



    x = tf.keras.applications.efficientnet.preprocess_input(
        x
    )



    x = base_model(

        x,

        training=True

    )



    x = layers.GlobalAveragePooling2D()(x)



    x = layers.Dropout(
        0.4
    )(x)



    outputs = layers.Dense(

        NUM_CLASSES,

        activation="softmax"

    )(x)



    model = tf.keras.Model(

        inputs,

        outputs

    )


    return model
  
