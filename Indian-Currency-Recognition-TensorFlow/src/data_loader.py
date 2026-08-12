import tensorflow as tf

from src.config import (
    DATA_DIR,
    IMAGE_SIZE,
    BATCH_SIZE
)



def load_data():


    train_path = DATA_DIR + "/train"

    val_path = DATA_DIR + "/validation"

    test_path = DATA_DIR + "/test"



    train_dataset = tf.keras.utils.image_dataset_from_directory(

        train_path,

        image_size=(
            IMAGE_SIZE,
            IMAGE_SIZE
        ),

        batch_size=BATCH_SIZE,

        shuffle=True
    )



    val_dataset = tf.keras.utils.image_dataset_from_directory(

        val_path,

        image_size=(
            IMAGE_SIZE,
            IMAGE_SIZE
        ),

        batch_size=BATCH_SIZE,

        shuffle=False
    )



    test_dataset = tf.keras.utils.image_dataset_from_directory(

        test_path,

        image_size=(
            IMAGE_SIZE,
            IMAGE_SIZE
        ),

        batch_size=BATCH_SIZE,

        shuffle=False
    )



    class_names = train_dataset.class_names



    AUTOTUNE = tf.data.AUTOTUNE


    train_dataset = train_dataset.prefetch(
        AUTOTUNE
    )

    val_dataset = val_dataset.prefetch(
        AUTOTUNE
    )

    test_dataset = test_dataset.prefetch(
        AUTOTUNE
    )


    return (
        train_dataset,
        val_dataset,
        test_dataset,
        class_names
    )
