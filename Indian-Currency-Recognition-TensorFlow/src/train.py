import tensorflow as tf
import os


from src.data_loader import load_data

from src.model import create_model
from src.utils import save_training_graphs
from src.config import (
    MODEL_PATH,
    EPOCHS,
    LEARNING_RATE
)



train_ds, val_ds, _, classes = load_data()



model = create_model()



model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=LEARNING_RATE

    ),


    loss="sparse_categorical_crossentropy",


    metrics=[

        "accuracy"

    ]

)



model.summary()



os.makedirs(

    "models",

    exist_ok=True

)



checkpoint = tf.keras.callbacks.ModelCheckpoint(

    MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    mode="max",

    verbose=1

)



early_stop = tf.keras.callbacks.EarlyStopping(

    monitor="val_accuracy",

    patience=5,

    restore_best_weights=True,

    verbose=1

)



reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.2,

    patience=3,

    min_lr=1e-7,

    verbose=1

)



history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    callbacks=[

        checkpoint,

        early_stop,

        reduce_lr

    ]

)

save_training_graphs(history)

print("\n==============================")

print("Training Completed")

print("==============================")
