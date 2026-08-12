import os


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "split"
)


MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_currency_model.keras"
)



IMAGE_SIZE = 224


BATCH_SIZE = 16


EPOCHS = 15


LEARNING_RATE = 0.0001



CLASS_NAMES = [

    "fifty_new",
    "fifty_old",
    "five_hundred",
    "hundred_new",
    "hundred_old",
    "ten_new",
    "ten_old",
    "twenty_new",
    "twenty_old",
    "two_hundred",
    "two_thousand"

]


NUM_CLASSES = len(CLASS_NAMES)
