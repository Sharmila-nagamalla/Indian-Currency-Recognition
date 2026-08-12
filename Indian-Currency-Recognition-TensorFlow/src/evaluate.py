import os
import numpy as np
import tensorflow as tf


from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


import matplotlib.pyplot as plt
import seaborn as sns


from src.data_loader import load_data

from src.config import MODEL_PATH



def evaluate():

    print("Loading Test Dataset...")


    _, _, test_dataset, class_names = load_data()



    print("Loading Model...")


    model = tf.keras.models.load_model(
        MODEL_PATH
    )


    print("Model Loaded Successfully")



    y_true = []

    y_pred = []



    for images, labels in test_dataset:


        predictions = model.predict(
            images,
            verbose=0
        )


        predicted_classes = np.argmax(
            predictions,
            axis=1
        )


        y_pred.extend(
            predicted_classes
        )


        y_true.extend(
            labels.numpy()
        )



    # Accuracy

    accuracy = accuracy_score(
        y_true,
        y_pred
    )


    print("\n==============================")

    print("Test Accuracy:")

    print(
        accuracy * 100
    )

    print("==============================")



    # Classification Report


    print("\nClassification Report:\n")


    print(

        classification_report(

            y_true,

            y_pred,

            target_names=class_names

        )

    )



    # Confusion Matrix


    cm = confusion_matrix(

        y_true,

        y_pred

    )



    os.makedirs(

        "outputs",

        exist_ok=True

    )



    plt.figure(

        figsize=(12,10)

    )



    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues",

        xticklabels=class_names,

        yticklabels=class_names

    )


    plt.xlabel(
        "Predicted"
    )


    plt.ylabel(
        "Actual"
    )


    plt.title(
        "Indian Currency Confusion Matrix"
    )


    plt.tight_layout()



    plt.savefig(

        "outputs/confusion_matrix.png"

    )


    plt.close()



    print(
        "\nConfusion Matrix Saved"
    )





if __name__ == "__main__":

    evaluate()
