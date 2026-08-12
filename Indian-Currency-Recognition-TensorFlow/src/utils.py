import matplotlib.pyplot as plt
import os



def save_training_graphs(history):


    os.makedirs(
        "outputs",
        exist_ok=True
    )


    # Accuracy

    plt.figure()


    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )


    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )


    plt.title(
        "Accuracy Curve"
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Accuracy"
    )


    plt.legend()


    plt.savefig(
        "outputs/accuracy_curve.png"
    )


    plt.close()



    # Loss

    plt.figure()


    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )


    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )


    plt.title(
        "Loss Curve"
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Loss"
    )


    plt.legend()


    plt.savefig(
        "outputs/loss_curve.png"
    )


    plt.close()
