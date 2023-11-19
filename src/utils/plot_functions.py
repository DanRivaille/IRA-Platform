import matplotlib.pyplot as plt
import numpy as np


def plot_signal(signal: np.ndarray, size: tuple = (3, 15), save: bool = True, filename: str = "") -> None:
    fig, axs = plt.subplots()
    fig.set_figheight(size[0])
    fig.set_figwidth(size[1])
    axs.set_title("Signal")
    axs.plot(signal, color='C0')
    axs.set_xlabel("Time")
    axs.set_ylabel("Amplitude")

    if save:
        plt.savefig('/home/ivan.santos/repositories/IRA-Platform/' + filename, bbox_inches='tight')
    else:
        plt.show()


def plot_training_curves(train_error: list,
                         validation_error: list,
                         size: tuple = (3, 15),
                         save: bool = True,
                         filename: str = "") -> None:
    fig, axs = plt.subplots()
    fig.set_figheight(size[0])
    fig.set_figwidth(size[1])
    axs.plot(train_error, label='Train error')
    axs.plot(validation_error, label='Validation error')
    axs.set_xlabel("Epochs")
    axs.set_ylabel("Error (MSE)")

    plt.legend()
    if save:
        plt.savefig('/home/ivan.santos/repositories/IRA-Platform/' + filename, bbox_inches='tight')
    else:
        plt.show()
