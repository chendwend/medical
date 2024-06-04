import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.metrics import create_report


def moving_average(x, w=5):
    return np.convolve(x, np.ones(w), "valid") / w


def plot_graph(data: tuple, description: dict) -> None:
    plt.figure()
    # if len(data) > 1:
    for i, data in enumerate(data):
        # plt.plot(data, label=description["label"][i])
        plt.plot(moving_average(data), label=description["label"][i])
    # else:
    #     plt.plot(data[0])

    plt.legend()
    plt.title(description["title"])
    plt.xlabel(description["xlabel"])
    plt.ylabel(description["ylabel"])
    plt.savefig(description["savefig"])
    plt.show(block=False)
    # input("Press Enter to close the plot...")


def plot_conf_matrix(conf_matrix, class_names) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show(block=False)
    input("Press Enter to close the plot...")


def plot_model_graphs(results: dict) -> None:
    fig, ax = plt.subplots(3, 1)
    ax[0].plot(results[0], label="Train loss")
    ax[0].plot(results[1], label="Val loss")
    ax[0].set_title("Loss")
    ax[0].set(xlabel="Epoch", ylabel="loss")

    ax[1].plot(results[2], label="Train Accuracy")
    ax[1].plot(results[3], label="Val Accuracy")
    ax[1].set_title("Accuracy")
    ax[1].set(xlabel="Epoch", ylabel="Accuracy [%]")

    ax[2].plot(results[4])
    ax[2].set_title("F1 Score")
    ax[2].set(xlabel="Epoch", ylabel="F1 Score")

    fig.savefig(results["savefig"])
    plt.show(block=False)
    input("Press Enter to close the plot...")


def plot_results(cfg, results, task):
    desc = {
        "title": "Loss Function",
        "xlabel": "Epoch",
        "ylabel": "Loss",
        "savefig": f"{cfg.folders.loss}",
        "label": ["Train", "Val"],
    }
    plot_graph(results["loss"], desc)

    desc = {
        "title": "Accuracy",
        "xlabel": "Epoch",
        "ylabel": "Accuracy",
        "savefig": f"{cfg.folders.accuracy}",
        "label": ["Train", "Val"],
    }

    plot_graph(results["accuracy"], desc)

    desc = {
        "title": "Average F1 Score",
        "xlabel": "Epoch",
        "ylabel": "F1 score",
        "savefig": f"{cfg.folders.f1}",
        "label": ["Train", "Val", "Best F1"],
    }
    plot_graph(results["f1"], desc)

    create_report(
        results["val_true_labels"],
        results["val_predicted_labels"],
        cfg["class_names"][task],
        cfg.folders.report,
    )
