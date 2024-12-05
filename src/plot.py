from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from metrics.metrics_calc_func import create_report

# import seaborn as sns



def moving_average(x, w=5):
    return np.convolve(x, np.ones(w), "valid") / w


def plot_graph(data: tuple, description: dict) -> None:
    plt.figure()
    # if len(data) > 1:
    for i, data in enumerate(data):
        # plt.plot(data, label=description["label"][i])
        if i == 2:
            plt.plot(data[0], data[1], label=description["label"][i])
        else:
            plt.plot(moving_average(data), label=description["label"][i])
    # else:
    #     plt.plot(data[0])

    plt.legend()
    plt.title(description["title"])
    plt.xlabel(description["xlabel"])
    plt.ylabel(description["ylabel"])
    plt.savefig(description["savefig"])
    # plt.show(block=False)
    # input("Press Enter to close the plot...")


# def plot_conf_matrix(conf_matrix, class_names) -> None:
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(
#         conf_matrix,
#         annot=True,
#         fmt="d",
#         cmap="Blues",
#         xticklabels=class_names,
#         yticklabels=class_names,
#     )
#     plt.xlabel("Predicted Labels")
#     plt.ylabel("True Labels")
#     plt.title("Confusion Matrix")
#     plt.savefig(description["savefig"])
#     # plt.show(block=False)
#     # input("Press Enter to close the plot...")


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
    # plt.show(block=False)
    # input("Press Enter to close the plot...")


def plot_results(output_folder, results, task, class_names):
    task_folder = Path(output_folder)/task/"graphs"
    task_folder.mkdir(parents=True, exist_ok=True)
    desc = {
        "title": "Loss Function",
        "xlabel": "Epoch",
        "ylabel": "Loss",
        "savefig": str(task_folder/"loss.png"),
        "label": ["Train", "Val"],
    }
    plot_graph(results["loss"], desc)

    desc = {
        "title": "Accuracy",
        "xlabel": "Epoch",
        "ylabel": "Accuracy",
        "savefig": str(task_folder/"accuracy.png"),
        "label": ["Train", "Val", "best acc"],
    }
    plot_graph(results["accuracy"], desc)

    desc = {
        "title": "Average F1 Score",
        "xlabel": "Epoch",
        "ylabel": "F1 score",
        "savefig": str(task_folder/"f1.png"),
        "label": ["Train", "Val"],
    }
    plot_graph(results["f1"], desc)

    create_report(
        results["val_true_labels"],
        results["val_predicted_labels"],
        class_names,
        str(task_folder/"classification_report.png"),
    )
