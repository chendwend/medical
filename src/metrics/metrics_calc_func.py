import numpy as np
from sklearn.metrics import classification_report


def accuracy(correct, total):
    return 100 * correct / total


def precision(tp, fp, epsilon=1e-7):
    return tp / (tp + fp + epsilon)


def recall(tp, fn, epsilon=1e-7):
    return tp / (tp + fn + epsilon)


def calc_conf_per_class(class_label, predicted, actual):
    tp = ((predicted == class_label) & (actual == class_label)).sum()
    fp = ((predicted == class_label) & (actual != class_label)).sum()
    fn = ((predicted != class_label) & (actual == class_label)).sum()

    return tp, fp, fn

 
def f1(recall, precision, epsilon=1e-7) -> np.array:
    return 2 * precision * recall / (precision + recall + epsilon)


def create_report(y_true, y_pred, class_names, path):
    # import matplotlib

    # matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    report_string = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=False
    )

    report_lines = report_string.split("\n")

    # Extract macro F1-score, accuracy, and weighted F1-score
    macro_f1_score = float(report_lines[-3].split()[4])
    accuracy = float(report_lines[-4].split()[1])
    weighted_f1_score = float(report_lines[-2].split()[4])

    # Create a styled visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")  # Turn off axis labels and ticks

    table_data = []
    for line in report_lines[2:-5]:  # Exclude headers and support info lines
        row = line.split()
        table_data.append(row)

    column_headers = ["", "precision", "recall", "f1-score", "support"]
    ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))
    table = ax.table(
        cellText=table_data,
        colLabels=column_headers,
        loc="center",
    )
    # Add macro F1-score, accuracy, and weighted F1-score rows
    table_data.append(["Macro Avg", "", "", macro_f1_score, ""])
    table_data.append(["Accuracy", "", "", accuracy, ""])
    table_data.append(["Weighted Avg", "", "", weighted_f1_score, ""])

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)  # Adjust cell size for better readability

    plt.title("Classification Report", fontsize=16)

    # Save the visualization as an image (e.g., PNG)

    plt.savefig(path, bbox_inches="tight", pad_inches=0.2, dpi=150)
    # plt.show()
