import torch
from tqdm import tqdm

from src.metrics import calc_accuracy, calc_conf_per_class, calc_f1


def validation_loop(model, val_loader, loss_function, num_classes, device):
    model.eval()

    val_total, val_correct, val_running_loss = 0, 0, 0
    with torch.no_grad():
        tp = torch.zeros(num_classes, device=device)
        fp = torch.zeros(num_classes, device=device)
        fn = torch.zeros(num_classes, device=device)

        true_labels = []
        predicted_labels = []

        for _, (inputs, labels) in tqdm(enumerate(val_loader), total=len(val_loader)):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )
            outputs = model(inputs)
            val_running_loss += loss_function(outputs, labels).item()

            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

            for class_label in range(num_classes):
                true_pos, false_pos, false_neg = calc_conf_per_class(
                    class_label, predicted, labels
                )
                tp[class_label] += true_pos
                fp[class_label] += false_pos
                fn[class_label] += false_neg

        f1 = calc_f1(tp, fp, fn)
        avg_f1 = f1.mean().item()

        avg_val_loss = val_running_loss / len(val_loader)  # divide by no. of batches
        val_accuracy = calc_accuracy(val_correct, val_total)

    return avg_f1, avg_val_loss, val_accuracy, true_labels, predicted_labels
