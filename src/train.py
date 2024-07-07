import numpy as np
import torch
from tqdm import tqdm

from src.metrics import calc_accuracy, calc_conf_per_class, calc_f1



def train_loop(model, train_loader, criterion, optimizer, num_classes, device):
    model.train()

    train_running_loss = 0
    train_total_samples = 0
    train_correct = 0
    loss = 0

    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)

    for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, total=len(train_loader))):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs, 1)

        train_running_loss += loss.item()
        train_total_samples += labels.size(0)
        train_correct += (predicted == labels).sum().item()


        for class_label in range(num_classes):
            true_pos, false_pos, false_neg = calc_conf_per_class(
                class_label, predicted, labels
            )
            tp[class_label] += true_pos
            fp[class_label] += false_pos
            fn[class_label] += false_neg

    loss = train_running_loss / len(train_loader)
    accuracy = calc_accuracy(train_correct, train_total_samples)
    f1 = calc_f1(tp, fp, fn)
    avg_f1 = f1.mean().item()

    return avg_f1, loss, accuracy
