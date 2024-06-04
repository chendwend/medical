from os import system
from pathlib import Path

import hydra
import torch
import torch.optim as optim

from src.load_dataset import load_dataset
from src.model import CustomResNet
from src.plot import plot_results
from src.train import train_loop
from src.utils import chngdir
from src.validation import validation_loop


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg):
    system('clear')
    chngdir()
    torch.manual_seed(cfg.seed)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Computation device: {DEVICE}\n")

    train_loader, val_loader, test_loader = load_dataset(cfg, cfg.task, cfg.testing)

    model = CustomResNet(num_classes=cfg["classes_per_task"][cfg.task])

    loss = torch.nn.CrossEntropyLoss(label_smoothing=0.25)
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.hp.lr)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=0.00001)

    train_loss_hist, val_loss_hist = [], []
    val_accuracy_history, train_accuracy_history = [], []
    train_f1_hist, val_f1_hist = [], []

    best_model, best_f1, val_at_best_f1 = None, 0, 0

    lr = cfg.hp.lr
    for epoch in range(cfg.hp.epochs):
        train_avg_f1, train_loss, train_accuracy = train_loop(
            model, train_loader, loss, optimizer, cfg["classes_per_task"][cfg.task], DEVICE
        )

        train_f1_hist.append(train_avg_f1)
        train_loss_hist.append(train_loss)
        train_accuracy_history.append(train_accuracy)

        (
            val_avg_f1,
            val_loss,
            val_accuracy,
            val_true_labels,
            val_prediced_labels,
        ) = validation_loop(model, val_loader, loss, cfg["classes_per_task"][cfg.task], DEVICE)

        val_loss_hist.append(val_loss)
        val_accuracy_history.append(val_accuracy)
        val_f1_hist.append(val_avg_f1)

        scheduler.step(val_loss)
        new_lr = scheduler.get_last_lr()[0]
        if new_lr < lr:
            lr = new_lr
            print(f"lr updated -> {scheduler.get_last_lr()[0]}.")

        epoch_summary = ", ".join(
            [
                f"[Epoch {epoch+1}/{cfg.hp.epochs}]:",
                f"Train loss {train_loss_hist[-1]:.2f}",
                f"Val accuracy {val_accuracy:.2f}%",
                f"Val average F1 score: {val_avg_f1:.2f}",
            ]
        )
        print("\n" + epoch_summary + "\n")

        if val_avg_f1 > best_f1:
            best_f1 = val_avg_f1
            best_model = model
            val_at_best_f1 = val_accuracy

    summary = ", ".join(
        ["-" * 30, f"best f1: {best_f1:.2f}", f"validation accuray: {val_at_best_f1}"]
    )
    print("\n" + summary + "\n")

    results = {
        "loss": (train_loss_hist, val_loss_hist),
        "accuracy": (train_accuracy_history, val_accuracy_history),
        "f1": (train_f1_hist, val_f1_hist),
        "val_true_labels": (val_true_labels),
        "val_predicted_labels": (val_prediced_labels),
        "best_f1": best_f1,
    }

    plot_results(cfg, results, cfg.task)


def save_checkpoint(state, task):
    filename=Path(f"best_models/{task}")
    filename.parent.mkdir(exist_ok=True)
    print("=> Saving a new best model")
    torch.save(state, str(filename))


if __name__ == "__main__":
    main()
