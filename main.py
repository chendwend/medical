import os
# from pathlib import Path
# from tqdm import tqdm

import hydra
import torch
import torch.optim as optim

from src.dataloaders import get_dataloaders
from src.model import CustomResNet
# from src.plot import plot_results
# from src.train import train_loop
# from src.utils import chngdir
# from src.validation import validation_loop
from src.Trainer import Trainer

# from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup() -> None:

    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))



@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg):
    # chngdir()
    os.system("clear")
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        ddp_setup()
    torch.manual_seed(cfg.seed+ddp)
        # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


        # print(f"Computation device: {DEVICE}\n")

    train_loader, val_loader, test_loader = get_dataloaders(cfg, cfg.task, cfg.testing)


    model = CustomResNet(num_classes=cfg["classes_per_task"][cfg.task], model_name="resnet50")

        # model.to(DEVICE)

    loss = torch.nn.CrossEntropyLoss(label_smoothing=cfg.hp.label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=cfg.hp.lr)
        
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=6, min_lr=1e-5)

    max_epochs = cfg.hp.epochs
    if cfg["testing"]:
        max_epochs = 2

    trainer = Trainer("pathology", 
                    model, 
                    train_loader, 
                    val_loader, 
                    optimizer,
                    loss,
                    scheduler,
                    cfg["class_names"][cfg.task],
                    max_epochs,
                    cfg.hp.lr
                    )
    trainer.train_model()
    if ddp:
        destroy_process_group()
    # train_loss_hist, val_loss_hist = [], []
    # val_accuracy_history, train_accuracy_history = [], []
    # train_f1_hist, val_f1_hist = [], []

    # best_model, best_epoch, best_acc, f1_at_best_val = None, 0, 0, 0

    
    # num_epochs = 2 if cfg.testing else cfg.hp.epochs
    # for epoch in tqdm(range(1, num_epochs+1), total=num_epochs):
    #     (
    #         train_avg_f1,
    #         train_loss, 
    #         train_accuracy 
    #     ) = train_loop(model, train_loader, loss, optimizer, cfg["classes_per_task"][cfg.task], DEVICE)

    #     train_f1_hist.append(train_avg_f1)
    #     train_loss_hist.append(train_loss)
    #     train_accuracy_history.append(train_accuracy)

    #     (
    #         val_avg_f1,
    #         val_loss,
    #         val_accuracy,
    #         val_true_labels,
    #         val_prediced_labels,
    #     ) = validation_loop(model, val_loader, loss, cfg["classes_per_task"][cfg.task], DEVICE)

    #     val_loss_hist.append(val_loss)
    #     val_accuracy_history.append(val_accuracy)
    #     val_f1_hist.append(val_avg_f1)

    #     scheduler.step(val_loss)
    #     new_lr = scheduler.get_last_lr()[0]
    #     if new_lr < lr:
    #         lr = new_lr
    #         print(f"lr updated -> {new_lr}.")


    #     if val_accuracy > best_acc:
    #         best_acc = val_accuracy
    #         best_model = model
    #         best_epoch = epoch
    #         f1_at_best_val = val_avg_f1
    #         save_checkpoint(best_model.state_dict(), cfg.task, "resnet50")
    #         print("\n-------new best:-------")

    #     epoch_summary = ", ".join(
    #         [
    #             f"[Epoch {epoch}/{num_epochs}]:",
    #             f"Train loss {train_loss_hist[-1]:.2f}",
    #             f"Val accuracy {val_accuracy:.2f}%",
    #             f"Val average F1 score: {val_avg_f1:.2f}",
    #         ]
    #     )
    #     print("\n" + epoch_summary + "\n")

    #     torch.cuda.empty_cache()

    # summary = ", ".join(
    #     ["-" * 30, f"validation accuray: {val_accuracy:.2f}", f"f1 score: {f1_at_best_val}"]
    # )
    # print("\n" + summary + "\n")

    # results = {
    #     "loss": (train_loss_hist, val_loss_hist),
    #     "accuracy": (train_accuracy_history, val_accuracy_history, (best_epoch, best_acc)),
    #     "f1": (train_f1_hist, val_f1_hist),
    #     "val_true_labels": (val_true_labels),
    #     "val_predicted_labels": (val_prediced_labels)
    # }

    # plot_results(cfg, results, cfg.task)

# def save_checkpoint(state, task, model_name):
#     Path("best_models").mkdir(exist_ok=True)
#     filename=Path(f"best_models/{task}/{model_name}.pth")
#     filename.parent.mkdir(exist_ok=True)
#     print("=> Saving a new best model")
#     torch.save(state, str(filename))


if __name__ == "__main__":
    os.system('clear')
    main()
