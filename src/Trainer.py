from datetime import datetime
from os import environ
from pathlib import Path
from typing import Tuple

import torch
import torch.distributed as dist
from omegaconf import DictConfig
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from metrics import metrics_calc_func
from metrics.metrics import Metrics
from utils import ddp_utils, general_utils, torch_utils


class Trainer():

    def __init__(
        self,
        task: str,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        loss_func, 
        class_labels: list, 
        hp_dict: dict,
        patience: int,
        world_size: int,
        master_process: bool,
        testing: bool,
    ) -> None:
        self.rank = int(environ["LOCAL_RANK"])
        self.model = model
        self.train_data = train_data
        self.validation_data = val_data
        self.loss_func = loss_func
        self.epochs_run = 0
        self.hp_dict = hp_dict
        self.stopper, self.stop = torch_utils.EarlyStopping(patience), False
        self.world_size = world_size
        self.master_process = master_process
        self.testing = testing
        self._best, self._is_best = torch_utils.BestModel(), True
        self.task = task
        self.class_names = class_labels
        self.num_classes = len(class_labels)
        self._device = torch.device(f'cuda:{self.rank}')
             
        # self.snapshot_path = Path(snapshot_path)
        # if self.snapshot_path.exists():
        #     print("Loading snapshot")
        #     self._load_snaphot(self.snapshot_path)
    
    def _load_snaphot(self, snapshot_path: Path) -> None:
        loc = f"cuda:{self.rank}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")
    
    def _train_loop(self) -> Metrics:
        total_samples, accumulated_correct, accumulated_loss = 0, 0, 0
        global_epoch_loss = 0

        tp = torch.zeros(self.num_classes, device=self._device)
        fp = torch.zeros(self.num_classes, device=self._device)
        fn = torch.zeros(self.num_classes, device=self._device)

        self.model.train()

        for inputs, targets in self.train_data:
            inputs, targets = inputs.to(self._device, non_blocking=True), targets.to(self._device, non_blocking=True)    
            predicted = self.model(inputs)
            loss = self.loss_func(predicted, targets)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            _, predicted = torch.max(predicted, 1)

            accumulated_loss += loss.item()*targets.size(0)
            total_samples += targets.size(0)
            accumulated_correct += (predicted == targets).sum().item()

            targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=self.num_classes)
            predicted_one_hot = torch.nn.functional.one_hot(predicted, num_classes=self.num_classes)

            # tensor arrays of num_classes size, each entry is an array of sum on all samples
            tp += (predicted_one_hot & targets_one_hot).sum(dim=0) 
            fp += (predicted_one_hot & ~targets_one_hot).sum(dim=0) 
            fn += (~predicted_one_hot & targets_one_hot).sum(dim=0) 
            
        total_accumulated_correct = torch.tensor([accumulated_correct], device=self._device)
        total_accumulated_loss = torch.tensor([accumulated_loss], device=self._device)
        total_total_samples = torch.tensor([total_samples], device=self._device)
        # Reduce results across all processes
        dist.reduce(total_accumulated_loss, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(total_total_samples, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(total_accumulated_correct, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(tp, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(fp, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(fn, dst=0, op=dist.ReduceOp.SUM)

        if self.master_process:
            global_epoch_loss = total_accumulated_loss.item() / total_total_samples.item()
            accuracy = metrics_calc_func.accuracy(total_accumulated_correct.item(), total_total_samples.item())   
            recall = metrics_calc_func.recall(tp.cpu(), fn.cpu())
            precision = metrics_calc_func.precision(tp.cpu(), fp.cpu())
            f1 = metrics_calc_func.f1(recall, precision)
            avg_f1 = 100 * f1.mean().item()
        else:
            global_epoch_loss, accuracy, avg_f1 = None, None, None


        metrics = {
            "loss": global_epoch_loss,
            "accuracy": accuracy,
            "f1": avg_f1,
            "recall": recall,
            "precision": precision
        }

        return metrics

    def _validation_loop(self) -> Tuple[Metrics, list]:
        total_samples, accumulated_correct, accumulated_loss = 0, 0, 0
        global_epoch_loss = 0

        tp = torch.zeros(self.num_classes, device=self._device)
        fp = torch.zeros(self.num_classes, device=self._device)
        fn = torch.zeros(self.num_classes, device=self._device)

        true_labels = []
        predicted_labels = []

        with torch.no_grad():
            self.model.eval()
            for (inputs, targets) in self.validation_data:
                inputs, targets = inputs.to(self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
                predicted = self.model(inputs)
                loss = self.loss_func(predicted, targets)
                
                _, predicted = torch.max(predicted, 1)

                accumulated_loss += loss.item()*targets.size(0)
                total_samples += targets.size(0)
                accumulated_correct += (predicted == targets).sum().item()

                true_labels.extend(targets.cpu().numpy()) 
                predicted_labels.extend(predicted.cpu().numpy())

                targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=self.num_classes)
                predicted_one_hot = torch.nn.functional.one_hot(predicted, num_classes=self.num_classes)

                # tensor arrays of num_classes size, each entry is an array of sum on all samples
                tp += (predicted_one_hot & targets_one_hot).sum(dim=0) 
                fp += (predicted_one_hot & ~targets_one_hot).sum(dim=0) 
                fn += (~predicted_one_hot & targets_one_hot).sum(dim=0) 
        
            total_accumulated_correct = torch.tensor([accumulated_correct], device=self._device) #TODO: reduce uneccesary GPU <-> CPU traffic
            total_accumulated_loss = torch.tensor([accumulated_loss], device=self._device)
            total_total_samples = torch.tensor([total_samples], device=self._device)
            # Reduce results across all processes
            dist.reduce(total_accumulated_loss, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(total_total_samples, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(total_accumulated_correct, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(tp, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(fp, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(fn, dst=0, op=dist.ReduceOp.SUM)

            if self.master_process:
                global_epoch_loss = total_accumulated_loss.item() / total_total_samples.item()
                accuracy = metrics_calc_func.accuracy(total_accumulated_correct.item(), total_total_samples.item())
                recall = metrics_calc_func.recall(tp.cpu(), fn.cpu())
                precision = metrics_calc_func.precision(tp.cpu(), fp.cpu())
                f1 = metrics_calc_func.f1(recall, precision)
                avg_f1 = 100 * f1.mean().item()

                # Gather true_labels and predicted_labels from all GPUs
                gathered_true_labels = [None for _ in range(dist.get_world_size())]
                gathered_predicted_labels = [None for _ in range(dist.get_world_size())]
            else:
                global_epoch_loss, accuracy, avg_f1 = None, None, None
                gathered_true_labels = None
                gathered_predicted_labels = None

        dist.gather_object(true_labels, gathered_true_labels, dst=0)
        dist.gather_object(predicted_labels, gathered_predicted_labels, dst=0)

        if self.master_process:
            combined_true_labels = []
            combined_predicted_labels = []
            for sublist in gathered_true_labels:
                combined_true_labels.extend(sublist)
            for sublist in gathered_predicted_labels:
                combined_predicted_labels.extend(sublist)

            class_report = classification_report(
                combined_true_labels,
                combined_predicted_labels, 
                target_names=self.class_names).splitlines()
        else:
            combined_true_labels, combined_predicted_labels = None, None
            class_report = None

        metrics = {
            "loss": global_epoch_loss,
            "accuracy": accuracy,
            "f1": avg_f1,
            "recall": recall,
            "precision": precision
        }

        return metrics, class_report
      
    def _log_metrics(self, epoch:int, train_metrics: Metrics, val_metrics: Metrics):
        if not(self.testing):
            wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss": train_metrics["loss"], 
                        "val/loss": val_metrics["loss"],
                        "train/accuracy": train_metrics["accuracy"],
                        "val/accuracy": val_metrics["accuracy"],
                        "train/f1": train_metrics["f1"],
                        "val/f1": val_metrics["f1"],
                        "lr": self.scheduler.get_last_lr()[0]
                    }
                )
            
    def _update_lr(self, loss):
        cur_lr = self.scheduler.get_last_lr()[0]
        self.scheduler.step(loss)
        new_lr = self.scheduler.get_last_lr()[0]
        if new_lr != cur_lr and self.master_process:
            print(f"\nlr updated ---------------> {new_lr}.\n")

    @general_utils.time_it
    def _run_epoch(self, epoch:int) -> Tuple[Metrics, Metrics, list]:
  
        self.train_data.sampler.set_epoch(epoch)

        train_metrics = self._train_loop()
        
        (
            val_metrics,
            class_report
        ) = self._validation_loop()

        # broadcast val loss and update lr on all GPUs
        val_loss = ddp_utils.broadcast_object(val_metrics["loss"], self.master_process)
        self._update_lr(val_loss)

        if self.master_process:
            self._log_metrics(epoch, train_metrics, val_metrics)
            return train_metrics, val_metrics, class_report
        
        return (None,)*3
    
    def _set_optimizer_scheduler(self) -> None:
        import torch.optim as optim

        beta1 = 0.9
        beta2 = 0.999
        
        self.optimizer = optim.AdamW(params=
                                        [
        {"params": [p for n, p in self.model.named_parameters() if "bn" in n and p.requires_grad], "weight_decay": 0.0},
        {"params": [p for n, p in self.model.named_parameters() if "bn" not in n and p.requires_grad]},
                                        ],
                                     lr=self.hp_dict["lr"], 
                                     weight_decay=self.hp_dict["weight_decay"], 
                                     betas=(beta1, beta2))
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.8, patience=10, min_lr=1e-5)
        
    def train_model(self):

        self._set_optimizer_scheduler()
        max_epochs = 2 if self.testing else self.hp_dict["epochs"]
        # max_epochs = self.hp_dict["epochs"]

        for epoch in tqdm(range(1, max_epochs+1), total=max_epochs,  disable=(not(self.master_process)), colour='green'):
            (train_metrics, val_metrics, class_report), exec_time = self._run_epoch(epoch)
            if self.master_process and not self.testing:
                self.stop = self.stopper(val_metrics["accuracy"], epoch)
                self._is_best = self._best(val_metrics["accuracy"], val_metrics["f1"], class_report, epoch, self.model)
                if self._is_best:
                    self._save_snapshot(epoch, self.hp_dict, threshold=90.0, metric=val_metrics["accuracy"])
            
                epoch_summary = ", ".join(
                    [
                        f"[Epoch {epoch}/{self.hp_dict['epochs']}]:",
                        f"Train/val loss {train_metrics['loss']:.2f}|{val_metrics['loss']:.2f}",
                        f"Val F1 score: {val_metrics['f1']:.2f}%",
                        f"Val accuracy {val_metrics['accuracy']:.2f}|{self._best.accuracy:.2f}%",
                    ]
                )
                print("\n" + epoch_summary + "\n")

            # broadcast_list = [self.stop if self.master_process else None]
            self.stop = ddp_utils.broadcast_object(self.stop, self.master_process)
            # dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            # self.stop = broadcast_list[0]
            if self.stop:
                break # must break all DDP ranks

        # finished training -> move to reporting      
        if self.master_process:

            report_table = []
            for line in self._best.class_report[2:(len(self.class_names)+2)]:
                report_table.append(line.split())
            
            report_columns = ["Class", "Precision", "Recall", "F1-score", "Support"]
            wandb.log({
                    "Classification Report": wandb.Table(data=report_table, columns=report_columns)
                })

            summary = ", ".join(
                [
                    f"best val accuray {self._best.accuracy:.2f}", 
                    f"achieved at epoch {self._best.epoch}", 
                    f"with f1 score {self._best.f1:.2f}."
                ]
            )
            print("-" * 30)
            print("\n")
            print(summary + "\n")
            print("-" * 30)
            print("\n")
        
        return exec_time

    def _scheduler_step(self, val_loss):
        self.scheduler.step(val_loss)
        new_lr = self.scheduler.get_last_lr()[0]
        if new_lr < self.lr:
            self.lr = new_lr
            print(f"lr updated -> {new_lr}.")

    def _save_snapshot(self, epoch:int, hp_dict:dict, threshold:float, metric:float) -> None:
        if metric > threshold:

            date_time = datetime.now().strftime("%d_%m_%Y_%H_%M")

            param_string = "_".join([f"{k}={v}" for k, v in hp_dict.items()])
            Path("best_models").mkdir(exist_ok=True)
            Path(f"best_models/{self.task}").mkdir(exist_ok=True)
            results_folder = Path(f"best_models/{self.task}/{param_string}-{date_time}")
            model_path=Path(results_folder/"best_model.pt")
            class_rpt_path = Path(results_folder/"classification_report.txt")
            model_path.parent.mkdir(exist_ok=True)
            with open(class_rpt_path, "w") as file:
                file.write(self._best.class_report)

            snapshot = {
                "MODEL_STATE": self.model.module.state_dict(),
                "EPOCHS_RUN": epoch,
            }
            torch.save(snapshot, model_path)
            print(f"epoch {epoch} => Saving a new best model.")







def main(cfg:DictConfig):
    train_loader, val_loader, test_loader = get_dataloaders(
        cfg.folders.pathology,
        cfg.preprocessing.image_size,
        cfg.preprocessing.norm,
        batch_size=32,
        world_size=1,
        rank=0,
        num_workers=2
        )




if __name__ == "__main__":
    import hydra

    from dataloaders import get_dataloaders

    with hydra.initialize(version_base="1.3", config_path="../conf"):
        cfg = hydra.compose(config_name="config")


        main(cfg)