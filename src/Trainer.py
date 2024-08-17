import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics import calc_accuracy, calc_conf_per_class, calc_f1
from torch_utils import BestModel, EarlyStopping
from utils import time_it

# from src.plot import plot_results

# from typing import Union

class Trainer():
    def __init__(
        self,
        task: str,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        loss, 
        class_names: list, 
        hp_dict: dict,
        patience: int,
        world_size: int,
        master_process: bool,
        testing: bool,
    ) -> None:
        self.rank = int(os.environ["LOCAL_RANK"])
        self.model = DDP(model.to(self.rank), device_ids=[self.rank])
        self.train_data = train_data
        self.validation_data = val_data
        self.loss_func = loss
        self.epochs_run = 0
        self.hp_dict = hp_dict
        self.stopper, self.stop = EarlyStopping(patience), False
        self.world_size = world_size
        self.master_process = master_process
        self.testing = testing
        self._val_true_labels = []
        self._val_predicted_labels = []
        self._best, self._is_best = BestModel(), True
        self.task = task
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        
        now = datetime.now()
        self.date_time = now.strftime("%d_%m_%Y_%H_%M")
        

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
    
    def _validation_loop(self):
        total_samples, accumulated_correct, accumulated_loss = 0, 0, 0
        tp = np.zeros(self.num_classes)
        fp = np.zeros(self.num_classes)
        fn = np.zeros(self.num_classes)

        true_labels = []
        predicted_labels = []
        self.model.eval()

        with torch.no_grad():

            for (inputs, targets) in self.validation_data:
                inputs, targets = inputs.to(self.rank, non_blocking=True), targets.to(self.rank, non_blocking=True)
                predicted = self.model(inputs)
                loss = self.loss_func(predicted, targets)
                accumulated_loss += loss.item()*targets.size(0)

                _, predicted = torch.max(predicted, 1)
                total_samples += targets.size(0)
                accumulated_correct += (predicted == targets).sum().item()

                true_labels.extend(targets.cpu().numpy())
                predicted_labels.extend(predicted.cpu().numpy())

                for class_label in range(self.num_classes):
                    true_pos, false_pos, false_neg = calc_conf_per_class(
                        class_label, 
                        predicted, 
                        targets
                    )
                    tp[class_label] += true_pos
                    fp[class_label] += false_pos
                    fn[class_label] += false_neg
                
            epoch_loss = accumulated_loss / total_samples
            epoch_correct = accumulated_correct
        
            epoch_loss = torch.tensor([epoch_loss], device=self.rank)
            epoch_correct = torch.tensor([epoch_correct], device=self.rank)
            total_samples = torch.tensor([total_samples], device=self.rank)
            dist.reduce(epoch_loss, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(epoch_correct, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(total_samples, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(torch.tensor(tp, device=self.rank), dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(torch.tensor(fp, device=self.rank), dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(torch.tensor(fn, device=self.rank), dst=0, op=dist.ReduceOp.SUM)

            if self.master_process:
                global_epoch_loss = 100 * epoch_loss.item() / self.world_size
                accuracy = calc_accuracy(epoch_correct.item(), total_samples.item())
                f1 = calc_f1(tp, fp, fn)
                avg_f1 = 100 * f1.mean().item()
            else:
                global_epoch_loss, accuracy, avg_f1 = None, None, None

        # Gather true_labels and predicted_labels from all GPUs
        gathered_true_labels = [None for _ in range(dist.get_world_size())]
        gathered_predicted_labels = [None for _ in range(dist.get_world_size())]

        dist.all_gather_object(gathered_true_labels, true_labels)
        dist.all_gather_object(gathered_predicted_labels, predicted_labels)

        if dist.get_rank() == 0:
            combined_true_labels = [label for sublist in gathered_true_labels for label in sublist]
            combined_predicted_labels = [label for sublist in gathered_predicted_labels for label in sublist]
        else:
            combined_true_labels, combined_predicted_labels = None, None

        return avg_f1, global_epoch_loss, accuracy, combined_true_labels, combined_predicted_labels
    
    
    def _train_loop(self):
        accumulated_loss = 0
        total_samples = 0
        accumulated_correct = 0
        global_epoch_loss = 0

        tp = np.zeros(self.num_classes)
        fp = np.zeros(self.num_classes)
        fn = np.zeros(self.num_classes)

        self.model.train()

        for inputs, targets in self.train_data:
            inputs, targets = inputs.to(self.rank, non_blocking=True), targets.to(self.rank, non_blocking=True)
            self.optimizer.zero_grad()
            predicted = self.model(inputs)
            loss = self.loss_func(predicted, targets)
            loss.backward()
            self.optimizer.step()

            _, predicted = torch.max(predicted, 1)

            accumulated_loss += loss.item()*targets.size(0)
            total_samples += targets.size(0)
            accumulated_correct += (predicted == targets).sum().item()

            for class_label in range(self.num_classes):
                true_pos, false_pos, false_neg = calc_conf_per_class(
                class_label, 
                predicted, 
                targets
            )
                tp[class_label] += true_pos
                fp[class_label] += false_pos
                fn[class_label] += false_neg

        epoch_loss = accumulated_loss / total_samples # epoch loss per GPU
        epoch_correct = accumulated_correct

        # Reduce results across all processes
        epoch_loss = torch.tensor([epoch_loss], device=self.rank)
        epoch_correct = torch.tensor([epoch_correct], device=self.rank)
        total_samples = torch.tensor([total_samples], device=self.rank)
        dist.reduce(epoch_loss, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(epoch_correct, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(total_samples, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(torch.tensor(tp, device=self.rank), dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(torch.tensor(fp, device=self.rank), dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(torch.tensor(fn, device=self.rank), dst=0, op=dist.ReduceOp.SUM)

        if self.master_process:
            global_epoch_loss = 100 * epoch_loss.item() / self.world_size
            accuracy = calc_accuracy(epoch_correct.item(), total_samples.item())
            f1 = calc_f1(tp, fp, fn)
            avg_f1 = 100 * f1.mean().item()
        else:
            global_epoch_loss, accuracy, avg_f1 = None, None, None


        return avg_f1, global_epoch_loss, accuracy

    @time_it
    def _run_epoch(self, epoch):

        self.train_data.sampler.set_epoch(epoch)
        self.validation_data.sampler.set_epoch(epoch)

        train_avg_f1, train_loss, train_accuracy = self._train_loop()
        # if self.master_process:
            # self._train_f1_hist.append(train_avg_f1)
            # self._train_loss_hist.append(train_loss)
            # self._train_accuracy_history.append(train_accuracy)

        (
            val_avg_f1,
            val_loss,
            val_accuracy,
            val_true_labels,
            val_prediced_labels,
        ) = self._validation_loop()
        
        if self.master_process:
            if not(self.testing):
                import wandb
                
                wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss": train_loss, 
                        "val/loss": val_loss,
                        "train/accuracy": train_accuracy,
                        "val/accuracy": val_accuracy,
                        "train/f1": train_avg_f1,
                        "val/f1": val_avg_f1,
                        "lr": self.scheduler.get_last_lr()[0]
                    }
                )
            
            self._val_true_labels.extend(val_true_labels)
            self._val_predicted_labels.extend(val_prediced_labels)

            cur_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step(val_loss)
            new_lr = self.scheduler.get_last_lr()[0]
            if new_lr != cur_lr:
                print(f"\nlr updated -> {new_lr}.\n")

            f1 = {"train": train_avg_f1,
                "val": val_avg_f1}

            accuracy = {"train": train_accuracy,
                        "val": val_accuracy}
            
            loss = {"train": train_loss,
                    "val": val_loss}

            return f1, accuracy, loss
        return None, None, None
    
    def _set_optimizer_scheduler(self) -> None:
        import torch.optim as optim

        beta1 = 0.9
        beta2 = 0.999
        
        self.optimizer = optim.AdamW(params=self.model.parameters(), 
                                     lr=self.hp_dict["lr"], 
                                     weight_decay=self.hp_dict["weight_decay"], 
                                     betas=(beta1, beta2))
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.8, patience=15, min_lr=1e-5)
        
    def train_model(self):

        self._set_optimizer_scheduler()
        max_epochs = 2 if self.testing  else self.hp_dict["epochs"]
        # max_epochs = self.hp_dict["epochs"]

        for epoch in tqdm(range(1, max_epochs+1), total=max_epochs,  disable=(not(self.master_process)), colour='green'):
            (f1, accuracy, loss), exec_time = self._run_epoch(epoch)
            if self.master_process and not self.testing:
                self.stop = self.stopper(accuracy["val"], epoch)
                self._is_best = self._best(accuracy["val"], f1["val"], epoch, self.model)
                if self._is_best:
                    self._save_snapshot(epoch, self.hp_dict, self.date_time, threshold=90.0, metric=accuracy["val"])
            
                epoch_summary = ", ".join(
                    [
                        f"[Epoch {epoch}/{self.hp_dict['epochs']}]:",
                        f"Train/val loss {loss['train']:.2f}|{loss['val']:.2f}",
                        f"Val F1 score: {f1['val']:.2f}%",
                        f"Val accuracy {accuracy['val']:.2f}|{self._best.accuracy:.2f}%",
                    ]
                )
                print("\n" + epoch_summary + "\n")

            broadcast_list = [self.stop if self.master_process else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            self.stop = broadcast_list[0]
            if self.stop:
                break # must break all DDP ranks
                
        if self.master_process:
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

    def _save_snapshot(self, epoch:int, hp_dict:dict, date_time:str, threshold:float, metric:float) -> None:
        if metric > threshold:
            param_string = "_".join([f"{k}={v}" for k, v in hp_dict.items()])
            Path("best_models").mkdir(exist_ok=True)
            Path(f"best_models/{self.task}").mkdir(exist_ok=True)
            filename=Path(f"best_models/{self.task}/{param_string}-{date_time}/best_model.pt")
            filename.parent.mkdir(exist_ok=True)

            snapshot = {
                "MODEL_STATE": self.model.module.state_dict(),
                "EPOCHS_RUN": epoch,
            }
            torch.save(snapshot, filename)
            print(f"epoch {epoch} => Saving a new best model.")