import os
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.metrics import calc_accuracy, calc_conf_per_class, calc_f1
from src.plot import plot_results


class Trainer():
    def __init__(
        self,
        task: str,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss, 
        scheduler,
        # snapshot_path: Union[Path, str],
        class_names: list, 
        max_epochs: int,
        lr: float

    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.validation_data = val_data
        self.optimizer = optimizer
        self.loss_func = loss
        self.epochs_run = 0
        self.max_epochs = max_epochs
        self.lr = lr
        self.scheduler = scheduler
        self._train_loss_hist = []
        self._val_loss_hist = []
        self._train_accuracy_history = []
        self._val_accuracy_history = []
        self._train_f1_hist = []
        self._val_f1_hist = []
        self._val_true_labels = []
        self._val_predicted_labels = []
        self._best = {"accuracy": 0,
                      "model": None,
                      "epoch": 0,
                      "f1": 0}
        self.task = task
        self.class_names = class_names
        self.num_classes = len(class_names)

        # self.snapshot_path = Path(snapshot_path)
        # if self.snapshot_path.exists():
        #     print("Loading snapshot")
        #     self._load_snaphot(self.snapshot_path)

        self.model = DDP(self.model, device_ids=[self.gpu_id])
    
    def _load_snaphot(self, snapshot_path: Path) -> None:
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")
    

    def _validation_loop(self):
        val_total, val_correct, val_running_loss = 0, 0, 0
        tp = torch.zeros(self.num_classes, device=self.gpu_id)
        fp = torch.zeros(self.num_classes, device=self.gpu_id)
        fn = torch.zeros(self.num_classes, device=self.gpu_id)

        true_labels = []
        predicted_labels = []
        self.model.eval()

        with torch.no_grad():

            for (inputs, targets) in self.validation_data:
                inputs, targets = inputs.to(self.gpu_id, non_blocking=True), targets.to(self.gpu_id, non_blocking=True)
                predicted = self.model(inputs)
                val_running_loss += self.loss_func(predicted, targets).item()

                _, predicted = torch.max(predicted, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

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
                
        f1 = calc_f1(tp, fp, fn)
        avg_f1 = f1.mean().item()

        avg_val_loss = val_running_loss / len(self.validation_data)  # divide by no. of batches
        val_accuracy = calc_accuracy(val_correct, val_total)

        total_loss = torch.tensor([val_running_loss], device=self.gpu_id)
        total_correct  = torch.tensor([val_correct], device=self.gpu_id)
        total = torch.tensor([val_total], device=self.gpu_id)
        dist.reduce(total_loss, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(total_correct, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(total, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(tp, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(fp, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(fn, dst=0, op=dist.ReduceOp.SUM)

        if dist.get_rank() == 0:
            avg_val_loss = total_loss.item() / len(self.validation_data)
            val_accuracy = 100* total_correct.item() / total.item()
            f1 = calc_f1(tp, fp, fn)
            avg_f1 = f1.mean().item()
        else:
            avg_val_loss, val_accuracy, avg_f1 = None, None, None


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

        return avg_f1, avg_val_loss, val_accuracy, combined_true_labels, combined_predicted_labels
    
    def _train_loop(self):
        train_running_loss = 0
        train_total_samples = 0
        train_correct = 0
        loss = 0

        tp = np.zeros(self.num_classes)
        fp = np.zeros(self.num_classes)
        fn = np.zeros(self.num_classes)

        self.model.train()

        for inputs, targets in tqdm(self.train_data, total=len(self.train_data)):
            inputs, targets = inputs.to(self.gpu_id, non_blocking=True), targets.to(self.gpu_id, non_blocking=True)
            self.optimizer.zero_grad()
            predicted = self.model(inputs)
            loss = self.loss_func(predicted, targets)
            loss.backward()
            self.optimizer.step()

            _, predicted = torch.max(predicted, 1)

            train_running_loss += loss.item()
            train_total_samples += targets.size(0)
            train_correct += (predicted == targets).sum().item()

            for class_label in range(self.num_classes):
                true_pos, false_pos, false_neg = calc_conf_per_class(
                class_label, 
                predicted, 
                targets
            )
                tp[class_label] += true_pos
                fp[class_label] += false_pos
                fn[class_label] += false_neg

            # Reduce results across all processes
        total_loss = torch.tensor([train_running_loss], device=self.gpu_id)
        total_correct = torch.tensor([train_correct], device=self.gpu_id)
        total_samples = torch.tensor([train_total_samples], device=self.gpu_id)
        dist.reduce(total_loss, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(total_correct, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(total_samples, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(torch.tensor(tp, device=self.gpu_id), dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(torch.tensor(fp, device=self.gpu_id), dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(torch.tensor(fn, device=self.gpu_id), dst=0, op=dist.ReduceOp.SUM)

        if dist.get_rank() == 0:
            loss = train_running_loss / len(self.train_data)
            accuracy = calc_accuracy(train_correct, train_total_samples)
            f1 = calc_f1(tp, fp, fn)
            avg_f1 = f1.mean().item()
        else:
            loss, accuracy, avg_f1 = None, None, None


        return avg_f1, loss, accuracy

    def _run_epoch(self, epoch):

        b_sz = len(next(iter(self.train_data))[0])
        # print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch) # ----------
        self.validation_data.sampler.set_epoch(epoch)

        train_avg_f1, train_loss, train_accuracy = self._train_loop()
        if self.gpu_id == 0:
            self._train_f1_hist.append(train_avg_f1)
            self._train_loss_hist.append(train_loss)
            self._train_accuracy_history.append(train_accuracy)

        (
            val_avg_f1,
            val_loss,
            val_accuracy,
            val_true_labels,
            val_prediced_labels,
        ) = self._validation_loop()
        if self.gpu_id == 0:
            self._val_true_labels.extend(val_true_labels)
            self._val_predicted_labels.extend(val_prediced_labels)
            self._val_loss_hist.append(val_loss)
            self._val_accuracy_history.append(val_accuracy)
            self._val_f1_hist.append(val_avg_f1)


            self.scheduler.step(val_loss)
            new_lr = self.scheduler.get_last_lr()[0]
            if new_lr < self.lr:
                self.lr = new_lr
                print(f"lr updated -> {new_lr}.")

            epoch_summary = ", ".join(
                [
                    f"[Epoch {epoch}/{self.max_epochs}]:",
                    f"Train loss {self._train_loss_hist[-1]:.2f}",
                    f"Val accuracy {val_accuracy:.2f}%",
                    f"Val average F1 score: {val_avg_f1:.2f}",
                ]
            )

            f1 = {"train": train_avg_f1,
                "val": val_avg_f1}

            accuracy = {"train": train_accuracy,
                        "val": val_accuracy}
            
            loss = {"train": train_loss,
                    "val": val_loss}
            print("\n" + epoch_summary + "\n")

            return f1, accuracy, loss
        return None, None, None
    
    def train_model(self):
        for epoch in tqdm(range(1, self.max_epochs+1), total=self.max_epochs):
            f1, accuracy, loss = self._run_epoch(epoch)
            if self.gpu_id == 0:
                self._update_hist(f1, accuracy, loss)
                self._update_best(accuracy["val"], f1["val"], epoch)
                # self._scheduler_step(loss["val"])

        self._cleanup()

        if self.gpu_id == 0:
            summary = ", ".join(
                [
                    "-" * 30, 
                    f"best val accuray {self._best['accuracy']*100:.2f} ", 
                    f"achieved at epoch {self._best['epoch']}, ", 
                    f"with f1 score {self._best['f1']*100:.2f}."
                ]
            )
            print("\n" + summary + "\n")
        

            results = {
                "loss": (self._train_loss_hist, self._val_loss_hist),
                "accuracy": (self._train_accuracy_history, self._val_accuracy_history, (self._best["epoch"], self._best["accuracy"])),
                "f1": (self._train_f1_hist, self._val_f1_hist),
                "val_true_labels": (self._val_true_labels),
                "val_predicted_labels": (self._val_predicted_labels)
            }
            self._plot_results(results)
    def _cleanup(self):
        torch.cuda.empty_cache()

    def _scheduler_step(self, val_loss):
        self.scheduler.step(val_loss)
        new_lr = self.scheduler.get_last_lr()[0]
        if new_lr < self.lr:
            self.lr = new_lr
            print(f"lr updated -> {new_lr}.")


    def _update_hist(self, f1:dict, accuracy:dict, loss:dict) -> None:
        self._train_accuracy_history.append(accuracy["train"])
        self._val_accuracy_history.append(accuracy["val"])
        self._train_loss_hist.append(loss["train"])
        self._val_loss_hist.append(loss["val"])
        self._train_f1_hist.append(f1["train"])
        self._val_f1_hist.append(f1["val"])

    def _update_best(self, val_accuracy, val_f1, epoch:int):
        if val_accuracy > self._best["accuracy"]:
            self._best["accuracy"] = val_accuracy
            self._best["model"] = self.model
            self._best["epoch"] = epoch
            self._best["f1"] = val_f1
            if self.gpu_id == 0:
                self._save_snapshot(epoch)
            print("\n-------new best:-------")

    def _save_snapshot(self, epoch):
        Path("best_models").mkdir(exist_ok=True)
        filename=Path(f"best_models/{self.task}/best_model.pt")
        filename.parent.mkdir(exist_ok=True)

        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, filename)
        print(f"epoch {epoch} => Saving a new best model at {filename}")

    def _plot_results(self, results):
        plot_results(Path("best_models"), results, self.task, self.class_names)
