
class EarlyStopping:
    def __init__(self, patience: int = 10):
        self.patience = patience
        self.best_epoch = 0
        self.best_score = 0.0
    
    def __call__(self, metric, epoch) -> bool:
        stop = False

        if metric > self.best_score:
            self.best_score = metric
            self.best_epoch = epoch
        else:
            delta = epoch - self.best_epoch  # epochs without improvement
            stop = delta >= self.patience
            if stop:
                print(f"Early stopping occured on epoch {epoch}, after {self.patience} epochs of stagnation.")
        return stop
    

class BestModel:
    def __init__(self) -> None:
        self.accuracy = 0.0
        self.model = None
        self.epoch = 0
        self.f1 = 0.0
        self.class_report = None

    def __call__(self, cur_accuracy, cur_f1, class_report, epoch, model) -> bool:
        if cur_accuracy > self.accuracy:
            self.accuracy = cur_accuracy
            self.model = model
            self.f1 = cur_f1
            self.epoch = epoch
            self.class_report = class_report
            print("\n-------new best:-------\n")
            return True
        return False
        