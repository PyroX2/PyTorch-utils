import torch
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score, MulticlassAUPRC, MulticlassAUROC
from typing import List, Tuple


class EarlyStopping:
    def __init__(self, patience: int = 10, init_metric_value: float = float("inf"), task: str = "min") -> None:
        self.patience = patience
        self.best_value_recorded = init_metric_value
        self.no_improvement_count = 0

        assert task in ["min", "max"], print(f"Task value of {task} is incorrect. Choose from ['min', 'max']")
        self.task = task

    def check_early_stop(self, metric) -> bool:
        if (self.task == "min" and metric < self.best_value_recorded) or (self.task == "max" and metric > self.best_value_recorded):
            self.best_value_recorded = metric
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        if self.no_improvement_count == self.patience:
            return True
        else:
            return False


class MetricsCalculator:
    def __init__(self, num_classes):
        self.accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.f1_score = MulticlassF1Score(num_classes=num_classes)
        self.auprc = MulticlassAUPRC(num_classes=num_classes)
        self.auroc = MulticlassAUROC(num_classes=num_classes)
    
    def calculate(self, outputs: List, targets: List) -> Tuple:
        outputs = torch.tensor(outputs, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)

        self.accuracy.update(outputs, targets)
        accuracy = self.accuracy.compute()

        self.f1_score.update(outputs, targets)
        f1_score = self.f1_score.compute()

        self.auprc.update(outputs, targets)
        auprc = self.auprc.compute()

        self.auroc.update(outputs, targets)
        auroc = self.auroc.compute()

        self.accuracy.reset()
        self.f1_score.reset()
        self.auprc.reset()
        self.auroc.reset()

        return accuracy, f1_score, auprc, auroc