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