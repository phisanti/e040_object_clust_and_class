import numpy as np
import torch
import torch.nn.functional as F
from monai.metrics import ConfusionMatrixMetric, compute_confusion_matrix_metric

# Top-K Accuracy metric
class TopKAccuracy:
    def __init__(self, k: int = 3) -> None:
        self.k = k
        
    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Get top-k predictions
        _, top_k_indices = torch.topk(y_pred, k=self.k, dim=1)
        
        # Check if true class is in top-k predictions
        y_true_expanded = y_true.unsqueeze(1).expand_as(top_k_indices)
        correct = torch.eq(top_k_indices, y_true_expanded).any(dim=1)
        
        # Return average accuracy
        return torch.mean(correct.float())


class ConfusionMatrixMetricWrapper:
    def __init__(self, metric_name):
        self.metric_name = metric_name
        self.confusion_matrix_metric = ConfusionMatrixMetric(
            include_background=True,
            metric_name=None,  # We'll compute metrics manually
            reduction="sum"
        )
      
    def __call__(self, y_pred, y_true):
        # Convert predictions from probabilities to class indices
        y_pred_indices = torch.argmax(y_pred, dim=1)

        # Convert to one-hot encoding to match MONAI's expected format
        num_classes = y_pred.shape[1]
        y_pred_one_hot = torch.zeros_like(y_pred)
        y_pred_one_hot.scatter_(1, y_pred_indices.unsqueeze(1), 1)

        # Convert ground truth to one-hot as well
        y_true_one_hot = torch.zeros(y_true.shape[0], num_classes, device=y_true.device)
        y_true_one_hot.scatter_(1, y_true.unsqueeze(1), 1)

        # Get summed confusion matrix (across the batch)
        cm = self.confusion_matrix_metric(y_pred_one_hot, y_true_one_hot)

        # Compute the specific metric from confusion matrix
        result = compute_confusion_matrix_metric(self.metric_name, cm)
        scalar_result = None
        if isinstance(result, tuple):
            numeric_values = []
            for item in result:
                if isinstance(item, torch.Tensor):
                    item_mean = torch.nanmean(item.float()).item()
                    if np.isfinite(item_mean):
                            numeric_values.append(item_mean)
                elif isinstance(item, (int, float)) and np.isfinite(item):
                    numeric_values.append(float(item))
            if numeric_values:
                scalar_result = sum(numeric_values) / len(numeric_values)
            else:
                scalar_result = 0.0
        elif isinstance(result, torch.Tensor):
            scalar_result = torch.nanmean(result.float()).item()
        elif isinstance(result, (int, float)):
                scalar_result = float(result)
        else:
            print(f"Warning: Unexpected metric result type: {type(result)} for metric '{self.metric_name}'. Returning 0.0.")
            scalar_result = 0.0

        if scalar_result is None or not np.isfinite(scalar_result):
                scalar_result = 0.0

        return torch.tensor(scalar_result, device=cm.device, dtype=torch.float32)
