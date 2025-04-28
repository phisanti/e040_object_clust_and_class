import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from typing import Dict, List, Optional, Union, Any, Tuple

def plot_confusion_matrix(
    labels: Union[List[int], np.ndarray], 
    predictions: Union[List[int], np.ndarray], 
    class_labels: Optional[List[str]] = None, 
    use_percentage: bool = False, 
    figsize: Tuple[int, int] = (10, 8), 
    verbose: bool = True, 
    return_metrics: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Plot a confusion matrix and calculate classification metrics.
    
    Args:
        labels: Ground truth labels.
        predictions: Predicted labels.
        class_labels: List of class names corresponding to label indices.
        use_percentage: If True, display percentages instead of counts.
        figsize: Figure size as (width, height) in inches.
        verbose: If True, print detailed metrics report.
        return_metrics: If True, return a dictionary with calculated metrics.
        
    Returns:
        If return_metrics is True, returns a dictionary containing:
            - accuracy: Overall accuracy score
            - precision: Weighted precision score
            - recall: Weighted recall score (sensitivity)
            - f1: Weighted F1-score
            - specificity: Array of specificity values per class
            - confusion_matrix: The confusion matrix
        
    Raises:
        ValueError: If labels and predictions have different lengths or contain different classes.
    """
    # Input validation
    labels = np.array(labels)
    predictions = np.array(predictions)
    
    if len(labels) != len(predictions):
        raise ValueError(f"Labels and predictions must have the same length. Got {len(labels)} and {len(predictions)}.")
    
    if len(labels) == 0:
        raise ValueError("Labels and predictions cannot be empty.")
    
    unique_labels = np.unique(labels)
    unique_preds = np.unique(predictions)
    
    if not np.all(np.isin(unique_labels, unique_preds)):
        missing = set(unique_labels) - set(unique_preds)
        raise ValueError(f"Some classes in labels are not present in predictions: {missing}")
    
    # Calculate confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    # Calculate specificity for each class
    n_classes = len(unique_labels)
    specificity = np.zeros(n_classes)
    for i in range(n_classes):
        # True negatives: all not i correctly not classified as i
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        # False positives: all not i incorrectly classified as i
        fp = np.sum(np.delete(cm[:, i], i))
        specificity[i] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Format matrix for display
    if use_percentage:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        fmt = '.1f'
        data = cm_normalized
    else:
        fmt = 'd'
        data = cm
    
    # Print metrics report
    if verbose:
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Weighted Precision: {precision:.4f}")
        print(f"Weighted Recall (Sensitivity): {recall:.4f}")
        print(f"Weighted F1-Score: {f1:.4f}")
        print(f"Mean Specificity: {np.mean(specificity):.4f}")
        
        # Print detailed classification report
        print("\nClassification Report:")
        print(classification_report(labels, predictions, target_names=class_labels if class_labels else None))
    
        # Display per-class specificity
        if class_labels:
            print("\nSpecificity by class:")
            for i, label in enumerate(class_labels):
                print(f"{label}: {specificity[i]:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=figsize)
    sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
               xticklabels=class_labels,
               yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
    if return_metrics:
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity,
            'confusion_matrix': cm
        }
