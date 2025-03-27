import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    classification_report,
    roc_auc_score
)
from typing import Any, Dict

def evaluate_model(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    device: torch.device = None,
    multiclass: bool = False
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation function for PyTorch models.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained PyTorch model to evaluate
    dataloader : torch.utils.data.DataLoader
        DataLoader containing test data
    device : torch.device, optional
        Device to run evaluation on (CPU/GPU)
    multiclass : bool, default False
        Flag to indicate multiclass classification
    
    Returns:
    --------
    Dict with various evaluation metrics
    """
    # Set default device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move model to device and set to evaluation mode
    model.to(device)
    model.eval()
    
    # Lists to store predictions and true labels
    all_preds = []
    all_true_labels = []
    all_pred_proba = []
    
    # Disable gradient computation for evaluation
    with torch.no_grad():
        for batch in dataloader:
            # Assuming dataloader returns (inputs, labels)
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Get predictions
            if isinstance(outputs, tuple):
                # Handle models that return multiple outputs
                outputs = outputs[0]
            
            # Compute probabilities and predictions
            if multiclass:
                # Softmax for multiclass
                probas = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probas, dim=1)
            else:
                # Sigmoid for binary
                probas = torch.sigmoid(outputs)
                preds = (probas > 0.5).long()
            
            # Move to CPU for sklearn metrics
            all_preds.extend(preds.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
            all_pred_proba.extend(probas.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_true_labels = np.array(all_true_labels)
    all_pred_proba = np.array(all_pred_proba)
    
    # Compute evaluation metrics
    results = {
        'accuracy': accuracy_score(all_true_labels, all_preds),
        'confusion_matrix': confusion_matrix(all_true_labels, all_preds).tolist(),
        'classification_report': classification_report(all_true_labels, all_preds, output_dict=True)
    }
    
    # Precision, Recall, F1 with multi-class handling
    average_mode = 'weighted' if multiclass else 'binary'
    results.update({
        'precision': precision_score(all_true_labels, all_preds, average=average_mode),
        'recall': recall_score(all_true_labels, all_preds, average=average_mode),
        'f1_score': f1_score(all_true_labels, all_preds, average=average_mode)
    })
    
    # ROC AUC calculation
    try:
        results['roc_auc'] = roc_auc_score(
            all_true_labels, 
            all_pred_proba[:, 1] if multiclass else all_pred_proba,
            multi_class='ovr' if multiclass else 'raise'
        )
    except ValueError:
        results['roc_auc'] = None
    
    return results

def print_evaluation_results(results: Dict[str, Any]):
    """
    Pretty print the evaluation results.
    
    Parameters:
    -----------
    results : Dict
        Results from evaluate_model function
    """
    print("Model Evaluation Metrics:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    
    if results['roc_auc'] is not None:
        print(f"ROC AUC: {results['roc_auc']:.4f}")
    
    print("\nConfusion Matrix:")
    for row in results['confusion_matrix']:
        print(row)
    
    print("\nDetailed Classification Report:")
    for category, metrics in results['classification_report'].items():
        if category not in ['accuracy', 'macro avg', 'weighted avg']:
            print(f"{category}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")

# Example usage
def example_usage():
    """
    Example of how to use the evaluation function
    """
    # Assuming you have:
    # - model: Your trained PyTorch model
    # - test_dataloader: DataLoader for test data
    
    # Detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Evaluate the model
    results = evaluate_model(
        model=model, 
        dataloader=test_dataloader, 
        device=device,
        multiclass=True  # Set based on your problem
    )
    
    # Print results
    print_evaluation_results(results)