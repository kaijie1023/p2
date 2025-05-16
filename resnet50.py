import os
import numpy as np
import pandas as pd
import seaborn as sns
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler,  Subset
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassAUROC, MulticlassConfusionMatrix
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from PIL import Image
import random
import time
from typing import List, Dict, Tuple, Any, Optional
import copy
import cv2

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

gradcam_images = [
    '/Monkeypox/monkeypox1.png',
    '/Monkeypox/monkeypox32.png',
    '/Monkeypox/monkeypox59.png',
    '/Monkeypox/monkeypox100.png'
    '/Monkeypox/monkeypox183.png',
    '/Monkeypox/monkeypox252.png',
    '/Monkeypox/monkeypox271.png'
]

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Set seed for reproducibility
def set_seed(seed: int = 42) -> None:
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Function to load dataset from directory
def load_dataset_from_directory(data_dir):
    """
    Load images from a directory structure:
    data_dir/
        class1/
            img1.jpg
            img2.jpg
            ...
        class2/
            img1.jpg
            ...
        ...
    
    Returns:
        image_paths: List of paths to images
        labels: Corresponding labels (numeric)
        class_names: List of class names
    """
    image_paths = []
    labels = []
    class_names = sorted(os.listdir(data_dir))
    class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}
    
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    image_paths.append(img_path)
                    labels.append(class_to_idx[class_name])
    
    return image_paths, labels, class_names

# Custom Dataset class for loading images from a directory
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        # image = np.array(Image.open(img_path).convert('RGB'))
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            # augmented = self.transform(image=image)
            # image = augmented['image']  # get the transformed tensor
            
        return image, label

# Ensemble Model
class EnsembleModel(nn.Module):
    """Ensemble of ResNet, resnet, and DeiT models with weighted voting."""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        """
        Args:
            num_classes (int): Number of classes for classification
            pretrained (bool): Whether to use pretrained models
        """
        super(EnsembleModel, self).__init__()
        
        # Initialize models
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with weighted ensemble."""
        res_out = self.resnet(x)
        
        return res_out
    
    def get_model_outputs(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get individual model outputs for analysis."""
        res_out = self.resnet(x)
        dense_out = self.resnet(x)
        deit_out = self.deit(x)
        return res_out, dense_out, deit_out
    
    def name(self) -> str:
        return "resnet"

# Training utilities
class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, restore_best_weights: bool = True):
        """
        Args:
            patience (int): How many epochs to wait after validation loss improvement
            min_delta (float): Minimum change to qualify as improvement
            restore_best_weights (bool): Whether to restore model weights from best epoch
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.best_weights = copy.deepcopy(model.state_dict())
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_weights = copy.deepcopy(model.state_dict())
            self.counter = 0
        
        return self.early_stop

class LRScheduler:
    """Learning rate scheduler."""
    
    def __init__(self, optimizer, patience: int = 5, min_lr: float = 1e-6, factor: float = 0.5):
        """
        Args:
            optimizer: The optimizer to adjust LR for
            patience (int): Number of epochs with no improvement after which LR will be reduced
            min_lr (float): Lower bound on the learning rate
            factor (float): Factor by which the learning rate will be reduced
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.best_loss = float('inf')
        self.counter = 0
    
    def __call__(self, val_loss: float) -> None:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self._reduce_lr()
                self.counter = 0
    
    def _reduce_lr(self) -> None:
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            print(f"Reducing learning rate from {old_lr:.6f} to {new_lr:.6f}")

# Stratified Cross-Validation
def stratified_cross_validation(
    data_dir: str,
    model_class: Any,
    n_splits: int = 5,
    batch_size: int = 32,
    learning_rate: float = 5e-5,
    weight_decay: float = 1e-5,
    num_epochs: int = 50,
    patience: int = 7,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[str, Any]:
    """
    Perform stratified k-fold cross-validation.
    
    Args:
        dataset: Dataset object
        model_class: Model class to instantiate
        n_splits: Number of folds for cross-validation
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        weight_decay: Weight decay for optimizer
        num_epochs: Maximum number of training epochs
        patience: Patience for early stopping
        device: Device to use for training
        
    Returns:
        Dictionary with performance metrics and trained models
    """
    # Load dataset
    image_paths, labels, class_names = load_dataset_from_directory(data_dir)
    num_classes = len(class_names)

    # First, split into test and train+val with stratification
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # Create test dataset and loader
    test_dataset = ImageDataset(X_test, y_test, transform=val_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize k-fold cross validation
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Results storage
    results = {
        'fold_metrics': [],
        'models': [],
        'train_losses': [],
        'val_losses': [],
        'cams': {}
    }


    # GradCAM images key map to image id
    for i in gradcam_images:
        results['cams'][get_image_id(i)] = []
    
    # Loop through folds
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_val, y_train_val)):
        print(f"\n{'='*50}\nFold {fold+1}/{n_splits}\n{'='*50}")

         # Get fold training and validation data
        X_train = [X_train_val[i] for i in train_idx]
        y_train = [y_train_val[i] for i in train_idx]
        X_val = [X_train_val[i] for i in val_idx]
        y_val = [y_train_val[i] for i in val_idx]
        
        # Create datasets and dataloaders for this fold
        train_dataset = ImageDataset(X_train, y_train, transform=train_transforms)
        val_dataset = ImageDataset(X_val, y_val, transform=val_transforms)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        model = model_class(num_classes=num_classes).to(device)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Learning rate scheduler and early stopping
        scheduler = LRScheduler(optimizer, patience=5)
        early_stopping = EarlyStopping(patience=patience)
        
        # Training history
        train_losses = []
        val_losses = []
        
        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
            
            train_loss = running_loss / len(train_loader.sampler)
            train_losses.append(train_loss)
            
            # Validation phase
            model.eval()
            running_loss = 0.0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    running_loss += loss.item() * inputs.size(0)
            
            val_loss = running_loss / len(val_loader.sampler)
            val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Learning rate scheduler
            scheduler(val_loss)
            
            # Early stopping
            if early_stopping(val_loss, model):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Load best model weights
        if early_stopping.restore_best_weights:
            model.load_state_dict(early_stopping.best_weights)
        
        # Evaluate on test set
        model.eval()
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            preds = []
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                # _, preds = torch.max(outputs, 1)
                
                y_true.extend(labels)
                preds.append(torch.softmax(outputs, dim=1))

                # probs = torch.softmax(outputs, dim=1)
                # roc_auc_metric.update(probs.cpu(), labels.cpu())
            y_pred.append(torch.cat(preds))
        
        target_layer = model.resnet.layer4[-1].conv3
        cam = GradCAMPlusPlus(model=model.resnet, target_layers=[target_layer])
        # gradcam = GradCAM_CNN(model.resnet, target_layer)


        for image in gradcam_images:
            image_tensor = preprocess_image(f"{data_dir}{image}").to(device)

            # Run forward pass through the model
            output = model(image_tensor)

            # Get predicted class index (highest probability)
            pred_class = output.argmax(dim=1).item() 

            # Use the predicted class for Grad-CAM++
            targets = [ClassifierOutputTarget(pred_class)]

            # Generate heatmap
            grayscale_cam = cam(input_tensor=image_tensor, targets=targets)

            results['cams'][get_image_id(image)].append(grayscale_cam[0])
        
        

        # results['cams'].append(grayscale_cam[0])
        
        
        # Store results
        # results['fold_metrics'].append(fold_metrics)
        results['models'].append(model)
        results['train_losses'].append(train_losses)
        results['val_losses'].append(val_losses)
    
    # Calculate overall metrics
    # avg_metrics = {metric: np.mean([fold[metric] for fold in results['fold_metrics']]) 
                #  for metric in results['fold_metrics'][0].keys()}
    
    # results['avg_metrics'] = avg_metrics
    
    print("\nAverage Metrics Across All Folds:")
    # for metric, value in avg_metrics.items():
    #     print(f"{metric}: {value:.4f}")

    # Average predictions across models
    avg_preds = torch.stack(y_pred).mean(dim=0)
    final_preds = torch.argmax(avg_preds, dim=1)
    final_labels = torch.tensor(y_true).to(device)

    accuracy = MulticlassAccuracy(num_classes=num_classes).to(device)
    accuracy.update(final_preds, final_labels)
    accuracy = accuracy.compute().item()

    precision = MulticlassPrecision(num_classes=num_classes).to(device)
    precision.update(final_preds, final_labels)
    precision = precision.compute().item()

    recall = MulticlassRecall(num_classes=num_classes).to(device)
    recall.update(final_preds, final_labels)
    recall = recall.compute().item()

    f1 = MulticlassF1Score(num_classes=num_classes).to(device)
    f1.update(final_preds, final_labels)
    f1 = f1.compute().item()

    roc_auc = MulticlassAUROC(num_classes=num_classes).to(device)
    roc_auc.update(avg_preds, final_labels)
    roc_auc = roc_auc.compute().item()

    cm = MulticlassConfusionMatrix(num_classes=num_classes).to(device)
    cm.update(final_preds, final_labels)
    cm = cm.compute()

    results['metrics'] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }

    print(f"Test Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
    
    return results

# Plot training and validation loss
def plot_losses(train_losses, val_losses, fold=None):
    """Plot training and validation losses."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    title = 'Training and Validation Loss'
    if fold is not None:
        title += f' - Fold {fold+1}'
    plt.title(title)
    plt.legend()
    plt.grid(True)
    return plt

# Evaluate the model
def evaluate_model(model, test_loader, class_names, output_dir, save=False):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    roc_auc_metric = MulticlassAUROC(num_classes=len(class_names)).to(device)
    roc_auc_metric.reset()
    cm_metric = MulticlassConfusionMatrix(num_classes=len(class_names)).to(device)
    cm_metric.reset()
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            probs = torch.softmax(outputs, dim=1)
            roc_auc_metric.update(probs.cpu(), labels.cpu())
            cm_metric.update(torch.tensor(preds), torch.tensor(labels))
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    roc_auc = roc_auc_metric.compute().item()
    cm = cm_metric.compute()
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_metric, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    # plt.show()
    if (save):
        plt.savefig(f"{output_dir}/confusion_matrix.png")
        plt.close()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc': roc_auc,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'true_labels': all_labels,
        'probabilities': all_probs
    }

def save_cam_on_image(img_tensor, cam, save_path, title=None):
    import os

    img = img_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    overlay = 0.5 * heatmap + 0.5 * img
    overlay = np.uint8(255 * overlay)

    # Save with OpenCV
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    if title:
        print(f"Saved: {title} -> {save_path}")

def get_image_id(path):
    return path.split('/')[-1].split('.')[0]

def normalize_gradcam_iamge(image_path: str):
    image = Image.open(image_path).convert("RGB")
    return np.array(image.resize((224, 224))) / 255.0

def preprocess_image(image_path: str, image_size: int = 224) -> torch.Tensor:
    """
    Loads and preprocesses an image for CNN/ViT models.

    Args:
        image_path (str): Path to the input image.
        image_size (int): Target image size (default is 224).

    Returns:
        torch.Tensor: Preprocessed image tensor of shape [1, 3, H, W].
    """
    image = Image.open(image_path).convert("RGB")
    tensor = val_transforms(image).unsqueeze(0)  # Add batch dimension
    return tensor

# Main training workflow
def main(data_dir, output_dir=None, n_splits=5, batch_size=32, num_epochs=50, learning_rate=5e-5):
    """Main function to run the entire workflow."""
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Run stratified cross-validation
    results = stratified_cross_validation(
        data_dir=data_dir,
        model_class=EnsembleModel,
        n_splits=n_splits,
        batch_size=batch_size,
        num_epochs=num_epochs,
        device=device,
        learning_rate=learning_rate
    )
    
    # Plot losses for each fold
    if output_dir:

        for fold in range(n_splits):
            plt_fig = plot_losses(results['train_losses'][fold], results['val_losses'][fold], fold)
            plt_fig.savefig(os.path.join(output_dir, f'fold_{fold+1}_losses.png'))
            plt.close()

        # Save average metrics
        metrics_df = pd.DataFrame([results['metrics']], columns=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
        metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
        
        # Save individual fold metrics
        # fold_metrics_df = pd.DataFrame(results['fold_metrics'])
        # fold_metrics_df.to_csv(os.path.join(output_dir, 'fold_metrics.csv'), index=False)
        
        for img in results['cams']:
            avg_cam = np.mean(results['cams'][img], axis=0)

            # save_cam_on_image(preprocess_image(f"{data_dir}/Monkeypox/monkeypox1.png"), cam, f"{output_dir}/cam.png")

            # Visualize the average
            visualization = show_cam_on_image(normalize_gradcam_iamge(f"{data_dir}/Monkeypox/{img}.png"), avg_cam, use_rgb=True)

            # Save the visualization
            save_path = f"{output_dir}/gradcam_{img}.png"
            cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))


        # # Average Grad-CAM across folds
        # avg_cam = np.mean(results['cams'], axis=0)

        # # Load and normalize image
        # # bgr_image = cv2.imread(f"{data_dir}/Monkeypox/monkeypox1.png")
        # # rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        # # rgb_image_float = rgb_image.astype(np.float32) / 255.0

        # # Visualize the average
        # visualization = show_cam_on_image(normalize_gradcam_iamge(f"{data_dir}/Monkeypox/monkeypox1.png"), avg_cam, use_rgb=True)

        # # Save the visualization
        # save_path = f"{output_dir}/gradcam_image.png"
        # cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run stratified cross-validation for ensemble deep learning')
    parser.add_argument('--data_dir', type=str, default='./MSID3', required=False, help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./output_resnet', help='Output directory for results')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50, help='Maximum number of training epochs')
    parser.add_argument('--learning_rate', type=int, default=1e-4, help='Learning rate for training')
    
    args = parser.parse_args()
    
    results = main(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_splits=args.n_splits,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate
    )