import os
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassAUROC, MulticlassConfusionMatrix
from torchvision import datasets, transforms
from torchvision.models import densenet121, DenseNet121_Weights
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib.pyplot as plt
from PIL import Image
import random
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
    '/Monkeypox/monkeypox100.png',
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
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
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
        self.densenet = densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with weighted ensemble."""
        out = self.densenet(x)
        return out
    
    def name(self) -> str:
        return "densenet"

class MultiScaleDenseNet(nn.Module):
    def __init__(self, num_classes=3):
        super(MultiScaleDenseNet, self).__init__()
        
        # Load pretrained DenseNet
        base_model = densenet121(weights=DenseNet121_Weights.DEFAULT)
        
        # Extract the features before each dense block
        self.initial_layers = nn.Sequential(
            base_model.features.conv0,
            base_model.features.norm0,
            base_model.features.relu0,
            base_model.features.pool0
        )
        
        # Extract dense blocks
        self.block1 = base_model.features.denseblock1
        self.trans1 = base_model.features.transition1
        self.block2 = base_model.features.denseblock2
        self.trans2 = base_model.features.transition2
        self.block3 = base_model.features.denseblock3
        self.trans3 = base_model.features.transition3
        self.block4 = base_model.features.denseblock4
        
        # Final processing
        self.final_norm = base_model.features.norm5
        
        # Feature dimension for each block output (depends on model)
        self.scale1_dim = 256  # After block1
        self.scale2_dim = 512  # After block2
        self.scale3_dim = 1024 # After block3
        self.scale4_dim = 1024 # After block4
        
        # Multi-scale integration
        self.adaptation_layers = nn.ModuleDict({
            'scale1': nn.Conv2d(self.scale1_dim, 256, kernel_size=1),
            'scale2': nn.Conv2d(self.scale2_dim, 256, kernel_size=1),
            'scale3': nn.Conv2d(self.scale3_dim, 256, kernel_size=1),
            'scale4': nn.Conv2d(self.scale4_dim, 256, kernel_size=1)
        })
        
        # Upsampling layers
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256 * 4, num_classes)  # 256 * 4 because we concatenate 4 scales
        )
        
    def forward(self, x):
        x = self.initial_layers(x)
        
        # Scale 1 features
        scale1_features = self.block1(x)
        scale1_adapted = self.adaptation_layers['scale1'](scale1_features)
        
        # Scale 2 features
        x = self.trans1(scale1_features)
        scale2_features = self.block2(x)
        scale2_adapted = self.adaptation_layers['scale2'](scale2_features)
        
        # Scale 3 features
        x = self.trans2(scale2_features)
        scale3_features = self.block3(x)
        scale3_adapted = self.adaptation_layers['scale3'](scale3_features)
        
        # Scale 4 features (final dense block)
        x = self.trans3(scale3_features)
        scale4_features = self.block4(x)
        scale4_adapted = self.adaptation_layers['scale4'](scale4_features)
        
        # Ensure all feature maps have same spatial dimensions through upsampling
        target_size = scale1_adapted.size()[2:]
        scale2_adapted = nn.functional.interpolate(scale2_adapted, size=target_size, mode='bilinear', align_corners=True)
        scale3_adapted = nn.functional.interpolate(scale3_adapted, size=target_size, mode='bilinear', align_corners=True)
        scale4_adapted = nn.functional.interpolate(scale4_adapted, size=target_size, mode='bilinear', align_corners=True)
        
        # Concatenate features from all scales
        multi_scale_features = torch.cat([
            scale1_adapted, 
            scale2_adapted, 
            scale3_adapted, 
            scale4_adapted
        ], dim=1)
        
        # Classification
        output = self.classifier(multi_scale_features)
        
        return output
    
    def load(self, path: str, map_location: Optional[str] = None, weights_only: bool = False) -> None:
        """
        Load model weights from a file.
        
        Args:
            path (str): Path to the model weights file.
            map_location (Optional[str]): Device to map the weights to.
            weights_only (bool): If True, only load the state_dict.
        """
        if weights_only:
            self.load_state_dict(torch.load(path, map_location=map_location))
        else:
            checkpoint = torch.load(path, map_location=map_location)
            self.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model weights loaded from {path}")

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
        'best_accuracy': [],
        'best_model': None,
        'cams': {}
    }

    # GradCAM images key map to image id
    for i in gradcam_images:
        results['cams'][get_image_id(i)] = []

    # Prepare data for k-fold
    y_true = []
    y_pred = []
    
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
        model = MultiScaleDenseNet(num_classes=num_classes).to(device)
        
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
        
        with torch.no_grad():
            true_labels = []
            preds = []
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                y_true.extend(labels)
                true_labels.append(labels)
                preds.append(torch.softmax(outputs, dim=1))

            y_pred.extend(torch.cat(preds))

            acc = MulticlassAccuracy(num_classes=num_classes).to(device)
            acc.update(torch.cat(preds).argmax(dim=1), torch.cat(true_labels))
            acc = acc.compute().item()
            print(f"Val Accuracy: {acc:.4f}")
            if results['best_model'] is None:
                results['best_model'] = model
                results['best_accuracy'] = acc
                print('Best model inserted.')
            elif acc > results['best_accuracy']:
                results['best_model'] = model
                results['best_accuracy'] = acc
                print('Best model updated.')
               
        
        target_layer = model.block4
        cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])


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
        
        # Store results
        results['models'].append(model)
        results['train_losses'].append(train_losses)
        results['val_losses'].append(val_losses)
    
    print("\nAverage Metrics Across All Folds:")

    metrics = get_metrics(y_true=y_true, y_pred=y_pred, num_classes=num_classes)

    results['metrics'] = metrics

    print(f"Test Metrics - Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}, ROC AUC: {metrics['roc_auc']:.4f}")
    
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

def get_metrics(y_true, y_pred, num_classes):
    """
    Calculate various metrics for model evaluation.
    """
    stack_preds = torch.stack(y_pred)
    final_preds = torch.argmax(stack_preds, dim=1)
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
    roc_auc.update(stack_preds, final_labels)
    roc_auc = roc_auc.compute().item()

    cm = MulticlassConfusionMatrix(num_classes=num_classes).to(device)
    cm.update(final_preds, final_labels)
    cm = cm.compute()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }

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
        
        for img in results['cams']:
            avg_cam = np.mean(results['cams'][img], axis=0)

            # Visualize the average
            visualization = show_cam_on_image(normalize_gradcam_iamge(f"{data_dir}/Monkeypox/{img}.png"), avg_cam, use_rgb=True)

            # Save the visualization
            save_path = f"{output_dir}/gradcam_{img}.png"
            cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

        # Visualization of confusion matrix
        plt.figure(figsize=(6, 5))
        sns.heatmap(results['metrics']['confusion_matrix'].cpu(), annot=True, fmt="d", cmap="Blues", xticklabels=[f"Pred {i}" for i in range(3)],
                    yticklabels=[f"True {i}" for i in range(3)])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Multiclass Confusion Matrix")
        plt.tight_layout()

        # Save the figure
        plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300)
        plt.close()

    

    ### Retrain the best model on the entire dataset
    print("\nRetraining the best model on the entire dataset...")
    # Load dataset
    image_paths, labels, class_names = load_dataset_from_directory(data_dir)

    # First, split into train and val with stratification
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, labels, test_size=0.1, stratify=labels, random_state=42
    )

    # Create train dataset and loader
    train_dataset = ImageDataset(X_train, y_train, transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create val dataset and loader
    val_dataset = ImageDataset(X_val, y_val, transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = results['best_model']
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler and early stopping
    scheduler = LRScheduler(optimizer, patience=5)
    early_stopping = EarlyStopping(patience=7)
    
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

    model_path = os.path.join(output_dir, 'densenet121_msi.pth')
    torch.save(model.state_dict(), model_path)
    print("Training complete. Results saved.")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run stratified cross-validation for ensemble deep learning')
    parser.add_argument('--data_dir', type=str, default='./MSID3', required=False, help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./output_densenet_ms', help='Output directory for results')
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