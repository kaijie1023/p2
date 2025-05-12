import os
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler,  Subset
from torchmetrics.classification import MulticlassAUROC
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from PIL import Image
import random
import time
from typing import List, Dict, Tuple, Any, Optional
import copy
import cv2
from lime import lime_image

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

# Custom Dataset
class CustomImageDataset(Dataset):
    """
    Custom dataset that loads images from a directory structure.
    
    Expected directory structure:
    root_dir/
        class_1/
            img1.jpg
            img2.jpg
            ...
        class_2/
            img1.jpg
            ...
        ...
    """
    
    def __init__(self, root_dir: str, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images organized in class folders
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        self.targets = []
        
        # Build dataset
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append(img_path)
                    self.targets.append(class_idx)
        
        self.targets = np.array(self.targets)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.samples[idx]
        target = self.targets[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if there's an error
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, target

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
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Load dataset
    print(f"Loading dataset from {data_dir}...")
    dataset = CustomImageDataset(root_dir=data_dir)
    print(f"Dataset loaded with {len(dataset)} samples and {len(dataset.classes)} classes")

    # Get targets
    targets = dataset.targets
    
    # Results storage
    results = {
        'fold_metrics': [],
        'models': [],
        'test_indices': [],
        'train_losses': [],
        'val_losses': []
    }
    
    # Loop through folds
    for fold, (train_index, test_index) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        print(f"\n{'='*50}\nFold {fold+1}/{n_splits}\n{'='*50}")

        # Split train into train and validation
        train_idx, val_idx = train_test_split(train_index, test_size=0.125, stratify=targets)

        # Create train and validation datasets with appropriate transforms
        train_dataset = datasets.ImageFolder(
            root=data_dir,
            transform=train_transforms
        )
        
        val_dataset = datasets.ImageFolder(
            root=data_dir, 
            transform=val_transforms
        )

        # Use indices to create subsets
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(val_dataset, val_idx)
        test_subset = Subset(val_dataset, test_index)
        
        # Create data loaders
        train_loader = DataLoader(
            train_subset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_subset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=2
        )

        test_loader = DataLoader(
            test_subset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=2
        )
        
        # Initialize model
        num_classes = len(dataset.classes)
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

        roc_auc_metric = MulticlassAUROC(num_classes=num_classes).to(device)
        roc_auc_metric.reset()
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

                probs = torch.softmax(outputs, dim=1)
                roc_auc_metric.update(probs.cpu(), labels.cpu())
        
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        roc_auc = roc_auc_metric.compute().item()
        
        fold_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }
        
        print(f"Test Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC: {roc_auc:.4f}")
        
        # Store results
        results['fold_metrics'].append(fold_metrics)
        results['models'].append(model)
        results['test_indices'].append(test_index)
        results['train_losses'].append(train_losses)
        results['val_losses'].append(val_losses)
    
    # Calculate overall metrics
    avg_metrics = {metric: np.mean([fold[metric] for fold in results['fold_metrics']]) 
                 for metric in results['fold_metrics'][0].keys()}
    
    results['avg_metrics'] = avg_metrics
    
    print("\nAverage Metrics Across All Folds:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return results

# Utility for splitting train indices into train and validation
def train_test_split(indices, test_size=0.2, stratify=None):
    """Split indices into training and test sets."""
    if stratify is None:
        # Random split
        test_size = int(len(indices) * test_size)
        np.random.shuffle(indices)
        return indices[test_size:], indices[:test_size]
    else:
        # Stratified split
        unique_classes = np.unique(stratify)
        train_indices = []
        val_indices = []

        for cls in unique_classes:
            cls_indices = indices[stratify[indices] == cls]
            n_val = int(len(cls_indices) * test_size)
            
            np.random.shuffle(cls_indices)
            val_indices.extend(cls_indices[:n_val])
            train_indices.extend(cls_indices[n_val:])

        return np.array(train_indices), np.array(val_indices)

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


class GradCAM_CNN:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        def forward_hook(module, input, output):
            self.activations = output
            # print("activations:", self.activations.shape)  # Expect [1, C, H, W] (e.g., 1x512x7x7)

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
            # print("gradients:", self.gradients.shape)

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, x, class_idx=None):
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax().item()
        loss = output[:, class_idx]
        self.model.zero_grad()
        loss.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Invert weights if needed (for ResNet)
        if type(self.model).__name__ == 'ResNet':
            weights = -weights

        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return cam
    
class GradCAM_MultiLayer:
    def __init__(self, model, target_layers):
        self.model = model.eval()
        self.target_layers = target_layers  # List of layers
        self.activations = {}
        self.gradients = {}

        # Register hooks for each layer
        for i, layer in enumerate(self.target_layers):
            layer.register_forward_hook(self._make_forward_hook(i))
            layer.register_backward_hook(self._make_backward_hook(i))

    def _make_forward_hook(self, name):
        def forward_hook(module, input, output):
            self.activations[name] = output
        return forward_hook

    def _make_backward_hook(self, name):
        def backward_hook(module, grad_input, grad_output):
            self.gradients[name] = grad_output[0]
        return backward_hook

    def generate(self, x, class_idx=None, combine_method='average'):
        self.model.zero_grad()
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax().item()
        loss = output[:, class_idx]
        loss.backward()

        cams = []

        for i in self.activations.keys():
            act = self.activations[i]
            grad = self.gradients[i]

            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = (weights * act).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
            cam = cam.squeeze()
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            cams.append(cam.detach().cpu().numpy())

        if combine_method == 'average':
            final_cam = sum(cams) / len(cams)
            return final_cam
        else:
            return cams  # return all individual CAMs

class GradCAM_DeiT:             
    def __init__(self, model, target_block):
        self.model = model.eval()
        self.target_block = target_block
        self.attn_output = None
        self.gradients = None

        def forward_hook(module, input, output):
            self.attn_output = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_block.register_forward_hook(forward_hook)
        self.target_block.register_backward_hook(backward_hook)

    def generate(self, x, class_idx=None):
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax().item()
        loss = output[:, class_idx]
        self.model.zero_grad()
        loss.backward()

        grads = self.gradients.mean(dim=1)
        cam = (self.attn_output * grads.unsqueeze(-1)).sum(dim=2)
        cam = cam[:, 1:]  # Skip [CLS] 
        cam = cam.reshape(1, 14, 14).detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return cam.squeeze()

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
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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

        # GradCAM
        cams_resnet = []
        cams_deit = []
        cams_combined = []

        for fold in range(n_splits):
            plt_fig = plot_losses(results['train_losses'][fold], results['val_losses'][fold], fold)
            plt_fig.savefig(os.path.join(output_dir, f'fold_{fold+1}_losses.png'))
            plt.close()

            # GradCAM
            # Load trained model for this fold
            ensemble_model = results['models'][fold]
            ensemble_model.eval()

            # Extract submodels
            resnet = ensemble_model.resnet
            # deit = ensemble_model.deit

            # Prepare target layers
            target_layer = resnet.layer4[-1].conv3
            # target_layer = resnet.layer4[1].conv2
            # target_layer_deit = deit.blocks[-1].attn
            # target_layers = [
            #     resnet.layer2[-1].conv3,
            #     resnet.layer3[-1].conv3,
            #     resnet.layer4[-1].conv3
            # ]

            # Build GradCAM handlers
            cam_dense = GradCAM_CNN(resnet, target_layer)
            # cam_deit = GradCAM_DeiT(deit, target_layer_deit)
            # cam_dense = GradCAM_MultiLayer(resnet, target_layers)

            # 4. Load and preprocess input image
            image_tensor = preprocess_image(f"{data_dir}/Monkeypox/monkeypox1.png").to(device)
            # image_tensor = preprocess_image(f"{data_dir}/Monkeypox/monkeypox262.png").to(device)

            # Generate CAMs for image
            cam_d = cam_dense.generate(image_tensor)
            # cam_v = cam_deit.generate(image_tensor)

            # Resize ViT CAM to match CNN CAM
            # cam_v_resized = cv2.resize(cam_v, cam_d.shape[::-1])

            # Average this fold’s submodel CAMs (optional)
            # cam_fold_avg = (cam_d) / 1

            # Store for ensemble averaging later
            cams_resnet.append(cam_d)
            # cams_deit.append(cam_v_resized)
            cams_combined.append(cam_d)

            # Save intermediate fold CAMs
            save_cam_on_image(image_tensor.cpu(), cam_d, f"{output_dir}/cam_resnet_fold{fold}.jpg", title=f"resnet Fold {fold}")
            # save_cam_on_image(image_tensor.cpu(), cam_v_resized, f"{output_dir}/cam_deit_fold{fold}.jpg", title=f"DeiT Fold {fold}")
            # save_cam_on_image(image_tensor.cpu(), cam_fold_avg, f"{output_dir}/cam_ensemble_fold{fold}.jpg", title=f"Ensemble Fold {fold}")
        
        # Save average metrics
        metrics_df = pd.DataFrame([results['avg_metrics']])
        metrics_df.to_csv(os.path.join(output_dir, 'avg_metrics.csv'), index=False)
        
        # Save individual fold metrics
        fold_metrics_df = pd.DataFrame(results['fold_metrics'])
        fold_metrics_df.to_csv(os.path.join(output_dir, 'fold_metrics.csv'), index=False)


        # plt.figure(figsize=(6, 6))
        # plt.imshow(results['lime_mask'], cmap='hot')
        # plt.title("Aggregated LIME Mask Across K-Folds (Post-Training)")
        # plt.axis('off')
        # plt.colorbar()

        # plt.tight_layout()
        # plt.savefig(f"{output_dir}/aggregated_lime_heatmap.jpg", dpi=300, bbox_inches='tight')
        # plt.close()

        # Ensemble GradCAM across folds
        final_cam = np.mean(np.stack(cams_combined), axis=0)
        save_cam_on_image(image_tensor.cpu(), final_cam, f"{output_dir}/cam_ensemble_kfold.jpg", title="K-Fold Ensemble CAM")

    
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