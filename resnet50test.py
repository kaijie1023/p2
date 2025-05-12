import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import shap

from PIL import Image
import copy
import time
import random

import albumentations as A
from albumentations.pytorch import ToTensorV2

output_dir = "./output_resnet50"

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
        # image = Image.open(img_path).convert('RGB')
        image = np.array(Image.open(img_path).convert('RGB'))
        label = self.labels[idx]
        
        if self.transform:
            # image = self.transform(image)
            augmented = self.transform(image=image)
            image = augmented['image']  # get the transformed tensor
            
        return image, label

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

# Define data transforms
def get_transforms():
    # train_transform = transforms.Compose([
    #     transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(10),
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    #     transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    #     transforms.ToTensor(),
    #     transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    train_transform = A.Compose([
        # A.RandomResizedCrop(224, 224, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=1.0),
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
        
        # Color & brightness variations
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        
        # Blur, noise, and clinical artifact simulation
        A.OneOf([
            A.MotionBlur(blur_limit=3),
            A.MedianBlur(blur_limit=3),
            A.GaussianBlur(blur_limit=3),
            A.GaussNoise(var_limit=(10.0, 50.0))
        ], p=0.3),
        
        # Simulate occlusion or missing parts
        A.CoarseDropout(max_holes=2, max_height=32, max_width=32, fill_value=0, p=0.4),
        
        # Perspective / affine distortion
        A.Affine(scale=(0.95, 1.05), translate_percent=(0.02, 0.05), p=0.3),

        # Normalize and convert to tensor
        A.Normalize(mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    # val_transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    return train_transform, val_transform

# Create ResNet50 model
class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Freeze all layers
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    def forward(self, x):
        # return self.resnet(x).clone()
        return self.resnet(x)
    
def create_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Freeze early layers
    # for param in list(model.parameters())[:-30]:
    #     param.requires_grad = False
    
    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

# Training function with early stopping
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, patience=5):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    counter = 0
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Record loss and accuracy
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.item())
                
                # Update learning rate scheduler
                scheduler.step(epoch_loss)
                print(f"Current lr: {scheduler.get_last_lr()}")
                
                # Deep copy the model if best accuracy
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    counter = 0
                else:
                    counter += 1
        
        # Early stopping
        if counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
            
        print()
        
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{output_dir}/training_curves.png")
    plt.close()
    
    return model

# Evaluate the model
def evaluate_model(model, test_loader, class_names, save=False):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
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
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    # plt.show()
    if (save):
        plt.savefig(f"{output_dir}/confusion_matrix.png")
        plt.close()
        
    # ROC Curve (for binary classification)
    if len(class_names) == 2:
        fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        # plt.show()
        if (save):
            plt.savefig(f"{output_dir}/ROC.png")
            plt.close()
    # For multiclass, we can plot ROC curve for each class
    else:
        plt.figure(figsize=(10, 8))
        for i, class_name in enumerate(class_names):
            # One-vs-Rest approach
            y_true_binary = (all_labels == i).astype(int)
            y_scores = all_probs[:, i]
            
            fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw=2, label=f'{class_name} (area = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class ROC Curve (One-vs-Rest)')
        plt.legend(loc="lower right")
        # plt.show()
        if (save):
            plt.savefig(f"{output_dir}/ROC.png")
            plt.close()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        # 'roc': roc_auc,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'true_labels': all_labels,
        'probabilities': all_probs
    }

# LIME explanation function
def explain_with_lime(model, image_paths, labels, class_names, transform, num_samples=5):
    # Function to get model predictions for LIME
    def get_predictions(images):
        # batch = torch.stack([transform(image=Image.fromarray(img.astype('uint8')).convert('RGB')) for img in images])
        batch = torch.stack([
            transform(image=img.astype('uint8'))["image"] for img in images
        ])
        batch = batch.to(device)
        with torch.no_grad():
            outputs = model(batch)
            probs = torch.nn.functional.softmax(outputs, dim=1)
        return probs.cpu().numpy()
    
    # Create LIME explainer
    explainer = lime_image.LimeImageExplainer()
    
    # Select random images to explain
    indices = np.random.choice(len(image_paths), min(num_samples, len(image_paths)), replace=False)
    
    plt.figure(figsize=(20, 4 * num_samples))
    
    for i, idx in enumerate(indices):
        # Load image
        img_path = image_paths[idx]
        img = np.array(Image.open(img_path).convert('RGB'))
        true_label = labels[idx]
        
        # Get explanation
        explanation = explainer.explain_instance(
            img, 
            get_predictions, 
            top_labels=1, 
            hide_color=0, 
            num_samples=1000
        )
        
        # Get the explanation for the true label
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            # true_label,
            positive_only=False, 
            num_features=5, 
            hide_rest=False
        )
        
        # Create visualization
        plt.subplot(num_samples, 2, i*2+1)
        plt.imshow(img)
        plt.title(f"Original Image: {class_names[true_label]}")
        plt.axis('off')
        
        plt.subplot(num_samples, 2, i*2+2)
        plt.imshow(mark_boundaries(temp, mask))
        plt.title(f"LIME Explanation for {class_names[true_label]}")
        plt.axis('off')
    
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{output_dir}/lime.png")
    plt.close()

# SHAP explanation function
def explain_with_shap(model, test_loader, class_names, num_samples=5):
    # Get a batch of images
    batch_images = []
    batch_labels = []
    
    for images, labels in test_loader:
        batch_images.append(images)
        batch_labels.append(labels)
        if len(batch_images) * images.size(0) >= num_samples:
            break
    
    # Concatenate batches
    images = torch.cat(batch_images)[:num_samples]
    labels = torch.cat(batch_labels)[:num_samples]
    
    images = images.to(device)
    
    # Define a function that returns the output of the model
    def model_output(x):
        with torch.no_grad():
            return model(x).cpu().numpy()
    
    # Initialize the SHAP explainer
    explainer = shap.DeepExplainer(model, images[:num_samples])
    
    # Compute SHAP values
    shap_values = explainer.shap_values(images[:num_samples])
    
    # Plot the SHAP values
    plt.figure(figsize=(12, 6))
    for i in range(min(3, num_samples)):  # Show 3 examples
        plt.subplot(1, 3, i+1)
        # Convert from torch tensor format to numpy for visualization
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        # De-normalize the image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        # Get the predicted class
        with torch.no_grad():
            outputs = model(images[i].unsqueeze(0))
            _, pred = torch.max(outputs, 1)
            pred_class = pred.item()
        
        # Display the explanation for the predicted class
        shap.image_plot(shap_values[pred_class][i], img)
        plt.title(f"SHAP Explanation for {class_names[pred_class]}")
    
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{output_dir}/shap.png")
    plt.close()

# Main function to run the entire pipeline
def run_resnet50_pipeline(data_dir, num_folds=5, batch_size=32, num_epochs=25, patience=5, learning_rate=0.001):
    # Load dataset
    image_paths, labels, class_names = load_dataset_from_directory(data_dir)
    num_classes = len(class_names)
    
    print(f"Loaded {len(image_paths)} images across {num_classes} classes: {class_names}")
    
    # First, split into test and train+val with stratification
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    # Get data transforms
    train_transform, val_transform = get_transforms()
    
    # Create test dataset and loader
    test_dataset = ImageDataset(X_test, y_test, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize k-fold cross validation
    kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    # Lists to store metrics across folds
    fold_metrics = []
    
    # Perform k-fold cross validation
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_val, y_train_val)):
        print(f"\n{'='*20} Fold {fold+1}/{num_folds} {'='*20}")
        
        # Get fold training and validation data
        X_train = [X_train_val[i] for i in train_idx]
        y_train = [y_train_val[i] for i in train_idx]
        X_val = [X_train_val[i] for i in val_idx]
        y_val = [y_train_val[i] for i in val_idx]
        
        # Create datasets and dataloaders for this fold
        train_dataset = ImageDataset(X_train, y_train, transform=train_transform)
        val_dataset = ImageDataset(X_val, y_val, transform=val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        dataloaders = {
            'train': train_loader,
            'val': val_loader
        }
        
        # Create model
        model = ResNet50(num_classes=num_classes)
        model = model.to(device)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        
        # Train model
        print(f"Training fold {fold+1}...")
        model = train_model(model, dataloaders, criterion, optimizer, scheduler, 
                         num_epochs=num_epochs, patience=patience)
        
        # Evaluate model on the validation set of this fold
        print(f"Evaluating fold {fold+1}...")
        fold_results = evaluate_model(model, val_loader, class_names)
        fold_metrics.append(fold_results)
        
        # Only perform explainability analysis on the last fold to avoid redundancy
        if fold == num_folds - 1:
            print("Generating LIME explanations...")
            try:
                explain_with_lime(model, X_val, y_val, class_names, val_transform, num_samples=3)
            except ImportError:
                print("skimage not available. Install it for LIME visualization.")
            
            print("Generating SHAP explanations...")
            try:
                explain_with_shap(model, val_loader, class_names, num_samples=3)
            except Exception as e:
                print(f"Error in SHAP explanation: {str(e)}")
    
    # Summarize k-fold results
    print("\n" + "="*50)
    print("K-Fold Cross Validation Results:")
    
    # Calculate average metrics across folds
    avg_accuracy = np.mean([metrics['accuracy'] for metrics in fold_metrics])
    avg_precision = np.mean([metrics['precision'] for metrics in fold_metrics])
    avg_recall = np.mean([metrics['recall'] for metrics in fold_metrics])
    avg_f1 = np.mean([metrics['f1'] for metrics in fold_metrics])
    # avg_roc = np.mean([metrics['roc'] for metrics in fold_metrics])
    
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")
    # print(f"Average ROC AUC: {avg_roc:.4f}")
    
    # Final evaluation on the test set using the best model from the last fold
    print("\n" + "="*50)
    print("Final Evaluation on Test Set:")
    test_results = evaluate_model(model, test_loader, class_names, save=True)
    
    # Save the final model
    torch.save(model.state_dict(), 'resnet50_final_model.pth')
    print("Final model saved as 'resnet50_final_model.pth'")
    
    return model, test_results

# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./MSID3', required=False, help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./output_resnet', help='Output directory for results')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50, help='Maximum number of training epochs')
    parser.add_argument('--learning_rate', type=int, default=1e-3, help='Learning rate for training')

    args = parser.parse_args()

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Run the pipeline
    model, results = run_resnet50_pipeline(
        data_dir=args.data_dir,
        num_folds=args.n_splits,  # Number of folds for cross validation
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        patience=5,
        learning_rate=args.learning_rate,
    )