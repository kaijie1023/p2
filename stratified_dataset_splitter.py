import os
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

class StratifiedDatasetSplitter:
    def __init__(
        self, 
        data_dir: str, 
        test_size: float = 0.2, 
        val_size: float = 0.2, 
        random_state: int = 42,
        transform: transforms.Compose = None
    ):
        """
        Initialize stratified dataset splitter for image classification tasks.
        
        Args:
            data_dir (str): Path to root directory containing subdirectories of classes
            test_size (float): Proportion of dataset to reserve for testing
            val_size (float): Proportion of training data to use for validation
            random_state (int): Seed for reproducibility
            transform (transforms.Compose): Optional image transformations
        """
        self.data_dir = data_dir
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.transform = transform or self._default_transform()

    def _default_transform(self) -> transforms.Compose:
        """
        Provide default image transformations if none are specified.
        
        Returns:
            transforms.Compose: Default image transformation pipeline
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _get_class_labels(self) -> List[str]:
        """
        Extract class labels from subdirectories.
        
        Returns:
            List[str]: List of class names
        """
        return [d for d in os.listdir(self.data_dir) 
                if os.path.isdir(os.path.join(self.data_dir, d))]

    def split_dataset(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Perform stratified splitting of dataset.
        
        Returns:
            Tuple of train, validation, and test datasets
        """
        # Get class labels
        classes = self._get_class_labels()
        
        # Collect all image paths and corresponding labels
        image_paths = []
        labels = []
        
        for class_idx, class_name in enumerate(classes):
            class_dir = os.path.join(self.data_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                image_paths.append(img_path)
                labels.append(class_idx)
        
        # First split: separate test set
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            image_paths, 
            labels, 
            test_size=self.test_size, 
            stratify=labels, 
            random_state=self.random_state
        )
        
        # Second split: separate train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, 
            y_train_val, 
            test_size=self.val_size, 
            stratify=y_train_val, 
            random_state=self.random_state
        )
        
        # Create custom datasets
        train_dataset = CustomImageDataset(X_train, y_train, classes, transform=self.transform)
        val_dataset = CustomImageDataset(X_val, y_val, classes, transform=self.transform)
        test_dataset = CustomImageDataset(X_test, y_test, classes, transform=self.transform)
        
        return train_dataset, val_dataset, test_dataset

    def create_dataloaders(
        self, 
        batch_size: int = 32, 
        num_workers: int = 4
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create dataloaders for train, validation, and test datasets.
        
        Args:
            batch_size (int): Number of samples per batch
            num_workers (int): Number of subprocess workers for data loading
        
        Returns:
            Tuple of train, validation, and test dataloaders
        """
        train_dataset, val_dataset, test_dataset = self.split_dataset()
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers
        )
        
        return train_loader, val_loader, test_loader

class CustomImageDataset(Dataset):
    """
    Custom PyTorch dataset for handling image paths and labels.
    """
    def __init__(
        self, 
        image_paths: List[str], 
        labels: List[int], 
        class_names: List[str], 
        transform: transforms.Compose = None
    ):
        """
        Initialize custom image dataset.
        
        Args:
            image_paths (List[str]): List of image file paths
            labels (List[int]): Corresponding integer labels
            class_names (List[str]): List of class names
            transform (transforms.Compose): Image transformations
        """
        self.image_paths = image_paths
        self.labels = labels
        self.class_names = class_names
        self.transform = transform or transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Retrieve single image and its label.
        
        Args:
            idx (int): Index of image to retrieve
        
        Returns:
            Tuple of transformed image tensor and label
        """
        from PIL import Image
        
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        label = self.labels[idx]
        
        return image, label

# Example usage
def main():
    # Configuration
    data_directory = 'path/to/your/dataset'
    
    # Custom transforms (optional)
    custom_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Initialize splitter
    splitter = StratifiedDatasetSplitter(
        data_dir=data_directory, 
        test_size=0.2, 
        val_size=0.2, 
        random_state=42,
        transform=custom_transform
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = splitter.create_dataloaders(
        batch_size=32, 
        num_workers=4
    )
    
    # Print dataset information
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")

if __name__ == "__main__":
    main()