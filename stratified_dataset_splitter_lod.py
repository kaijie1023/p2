import torch
from torch.utils.data import DataLoader
from torchvision import transforms

def get_dataset_transforms(split: str = 'train') -> transforms.Compose:
    """
    Generate appropriate transforms for different dataset splits
    
    Parameters:
    -----------
    split : str
        Type of dataset split ('train', 'validation', or 'test')
    
    Returns:
    --------
    transforms.Compose
        Appropriate image transformations for the split
    """
    # Base transformations common to all splits
    base_transforms = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
    
    if split == 'train':
        # Train-specific transforms with data augmentation
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2, 
                contrast=0.2, 
                saturation=0.2
            ),
            transforms.RandomRotation(10),
        ] + base_transforms)
        return train_transforms
    
    elif split in ['validation', 'test']:
        # Validation and test transforms (minimal preprocessing)
        val_test_transforms = transforms.Compose(base_transforms)
        return val_test_transforms
    
    else:
        raise ValueError(f"Invalid split type: {split}")

class StratifiedDatasetSplitter:
    def __init__(
        self, 
        root_dir: str, 
        random_state: int = 42,
        train_size: float = 0.7,
        val_size: float = 0.1,
        test_size: float = 0.2
    ):
        """
        Modified to use split-specific transforms
        """
        self.root_dir = root_dir
        self.random_state = random_state
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
    
    def split_dataset(
        self, 
        batch_size: int = 32,
        num_workers: int = 4
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create DataLoaders with split-specific transforms
        """
        # Get split-specific transforms
        train_transform = get_dataset_transforms('train')
        val_transform = get_dataset_transforms('validation')
        test_transform = get_dataset_transforms('test')
        
        # Create datasets with appropriate transforms
        train_dataset = DirectoryImageDataset(
            self.root_dir, 
            transform=train_transform
        )
        val_dataset = DirectoryImageDataset(
            self.root_dir, 
            transform=val_transform
        )
        test_dataset = DirectoryImageDataset(
            self.root_dir, 
            transform=test_transform
        )
        
        # (Rest of the splitting logic remains the same)
        # ... (previous implementation of splitting)
        
        # Create DataLoaders with split-specific datasets
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader