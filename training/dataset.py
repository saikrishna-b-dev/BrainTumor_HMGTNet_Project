import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    EnsureTyped,
    ToTensord,
    NormalizeIntensityd,
    RandAffined,
    Rand3DElasticd,
    RandAdjustContrastd,
    RandSpatialCropd,
    Resized
)
from torch.utils.data import WeightedRandomSampler, Dataset
import torch
import numpy as np
import json

class BraTSDataset(Dataset):
    """
    PyTorch Dataset for loading preprocessed BraTS data.
    """
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        image = np.load(item["image"])  # (4, 128, 128, 128)
        
        # Dummy classification mapping
        label = int(hash(item["patient_id"]) % 4) 

        data = {
            "image": torch.from_numpy(image.astype(np.float32)),
            "label": torch.tensor(label, dtype=torch.long)
        }

        if self.transform:
            data = self.transform(data)

        return data["image"], data["label"]

def get_transforms(mode="train"):
    """
    Get advanced MONAI transforms for training and validation.
    """
    if mode == "train":
        return Compose([
            # Advanced Preprocessing
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            
            # Strong Augmentation
            RandAffined(
                keys=["image"], prob=0.5, 
                rotate_range=(0.26, 0.26, 0.26), # ~15 degrees in radians
                scale_range=(0.1, 0.1, 0.1)
            ),
            Rand3DElasticd(
                keys=["image"], prob=0.3,
                sigma_range=(5, 7), magnitude_range=(50, 150)
            ),
            RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.5, 2.0)),
            
            # Random Crop (assuming 64x64x64 focus)
            RandSpatialCropd(keys=["image"], roi_size=(64, 64, 64), random_center=True, random_size=False),
            
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            EnsureTyped(keys=["image"])
        ])
    else:
        return Compose([
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            Resized(keys=["image"], spatial_size=(64, 64, 64)),
            EnsureTyped(keys=["image"])
        ])

def get_dataloader(split_file, batch_size=2, num_workers=4, mode="train"):
    """
    Helper function to create DataLoader with weighted sampling for training.
    """
    with open(split_file, 'r') as f:
        split_info = json.load(f)

    if mode not in split_info:
        raise ValueError(f"Mode {mode} not found in split file.")

    dataset = BraTSDataset(split_info[mode], transform=get_transforms(mode))
    
    # Weighted Sampling for class imbalance
    if mode == "train":
        labels = [int(hash(item["patient_id"]) % 4) for item in dataset.data_list]
        class_counts = np.bincount(labels, minlength=4)
        # Avoid division by zero
        class_weights = 1.0 / np.where(class_counts > 0, class_counts, 1)
        sample_weights = [class_weights[l] for l in labels]
        
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset), replacement=True)
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return loader

if __name__ == "__main__":
    # Example usage
    SPLIT_PATH = r"d:\BrainTumor_HMGTNet_Project\data\processed\dataset_split.json"
    
    if os.path.exists(SPLIT_PATH):
        train_loader = get_dataloader(SPLIT_PATH, mode="train", batch_size=1)
        for images, labels in train_loader:
            print(f"Batch - Image shape: {images.shape}, Label shape: {labels.shape}")
            break
    else:
        print("Split file not found. Run preprocess_brats.py first.")
