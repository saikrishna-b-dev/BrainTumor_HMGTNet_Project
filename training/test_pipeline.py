import torch
import torch.nn as nn
from training.train import train_one_epoch, validate
from models.hmgt_net import HMGTNet
from training.dataset import BraTSDataset, get_transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import json
import os

def test_pipeline():
    print("Starting smoke test of HMGT-Net pipeline...")
    device = torch.device("cpu")
    
    # 1. Load test data
    SPLIT_PATH = r"d:\BrainTumor_HMGTNet_Project\data\processed\dataset_split_test.json"
    with open(SPLIT_PATH, 'r') as f:
        split_info = json.load(f)
    
    train_dataset = BraTSDataset(split_info["train"], transform=get_transforms("train"))
    val_dataset = BraTSDataset(split_info["val"], transform=get_transforms("val"))
    
    train_loader = DataLoader(train_dataset, batch_size=1)
    val_loader = DataLoader(val_dataset, batch_size=1)
    
    # 2. Model, Criterion, Optimizer
    model = HMGTNet(in_channels=4, num_classes=4, img_size=(128, 128, 128)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler(enabled=False) # Disable for CPU test
    
    # 3. Run 1 epoch
    print("Running 1 training epoch...")
    train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
    print(f"Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}")
    
    print("Running 1 validation epoch...")
    val_loss, val_metrics, _, _ = validate(model, val_loader, criterion, device)
    print(f"Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}")
    
    print("Smoke test PASSED!")

if __name__ == "__main__":
    test_pipeline()
