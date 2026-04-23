import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import time
import numpy as np
from tqdm import tqdm
import glob

from configs.config import Config
from models.hmgt_net import HMGTNet
from training.dataset import get_dataloader
from utils.utils import Logger, calculate_metrics, save_confusion_matrix, save_gradcam_heatmap, EarlyStopping

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
        return focal_loss.mean()

class HybridLoss(nn.Module):
    def __init__(self, ce_weight=0.5, focal_weight=0.5, label_smoothing=0.1):
        super(HybridLoss, self).__init__()
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.focal = FocalLoss(gamma=2.0)

    def forward(self, inputs, targets):
        return self.ce_weight * self.ce(inputs, targets) + self.focal_weight * self.focal(inputs, targets)

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        # Reduce input size to save memory
        if images.shape[-3:] != (64, 64, 64):
            images = F.interpolate(images, size=(64, 64, 64), mode='trilinear', align_corners=False)

        optimizer.zero_grad()
        
        with autocast(device_type='cuda' if 'cuda' in str(device) else 'cpu', enabled=Config.USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        if Config.GRAD_CLIP > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
        
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = calculate_metrics(all_labels, all_preds)
    return epoch_loss, metrics

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Validation"):
        images, labels = images.to(device), labels.to(device)
        
        # Reduce input size to save memory
        if images.shape[-3:] != (64, 64, 64):
            images = F.interpolate(images, size=(64, 64, 64), mode='trilinear', align_corners=False)
        
        with autocast(device_type='cuda' if 'cuda' in str(device) else 'cpu', enabled=Config.USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = calculate_metrics(all_labels, all_preds)
    return epoch_loss, metrics, all_preds, all_labels

def manage_top_k_checkpoints(current_score, epoch, model_state, k=3):
    """Saves the top K models and deletes older/worse ones to save space."""
    ckpt_dir = Config.CHECKPOINT_DIR
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Save current
    current_path = os.path.join(ckpt_dir, f"model_f1_{current_score:.4f}_ep_{epoch}.pth")
    torch.save(model_state, current_path)
    
    # Keep only top K
    all_ckpts = glob.glob(os.path.join(ckpt_dir, "model_f1_*.pth"))
    
    # Extract scores
    scores = []
    for p in all_ckpts:
        try:
            score = float(os.path.basename(p).split('_')[2])
            scores.append((score, p))
        except:
            pass
            
    scores.sort(reverse=True) # Highest first
    
    # Delete everything beyond top K
    for _, p in scores[k:]:
        if os.path.exists(p):
            os.remove(p)

    # Save the absolute best as best_model.pth for quick reference
    if len(scores) > 0:
        best_path = scores[0][1]
        torch.save(torch.load(best_path), os.path.join(ckpt_dir, "best_model.pth"))

def main():
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.HEATMAP_DIR, exist_ok=True)

    logger = Logger(os.path.join(Config.LOG_DIR, "training_log.csv"))
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Datasets & Dataloaders
    split_file = Config.SPLIT_JSON
    if not os.path.exists(split_file):
        print(f"\n[ERROR] Dataset split file not found at: {split_file}")
        print("Please ensure you have placed the raw NIfTI files in 'data/raw/' and ran the preprocessing script.")
        sys.exit(1)

    train_loader = get_dataloader(split_file, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, mode="train")
    val_loader = get_dataloader(split_file, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, mode="val")

    # 2. Model setup
    model = HMGTNet(
        in_channels=Config.IN_CHANNELS, 
        num_classes=Config.NUM_CLASSES, 
        img_size=Config.IMG_SIZE
    ).to(device)

    # 3. Loss & Optimizer
    criterion = HybridLoss(label_smoothing=Config.LABEL_SMOOTHING)
    
    if Config.OPTIMIZER == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    else:
        optimizer = optim.Adam(model.parameters(), lr=Config.LR)

    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=Config.T_0, T_mult=Config.T_MULT, eta_min=1e-6
    )

    scaler = GradScaler(device.type, enabled=Config.USE_AMP)
    early_stopping = EarlyStopping(patience=Config.EARLY_STOPPING_PATIENCE, delta=0.001)

    best_f1 = 0.0

    # 4. Training Loop
    for epoch in range(1, Config.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{Config.EPOCHS}")
        
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_metrics, val_preds, val_labels = validate(model, val_loader, criterion, device)
        
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1_macro']:.4f}")

        # Logging
        logger.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_metrics["accuracy"],
            "val_loss": val_loss,
            "val_acc": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
            "lr": optimizer.param_groups[0]['lr']
        })

        # Save checkpoint logic (Top 3)
        current_f1 = val_metrics["f1_macro"]
        if current_f1 > best_f1:
            best_f1 = current_f1
            print(f"New best F1-score: {best_f1:.4f}.")
            
            # Save Confusion Matrix for the best epoch
            cm_path = os.path.join(Config.LOG_DIR, f"confusion_matrix_epoch_{epoch}.png")
            save_confusion_matrix(val_labels, val_preds, ["Glioma", "Meningioma", "No Tumor", "Pituitary"], cm_path)
            
            # Save Grad-CAM for a sample from validation
            sample_img, sample_label = next(iter(val_loader))
            sample_img = sample_img[:1].to(device)
            if sample_img.shape[-3:] != (64, 64, 64):
                sample_img = F.interpolate(sample_img, size=(64, 64, 64), mode='trilinear', align_corners=False)
            
            cam_path = os.path.join(Config.HEATMAP_DIR, f"gradcam_epoch_{epoch}.png")
            save_gradcam_heatmap(model, sample_img, sample_label[0].item(), cam_path)

        # Always track top 3 checkpoints
        manage_top_k_checkpoints(current_f1, epoch, model.state_dict(), k=3)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

if __name__ == "__main__":
    main()
