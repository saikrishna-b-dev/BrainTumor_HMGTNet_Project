import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
from monai.visualize import GradCAM
import torch.nn.functional as F

class Logger:
    def __init__(self, log_path):
        self.log_path = log_path
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                f.write("epoch,train_loss,train_acc,val_loss,val_acc,val_f1_macro,lr\n")

    def log(self, metrics_dict):
        with open(self.log_path, "a") as f:
            f.write(f"{metrics_dict.get('epoch')},"
                    f"{metrics_dict.get('train_loss'):.4f},"
                    f"{metrics_dict.get('train_acc'):.4f},"
                    f"{metrics_dict.get('val_loss'):.4f},"
                    f"{metrics_dict.get('val_acc'):.4f},"
                    f"{metrics_dict.get('val_f1_macro'):.4f},"
                    f"{metrics_dict.get('lr'):.6f}\n")

def calculate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_macro": f1_macro
    }

def save_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def save_gradcam_heatmap(model, input_tensor, target_class, save_path):
    """
    Generate and save Grad-CAM heatmap for 3D volumes (middle slice visualization).
    """
    model.eval()
    
    # Try to target the CNN branch. Since we changed to ResNet50:
    # ResNet in MONAI has `layer4` as its last block.
    target_layer = "resnet_branch.layer4"
    
    try:
        cam = GradCAM(nn_module=model, target_layers=target_layer)
        result = cam(x=input_tensor, class_idx=target_class)
        heatmap = result[0, 0].cpu().numpy()
    except Exception as e:
        print(f"Warning: GradCAM failed ({e}). Proceeding without heatmap.")
        return
    
    # Take the middle slice for visualization
    mid_idx = heatmap.shape[0] // 2
    plt.figure(figsize=(10, 10))
    plt.imshow(input_tensor[0, 0, mid_idx].cpu().numpy(), cmap='gray')
    plt.imshow(heatmap[mid_idx], cmap='jet', alpha=0.5)
    plt.title(f'Grad-CAM Heatmap (Class: {target_class})')
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0
