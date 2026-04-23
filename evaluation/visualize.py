import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from itertools import cycle

def plot_confusion_matrix(cm, classes, save_path):
    """
    Plot and save confusion matrix heatmap.
    """
    plt.figure(figsize=(10, 8), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 14})
    plt.title('Confusion Matrix', fontsize=18)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curves(y_true, y_prob, classes, save_path):
    """
    Plot One-vs-Rest ROC curves for each class.
    """
    n_classes = len(classes)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Binarize labels for per-class ROC
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    plt.figure(figsize=(10, 8), dpi=300)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
    
    for i, color in zip(range(n_classes), colors):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of {classes[i]} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=18)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_precision_recall_curves(y_true, y_prob, classes, save_path):
    """
    Plot Precision-Recall curves for each class.
    """
    n_classes = len(classes)
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    plt.figure(figsize=(10, 8), dpi=300)
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    
    for i, color in zip(range(n_classes), colors):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        avg_prec = average_precision_score(y_true_bin[:, i], y_prob[:, i])
        plt.plot(recall, precision, color=color, lw=2,
                 label=f'PR curve of {classes[i]} (AP = {avg_prec:0.2f})')

    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curves', fontsize=18)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_training_curves(log_csv_path, save_dir):
    """
    Plot loss and accuracy curves from training log.
    """
    try:
        df = pd.read_csv(log_csv_path)
        if df.empty:
            print("Warning: Training log is empty. Skipping training curves.")
            return
            
        # Loss Plot
        plt.figure(figsize=(10, 6), dpi=300)
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss', lw=2)
        plt.plot(df['epoch'], df['val_loss'], label='Val Loss', lw=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss', fontsize=16)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{save_dir}/loss_curve.png")
        plt.close()
        
        # Accuracy Plot
        plt.figure(figsize=(10, 6), dpi=300)
        plt.plot(df['epoch'], df['val_acc'], label='Val Accuracy', color='green', lw=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Validation Accuracy', fontsize=16)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{save_dir}/accuracy_curve.png")
        plt.close()
        
    except Exception as e:
        print(f"Error plotting training curves: {e}")
