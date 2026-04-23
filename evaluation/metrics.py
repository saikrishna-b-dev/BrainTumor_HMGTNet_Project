import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, roc_auc_score, cohen_kappa_score
)

def compute_all_metrics(y_true, y_pred, y_prob, classes):
    """
    Compute comprehensive classification metrics.
    """
    # 1. Basic Metrics
    acc = accuracy_score(y_true, y_pred)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # 2. ROC-AUC (One-vs-Rest)
    try:
        roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
    except Exception as e:
        print(f"Warning: Could not compute ROC-AUC: {e}")
        roc_auc = 0.0
        
    # 3. Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # 4. Per-class metrics
    prec_pc, rec_pc, f1_pc, support_pc = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    per_class_metrics = []
    for i, class_name in enumerate(classes):
        per_class_metrics.append({
            "Class": class_name,
            "Precision": prec_pc[i],
            "Recall": rec_pc[i],
            "F1-score": f1_pc[i],
            "Support": int(support_pc[i])
        })
    
    per_class_df = pd.DataFrame(per_class_metrics)
    
    # 5. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    summary = {
        "Accuracy": acc,
        "Precision (Weighted)": prec_weighted,
        "Recall (Weighted)": rec_weighted,
        "F1-score (Weighted)": f1_weighted,
        "Precision (Macro)": prec_macro,
        "Recall (Macro)": rec_macro,
        "F1-score (Macro)": f1_macro,
        "ROC-AUC (Weighted)": roc_auc,
        "Cohen's Kappa": kappa
    }
    
    return summary, per_class_df, cm
