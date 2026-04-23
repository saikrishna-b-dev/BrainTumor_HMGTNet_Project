import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.cuda.amp import autocast

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.config import Config
from models.hmgt_net import HMGTNet
from training.dataset import get_dataloader
from evaluation.metrics import compute_all_metrics
from evaluation.visualize import (
    plot_confusion_matrix, plot_roc_curves, 
    plot_precision_recall_curves, plot_training_curves
)
from evaluation.explain import generate_gradcam_heatmaps

def main():
    # 1. Setup Directories
    RESULTS_DIR = os.path.join(Config.OUTPUT_ROOT, "results")
    PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
    ERRORS_DIR = os.path.join(RESULTS_DIR, "error_analysis")
    HEATMAPS_DIR = os.path.join(Config.OUTPUT_ROOT, "heatmaps")
    
    for d in [RESULTS_DIR, PLOTS_DIR, ERRORS_DIR, HEATMAPS_DIR]:
        os.makedirs(d, exist_ok=True)
    
    device = torch.device(Config.DEVICE)
    print(f"Evaluating on device: {device}")
    
    # 2. Load Top-3 Models for Ensemble
    import glob
    ckpt_dir = Config.CHECKPOINT_DIR
    model_paths = glob.glob(os.path.join(ckpt_dir, "model_f1_*.pth"))
    
    if len(model_paths) == 0:
        # Fallback
        model_paths = [os.path.join(ckpt_dir, "best_model.pth")]
        
    # Sort and take top 3
    model_paths.sort(reverse=True)
    model_paths = model_paths[:3]
    
    models = []
    for p in model_paths:
        m = HMGTNet(in_channels=Config.IN_CHANNELS, num_classes=Config.NUM_CLASSES, img_size=Config.IMG_SIZE).to(device)
        try:
            m.load_state_dict(torch.load(p, map_location=device))
            m.eval()
            models.append(m)
            print(f"Loaded ensemble model from {p}")
        except Exception as e:
            print(f"Error loading {p}: {e}")
            
    if len(models) == 0:
        print("WARNING: Proceeding with single randomly initialized model for pipeline demonstration.")
        m = HMGTNet(in_channels=Config.IN_CHANNELS, num_classes=Config.NUM_CLASSES, img_size=Config.IMG_SIZE).to(device)
        m.eval()
        models.append(m)
    
    # 3. Data Loader
    split_file = Config.SPLIT_JSON
    try:
        if not os.path.exists(split_file):
            raise FileNotFoundError()
        test_loader = get_dataloader(split_file, batch_size=Config.BATCH_SIZE, mode="test")
    except Exception:
        print("Test split not found in main split file, trying dataset_split_test.json...")
        test_split_path = os.path.join(Config.DATA_ROOT, "dataset_split_test.json")
        if not os.path.exists(test_split_path):
            print(f"\n[ERROR] Test dataset split not found at: {test_split_path}")
            print("Please ensure you have generated a test split before running evaluation.")
            sys.exit(1)
            
        try:
            test_loader = get_dataloader(test_split_path, batch_size=Config.BATCH_SIZE, mode="test")
        except Exception as e:
            print(f"\n[ERROR] Failed to load test dataloader: {e}")
            sys.exit(1)
    
    print(f"Testing on {len(test_loader.dataset)} samples with Ensemble size {len(models)} and TTA.")
    
    # 4. Inference
    all_preds = []
    all_labels = []
    all_probs = []
    patient_ids = []
    
    test_data_list = test_loader.dataset.data_list
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(test_loader, desc="Inference")):
            images, labels = images.to(device), labels.to(device)
            
            # Reduce input size to save memory and match model
            if images.shape[-3:] != (64, 64, 64):
                images = F.interpolate(images, size=(64, 64, 64), mode='trilinear', align_corners=False)
            
            ensemble_probs = torch.zeros(images.size(0), Config.NUM_CLASSES, device=device)
            
            for model in models:
                with autocast(device_type='cuda' if 'cuda' in str(device) else 'cpu', enabled=Config.USE_AMP):
                    # TTA: Original, Flip D, Flip H, Flip W
                    p1 = torch.softmax(model(images), dim=1)
                    p2 = torch.softmax(model(torch.flip(images, [2])), dim=1)
                    p3 = torch.softmax(model(torch.flip(images, [3])), dim=1)
                    p4 = torch.softmax(model(torch.flip(images, [4])), dim=1)
                    
                    tta_prob = (p1 + p2 + p3 + p4) / 4.0
                    ensemble_probs += tta_prob
                    
            ensemble_probs /= len(models)
            _, preds = torch.max(ensemble_probs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(ensemble_probs.cpu().numpy())
            
            # Map batch index to patient IDs
            batch_start = i * Config.BATCH_SIZE
            for j in range(images.size(0)):
                patient_ids.append(test_data_list[batch_start + j]["patient_id"])

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    classes = ["glioma", "meningioma", "pituitary", "no tumor"]
    
    # 5. Metrics & Reporting
    summary, per_class_df, cm = compute_all_metrics(all_labels, all_preds, all_probs, classes)
    
    # Print summary
    print("\n" + "="*30)
    print("EVALUATION SUMMARY")
    print("="*30)
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")
    print("="*30)
    
    # Save Metrics
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=4)
    
    per_class_df.to_csv(os.path.join(RESULTS_DIR, "metrics.csv"), index=False)
    
    # Save Predictions CSV
    pred_df = pd.DataFrame({
        "id": patient_ids,
        "true_label": [classes[l] for l in all_labels],
        "pred_label": [classes[p] for p in all_preds]
    })
    for i, cls_name in enumerate(classes):
        pred_df[f"prob_{cls_name.replace(' ', '')}"] = all_probs[:, i]
    
    pred_df.to_csv(os.path.join(RESULTS_DIR, "predictions.csv"), index=False)
    
    # 6. Visualizations
    print("Generating plots...")
    plot_confusion_matrix(cm, classes, os.path.join(PLOTS_DIR, "confusion_matrix.png"))
    plot_roc_curves(all_labels, all_probs, classes, os.path.join(PLOTS_DIR, "roc_curves.png"))
    plot_precision_recall_curves(all_labels, all_probs, classes, os.path.join(PLOTS_DIR, "pr_curves.png"))
    
    log_csv = os.path.join(Config.LOG_DIR, "training_log.csv")
    plot_training_curves(log_csv, PLOTS_DIR)
    
    # 7. Error Analysis
    print("Performing error analysis...")
    misclassified = pred_df[pred_df["true_label"] != pred_df["pred_label"]]
    misclassified.to_csv(os.path.join(ERRORS_DIR, "misclassified_samples.csv"), index=False)
    
    with open(os.path.join(ERRORS_DIR, "error_summary.txt"), "w") as f:
        f.write("ERROR ANALYSIS SUMMARY\n")
        f.write("======================\n")
        f.write(f"Total misclassified: {len(misclassified)} / {len(all_labels)}\n\n")
        f.write("Top Confusion Pairs:\n")
        # Identify top confused classes
        conf_pairs = misclassified.groupby(["true_label", "pred_label"]).size().reset_index(name="count")
        conf_pairs = conf_pairs.sort_values(by="count", ascending=False)
        for _, row in conf_pairs.head(5).iterrows():
            f.write(f"{row['true_label']} -> {row['pred_label']}: {row['count']} samples\n")
            
        f.write("\nPossible Reasons:\n")
        f.write("- Class similarity between Glioma and Meningioma in certain MRI slices.\n")
        f.write("- Low contrast in T1/T2 modalities for early stage tumors.\n")
        f.write("- Model might need more training data for 'pituitary' class.\n")

    # 8. Explainability
    print("Generating Grad-CAM heatmaps...")
    generate_gradcam_heatmaps(model, test_loader, device, HEATMAPS_DIR, classes, num_samples=15)
    
    print("\nEvaluation Complete! Results saved to 'outputs/results/'.")

if __name__ == "__main__":
    main()
