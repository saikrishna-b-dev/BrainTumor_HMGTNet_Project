import os
import torch

class Config:
    # Paths
    PROJECT_ROOT = r"d:\BrainTumor_HMGTNet_Project"
    DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "processed")
    SPLIT_JSON = os.path.join(DATA_ROOT, "dataset_split.json")
    OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs")
    CHECKPOINT_DIR = os.path.join(OUTPUT_ROOT, "checkpoints")
    LOG_DIR = os.path.join(OUTPUT_ROOT, "logs")
    HEATMAP_DIR = os.path.join(OUTPUT_ROOT, "heatmaps")

    # Create directories
    for d in [CHECKPOINT_DIR, LOG_DIR, HEATMAP_DIR]:
        os.makedirs(d, exist_ok=True)

    # Model Hyperparameters
    IN_CHANNELS = 4
    NUM_CLASSES = 4
    IMG_SIZE = (64, 64, 64)
    
    # Branch Configs
    PATCH_SIZE = 16
    EMBED_DIM = 256
    NUM_HEADS = 8
    DEPTH = 4
    
    # Training Hyperparameters
    BATCH_SIZE = 2  # Maximize GPU memory while staying safe
    EPOCHS = 100
    LR = 3e-4
    WEIGHT_DECAY = 1e-4
    
    # Optimizer & Scheduler
    OPTIMIZER = "AdamW"
    SCHEDULER = "CosineAnnealingWarmRestarts"
    T_0 = 10
    T_MULT = 2
    EARLY_STOPPING_PATIENCE = 10
    
    # Advanced
    USE_AMP = True  # Automatic Mixed Precision
    GRAD_CLIP = 1.0
    LABEL_SMOOTHING = 0.1
    
    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
