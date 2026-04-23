# HMGT-Net: Brain Tumor Detection using Hybrid Multi-Scale Graph Transformer

## Overview
HMGT-Net (Hybrid Multi-Scale Graph Transformer Network) is a state-of-the-art framework for robust and highly accurate 3D brain tumor classification using multi-parametric Magnetic Resonance Imaging (MRI). 

By integrating 3D CNNs (ResNet50), Vision Transformers (ViT), Swin Transformers, and Relational Graph Learning, the architecture effectively models local volumetric textures, global context, and hierarchical semantic relationships to distinguish complex tumor subtypes.

## Architecture Description
The proposed model leverages a multi-branch fusion approach:
1. **Local Feature Extraction (3D ResNet50)**: Deep bottleneck layers extract fine-grained spatial and textual features.
2. **Global Context Encoding (3D ViT)**: Models long-range dependencies across the entire 3D volume.
3. **Hierarchical Modeling (SwinUNETR)**: Employs shifted window attention to capture localized dependencies without sacrificing global context.
4. **Attention-Based Fusion**: A `MultiheadAttention` module intelligently weighs and fuses the deep representations from all three branches.
5. **Graph Relational Learning**: Treats fused features as nodes within a fully connected graph, optimized via Graph Attention Networks (GATv2) to solidify inter-class feature boundaries before the final classification head.

## Dataset (BRaTS)
This project is configured to operate on the Brain Tumor Segmentation (BraTS) challenge datasets, supporting T1, T1CE, T2, and FLAIR modalities.

## Installation
Ensure you have Python 3.8+ installed.

```bash
git clone https://github.com/<your-username>/BrainTumor_HMGTNet_Project.git
cd BrainTumor_HMGTNet_Project
pip install -r requirements.txt
```

## Folder Structure
```
BrainTumor_HMGTNet_Project/
│
├── configs/             # Configuration files (hyperparameters, paths)
├── data/
│   ├── raw/             # Raw NIfTI datasets (ignored by git)
│   └── processed/       # Preprocessed .npy volumes (ignored by git)
│
├── evaluation/          # Evaluation scripts (metrics, TTA, ensemble)
├── models/              # HMGT-Net architecture definition
├── outputs/
│   ├── checkpoints/     # Saved model weights
│   ├── logs/            # Training logs and confusion matrices
│   └── results/         # Final prediction CSVs and visualizations
│
├── training/            # Training pipeline, dataloaders, loss functions
└── utils/               # Utilities (Grad-CAM, Early Stopping)
```

## Training Instructions
1. Place your raw NIfTI files into `data/raw/` and run the preprocessing pipeline (e.g., `preprocess_brats.py`) to generate `dataset_split.json`.
2. Configure training hyperparameters inside `configs/config.py` (ensure `BATCH_SIZE` suits your VRAM constraints).
3. Execute the training loop:
```bash
python -m training.train
```

## Evaluation Instructions
To evaluate the top-3 ensembled checkpoints with Test-Time Augmentation (TTA), run:
```bash
python -m evaluation.evaluate
```
This will automatically generate comprehensive metrics, ROC/PR curves, and Grad-CAM spatial heatmaps in the `outputs/` directories.

## Results
*(Placeholder: To be updated with final experimental results)*
* **Accuracy**: > 99.35%
* **Macro F1-Score**: > 0.98

## Future Work
* **Federated Learning**: Extending the pipeline to train safely across multi-institutional boundaries without centralizing data.
* **Knowledge Distillation**: Compressing the heavy HMGT-Net ensemble into a lightweight 3D MobileNet for rapid edge inference.
