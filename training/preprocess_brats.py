import os
import glob
import numpy as np
import nibabel as nib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json
import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Resized,
    ScaleIntensityRangePercentilesd,
    Orientationd,
    NormalizeIntensityd,
    EnsureTyped,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
)

def preprocess_data(data_dir, output_dir, target_shape=(128, 128, 128), limit=None):
    """
    Preprocess BraTS 2020 dataset: Load, Resize, Normalize, and Stack.
    """
    # Create output directories
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Find all patient directories
    # Path: BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_XXX
    patient_dirs = sorted(glob.glob(os.path.join(data_dir, 'BraTS2020_TrainingData', 'MICCAI_BraTS2020_TrainingData', 'BraTS20_Training_*')))
    if limit:
        patient_dirs = patient_dirs[:limit]
    print(f"Found {len(patient_dirs)} patients.")

    processed_data_list = []

    for patient_path in tqdm(patient_dirs, desc="Preprocessing Patients"):
        patient_id = os.path.basename(patient_path)
        
        # Define paths for modalities and segmentation
        t1_path = os.path.join(patient_path, f"{patient_id}_t1.nii")
        t1ce_path = os.path.join(patient_path, f"{patient_id}_t1ce.nii")
        t2_path = os.path.join(patient_path, f"{patient_id}_t2.nii")
        flair_path = os.path.join(patient_path, f"{patient_id}_flair.nii")
        seg_path = os.path.join(patient_path, f"{patient_id}_seg.nii")

        if not all(os.path.exists(p) for p in [t1_path, t1ce_path, t2_path, flair_path, seg_path]):
            print(f"Skipping {patient_id}: Missing files.")
            continue

        try:
            # Load images using MONAI pipeline for robustness
            loader = Compose([
                LoadImaged(keys=["t1", "t1ce", "t2", "flair", "seg"]),
                EnsureChannelFirstd(keys=["t1", "t1ce", "t2", "flair", "seg"]),
                Orientationd(keys=["t1", "t1ce", "t2", "flair", "seg"], axcodes="RAS"),
                Resized(keys=["t1", "t1ce", "t2", "flair"], spatial_size=target_shape, mode="trilinear"),
                Resized(keys=["seg"], spatial_size=target_shape, mode="nearest"),
                # Normalize each modality independently
                NormalizeIntensityd(keys=["t1", "t1ce", "t2", "flair"], nonzero=True, channel_wise=True),
            ])

            data_dict = loader({
                "t1": t1_path,
                "t1ce": t1ce_path,
                "t2": t2_path,
                "flair": flair_path,
                "seg": seg_path
            })

            # Stack modalities: (4, 128, 128, 128)
            stacked_img = np.stack([
                data_dict["t1"][0],
                data_dict["t1ce"][0],
                data_dict["t2"][0],
                data_dict["flair"][0]
            ], axis=0).astype(np.float32)

            # Segmentation label: (128, 128, 128)
            # Label mapping: BraTS labels (0, 1, 2, 4) -> (0, 1, 2, 3) for nnU-Net compatibility
            seg_img = data_dict["seg"][0].astype(np.uint8)
            seg_img[seg_img == 4] = 3

            # Save processed files
            img_save_path = os.path.join(images_dir, f"{patient_id}.npy")
            seg_save_path = os.path.join(labels_dir, f"{patient_id}_seg.npy")

            np.save(img_save_path, stacked_img)
            np.save(seg_save_path, seg_img)

            processed_data_list.append({
                "patient_id": patient_id,
                "image": img_save_path,
                "label": seg_save_path
            })

        except Exception as e:
            print(f"Error processing {patient_id}: {e}")

    # Split dataset: 70/15/15
    train_data, temp_data = train_test_split(processed_data_list, test_size=0.30, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.50, random_state=42)

    split_info = {
        "train": train_data,
        "val": val_data,
        "test": test_data
    }

    with open(os.path.join(output_dir, 'dataset_split.json'), 'w') as f:
        json.dump(split_info, f, indent=4)

    print(f"Preprocessing complete. Split saved to {output_dir}/dataset_split.json")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

if __name__ == "__main__":
    DATA_ROOT = r"d:\BrainTumor_HMGTNet_Project\data\brats20-dataset-training-validation"
    OUTPUT_ROOT = r"d:\BrainTumor_HMGTNet_Project\data\processed"
    
    preprocess_data(DATA_ROOT, OUTPUT_ROOT)
