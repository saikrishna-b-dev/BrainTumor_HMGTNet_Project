import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from monai.visualize import GradCAM
import random

def generate_gradcam_heatmaps(model, dataloader, device, output_dir, classes, num_samples=20):
    """
    Generate Grad-CAM heatmaps for a set of random test samples.
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    # Target the last Conv3d layer in the CNN branch (index 6) for spatial feature maps.
    # cnn_branch layout: [Conv3d, BN, ReLU, Conv3d, BN, ReLU, Conv3d(6), BN(7), ReLU(8), Pool(9)]
    # Using string path so MONAI can correctly register forward hooks.
    target_layer = "cnn_branch.6"
    cam = GradCAM(nn_module=model, target_layers=target_layer)
    
    samples_processed = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        # Reduce input size to save memory and match model
        if images.shape[-3:] != (64, 64, 64):
            images = F.interpolate(images, size=(64, 64, 64), mode='trilinear', align_corners=False)
        
        for i in range(images.size(0)):
            if samples_processed >= num_samples:
                break
                
            input_tensor = images[i:i+1]
            target_class = labels[i].item()
            
            try:
                # Grad-CAM requires gradients, so do NOT use torch.no_grad()
                input_tensor.requires_grad_(False)
                result = cam(x=input_tensor, class_idx=target_class)
                
                heatmap = result[0, 0].cpu().detach().numpy()
                img_np = input_tensor[0, 0].cpu().detach().numpy()
                
                # Middle slice index
                mid_idx = heatmap.shape[0] // 2
                
                # Plot
                fig, axes = plt.subplots(1, 2, figsize=(15, 7), dpi=300)
                
                # Original Slice
                axes[0].imshow(img_np[mid_idx], cmap='gray')
                axes[0].set_title(f"Original MRI (Slice {mid_idx})", fontsize=14)
                axes[0].axis('off')
                
                # Overlay
                axes[1].imshow(img_np[mid_idx], cmap='gray')
                axes[1].imshow(heatmap[mid_idx], cmap='jet', alpha=0.5)
                axes[1].set_title(f"Grad-CAM Heatmap (True: {classes[target_class]})", fontsize=14)
                axes[1].axis('off')
                
                plt.tight_layout()
                save_name = f"sample_{samples_processed}_class_{classes[target_class]}.png"
                plt.savefig(os.path.join(output_dir, save_name))
                plt.close()
                
                samples_processed += 1
                print(f"  Grad-CAM [{samples_processed}/{num_samples}] saved.")
                
            except Exception as e:
                print(f"  Warning: Grad-CAM failed for sample {i}: {e}")
                plt.close('all')
                continue
            
        if samples_processed >= num_samples:
            break
            
    print(f"Generated {samples_processed} Grad-CAM heatmaps in {output_dir}")
