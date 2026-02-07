import numpy as np
import matplotlib.pyplot as plt
from . import viz

def plot_alexnet_filters(ax=None):
    """
    Plots the first layer convolutional kernels of AlexNet.
    Requires torch and torchvision.
    """
    try:
        import torch
        import torchvision.models as models
        from torchvision.utils import make_grid
    except ImportError:
        if ax:
            ax.text(0.5, 0.5, "torch/torchvision not installed\nCannot generate AlexNet filters", 
                    ha='center', va='center', fontsize=12)
            ax.axis('off')
        else:
            print("Warning: torch or torchvision not found. Cannot generate AlexNet filters.")
        return

    # Load pre-trained AlexNet
    # Note: Standard PyTorch AlexNet has 64 filters in the first layer.
    # The original paper had 96.
    try:
        # Suppress download progress
        import logging
        logging.getLogger('torchvision').setLevel(logging.WARNING)
        alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    except:
        # Fallback for older torchvision or if weights enum not available
        alexnet = models.alexnet(pretrained=True)
    
    # Get weights: (64, 3, 11, 11)
    filters = alexnet.features[0].weight.data.clone()
    
    # Normalize to [0, 1] for visualization
    min_val = filters.min()
    max_val = filters.max()
    filters = (filters - min_val) / (max_val - min_val)
    
    # Make grid
    # nrow=8 -> 8 columns, 8 rows (for 64 filters)
    grid = make_grid(filters, nrow=8, padding=1, pad_value=1)
    
    # (C, H, W) -> (H, W, C)
    grid_np = grid.numpy().transpose((1, 2, 0))
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        
    ax.imshow(grid_np)
    ax.axis('off')
    # ax.set_title("AlexNet First Layer Kernels", fontsize=12)

def plot_sparsity_heatmap(ax=None, data=None):
    """
    Plots a sparsity heatmap similar to the Numenta figure.
    If data is None, uses mock data.
    """
    # Exact labels from image
    layers = [
        "stage0",
        "stage1.resblk1", "stage1.resblk2", "stage1.resblk3",
        "stage2.resblk1", "stage2.resblk2", "stage2.resblk3", "stage2.resblk4",
        "stage3.resblk1", "stage3.resblk2", "stage3.resblk3", "stage3.resblk4", "stage3.resblk5", "stage3.resblk6",
        "stage4.resblk1", "stage4.resblk2", "stage4.resblk3",
        "final_linear", "fullmodel"
    ]
    
    sparsities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    if data is None:
        # Mock data: Density (1 - Sparsity)
        # We'll generate random data biased towards the visual
        np.random.seed(42)
        # Base: 0.7 (Light)
        data = np.ones((len(sparsities), len(layers))) * 0.7
        
        # Add some "sparsity" (darker values)
        # Randomly subtract values, more in higher sparsity rows?
        # Or just random noise
        noise = np.random.rand(len(sparsities), len(layers)) * 0.6
        data = data - noise * 0.5
        
        # Ensure range
        data = np.clip(data, 0.15, 0.75)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        
    # Plot heatmap
    # cmap='magma' looks similar to the image (purple to orange/beige)
    im = ax.imshow(data, aspect='auto', cmap='magma', origin='lower', vmin=0.15, vmax=0.75)
    
    # Ticks
    ax.set_xticks(np.arange(len(layers)))
    ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=8)
    
    ax.set_yticks(np.arange(len(sparsities)))
    ax.set_yticklabels([f"{s:.1f}" for s in sparsities])
    
    ax.set_ylabel("Sparsity")
    
    # Remove grid for heatmap
    ax.grid(False)
    
    # Add colorbar if we created the figure
    if ax.figure and len(ax.figure.axes) == 1:
        cbar = ax.figure.colorbar(im, ax=ax)
        # cbar.set_label("Density")
