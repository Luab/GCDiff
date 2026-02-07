"""
Visual test for data loader - saves sample images and prints info.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from data_loader import CheXpertGraphDataset

def visualize_samples(num_samples=4):
    """Load and visualize sample images with their metadata."""
    
    # Load dataset
    dataset = CheXpertGraphDataset(
        split="train",
        max_samples=num_samples,
    )
    
    # Create figure
    fig, axes = plt.subplots(1, num_samples, figsize=(4*num_samples, 5))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        sample = dataset[i]
        
        # Denormalize image from [-1, 1] to [0, 1] for display
        img = sample['image']
        img = (img + 1) / 2
        img = img.permute(1, 2, 0).numpy()  # [H, W, C]
        
        # Display
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        
        # Title with metadata
        title = f"Patient: {sample['patient_id']}\n"
        title += f"Prompt: {sample['text_prompt'][:50]}..."
        axes[i].set_title(title, fontsize=8)
        
        # Print detailed info
        print(f"\n{'='*80}")
        print(f"Sample {i+1}:")
        print(f"  Patient ID: {sample['patient_id']}")
        print(f"  Image path: {sample['image_path']}")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Graph embedding shape: {sample['graph_embedding'].shape}")
        print(f"  Graph embedding stats:")
        print(f"    - Mean: {sample['graph_embedding'].mean():.4f}")
        print(f"    - Std: {sample['graph_embedding'].std():.4f}")
        print(f"    - Min: {sample['graph_embedding'].min():.4f}")
        print(f"    - Max: {sample['graph_embedding'].max():.4f}")
        print(f"  Text prompt: {sample['text_prompt'][:200]}")
    
    plt.tight_layout()
    plt.savefig('dataloader_samples.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to dataloader_samples.png")

if __name__ == "__main__":
    visualize_samples(num_samples=4)

