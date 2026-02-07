"""
Utility functions for image preprocessing and conversion.
"""

import torch
import numpy as np
from PIL import Image
from typing import Union, List, Tuple, Optional


# Model-specific preprocessing configurations
MODEL_PREPROCESS_CONFIGS = {
    'densenet121-res224-all': {
        'target_size': 224,
        'output_range': (-1024, 1024),
        'channels': 1,
    },
    'densenet121-res224-nih': {
        'target_size': 224,
        'output_range': (-1024, 1024),
        'channels': 1,
    },
    'densenet121-res224-pc': {
        'target_size': 224,
        'output_range': (-1024, 1024),
        'channels': 1,
    },
    'densenet121-res224-chex': {
        'target_size': 224,
        'output_range': (-1024, 1024),
        'channels': 1,
    },
    'densenet121-res224-mimic_nb': {
        'target_size': 224,
        'output_range': (-1024, 1024),
        'channels': 1,
    },
    'densenet121-res224-mimic_ch': {
        'target_size': 224,
        'output_range': (-1024, 1024),
        'channels': 1,
    },
    'resnet50-res512-all': {
        'target_size': 512,
        'output_range': (-1024, 1024),
        'channels': 1,
    },
    'jfhealthcare': {
        'target_size': 512,
        'output_range': (-2, 2),  # Direct - we bypass the wrapper's /512 division
        'channels': 3,
    },
}


def _to_grayscale_tensor(
    image: Union[Image.Image, torch.Tensor, np.ndarray],
    input_range: Tuple[float, float] = (-1, 1),
) -> torch.Tensor:
    """
    Convert image to grayscale tensor in [0, 1] range.
    
    Args:
        image: Input image (PIL, tensor, or numpy)
        input_range: Expected input range for tensors (default: (-1, 1) from VAE)
        
    Returns:
        Grayscale tensor of shape (1, H, W) in [0, 1] range
    """
    if isinstance(image, Image.Image):
        # Convert PIL to numpy
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)
        
        if len(image_tensor.shape) == 2:
            image_tensor = image_tensor.unsqueeze(0)  # (H, W) -> (1, H, W)
        else:
            image_tensor = image_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            
    elif isinstance(image, np.ndarray):
        image_tensor = torch.from_numpy(image.astype(np.float32))
        # Normalize to [0, 1]
        if image.max() > 1.0:
            image_tensor = image_tensor / 255.0
            
        if len(image_tensor.shape) == 2:
            image_tensor = image_tensor.unsqueeze(0)
        elif len(image_tensor.shape) == 3 and image_tensor.shape[2] in [1, 3, 4]:
            image_tensor = image_tensor.permute(2, 0, 1)
            
    elif isinstance(image, torch.Tensor):
        image_tensor = image.clone().float()
        
        # Convert from input_range to [0, 1]
        in_min, in_max = input_range
        if image_tensor.min() < 0 or in_min < 0:
            # Input is in [-1, 1] or similar signed range
            image_tensor = (image_tensor - in_min) / (in_max - in_min)
        elif image_tensor.max() > 1.0:
            # Input is in [0, 255] range
            image_tensor = image_tensor / 255.0
        # else: already in [0, 1]
        
        # Handle shape
        if len(image_tensor.shape) == 2:
            image_tensor = image_tensor.unsqueeze(0)
        elif len(image_tensor.shape) == 4:
            # Batch dimension - remove it for now, will be added back later
            image_tensor = image_tensor.squeeze(0)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    # Ensure (C, H, W) format
    if len(image_tensor.shape) == 2:
        image_tensor = image_tensor.unsqueeze(0)
    
    # Convert RGB to grayscale if needed
    if image_tensor.shape[0] == 3:
        image_tensor = (
            0.299 * image_tensor[0:1] + 
            0.587 * image_tensor[1:2] + 
            0.114 * image_tensor[2:3]
        )
    elif image_tensor.shape[0] == 4:
        # RGBA - use RGB channels
        image_tensor = (
            0.299 * image_tensor[0:1] + 
            0.587 * image_tensor[1:2] + 
            0.114 * image_tensor[2:3]
        )
    
    return image_tensor.clamp(0, 1)


def preprocess_for_model(
    image: Union[Image.Image, torch.Tensor, np.ndarray],
    model_name: str,
    input_range: Tuple[float, float] = (-1, 1),
) -> torch.Tensor:
    """
    Model-specific preprocessing. Call ONCE per model, not shared.
    
    Each model has its own target size, output range, and channel count.
    This avoids inefficient double normalization/resize.
    
    Args:
        image: Input image (PIL, tensor, or numpy)
        model_name: Name of the model (key in MODEL_PREPROCESS_CONFIGS)
        input_range: Expected input range for tensors (default: (-1, 1) from VAE)
        
    Returns:
        Preprocessed tensor ready for the specific model
    """
    if model_name not in MODEL_PREPROCESS_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_PREPROCESS_CONFIGS.keys())}")
    
    config = MODEL_PREPROCESS_CONFIGS[model_name]
    target_size = config['target_size']
    out_min, out_max = config['output_range']
    out_channels = config['channels']
    
    # Step 1: Convert to grayscale tensor in [0, 1] range
    tensor = _to_grayscale_tensor(image, input_range)  # (1, H, W) in [0, 1]
    
    # Step 2: Resize to model's native resolution
    if tensor.shape[1] != target_size or tensor.shape[2] != target_size:
        tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0),
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
    
    # Step 3: Normalize to output range
    # [0, 1] -> [out_min, out_max]
    out_scale = out_max - out_min
    tensor = tensor * out_scale + out_min
    
    # Step 4: Expand to required channels
    if out_channels == 3 and tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)  # (1, H, W) -> (3, H, W)
    
    return tensor


def batch_images_for_model(
    images: List[Union[Image.Image, torch.Tensor, np.ndarray]],
    model_name: str,
    device: str = 'cuda',
    input_range: Tuple[float, float] = (-1, 1),
) -> torch.Tensor:
    """
    Batch process multiple images for a specific model.
    
    Args:
        images: List of images
        model_name: Name of the model
        device: Device to place tensors on
        input_range: Expected input range for tensors
        
    Returns:
        Batched tensor with shape (B, C, H, W)
    """
    processed = []
    for img in images:
        img_tensor = preprocess_for_model(img, model_name, input_range)
        processed.append(img_tensor)
    
    batch = torch.stack(processed, dim=0)
    return batch.to(device)


def preprocess_for_xrv(
    image: Union[Image.Image, torch.Tensor, np.ndarray],
    target_size: int = 224,
) -> torch.Tensor:
    """
    Preprocess image for TorchXRayVision models.
    
    TorchXRayVision expects:
    - Single channel (grayscale)
    - Normalized to [-1024, 1024] range using xrv.datasets.normalize formula
    - Shape: (1, H, W) or (B, 1, H, W)
    
    The normalization formula is: (img / maxval) * 2048 - 1024
    which maps [0, maxval] → [-1024, 1024]
    
    Args:
        image: Input image (PIL, tensor, or numpy)
        target_size: Target image size (height and width)
        
    Returns:
        Preprocessed tensor ready for XRV models in [-1024, 1024] range
    """
    # Step 1: Convert to tensor in [0, 255] range
    if isinstance(image, Image.Image):
        image_np = np.array(image).astype(np.float32)
        # Keep in [0, 255] range for now
        image_tensor = torch.from_numpy(image_np)
        
        # Add channel dim if needed
        if len(image_tensor.shape) == 2:
            image_tensor = image_tensor.unsqueeze(0)  # (H, W) -> (1, H, W)
        else:
            # (H, W, C) -> (C, H, W)
            image_tensor = image_tensor.permute(2, 0, 1)
            
    elif isinstance(image, np.ndarray):
        image_tensor = torch.from_numpy(image.astype(np.float32))
        # If already normalized to [0, 1], scale back to [0, 255]
        if image.max() <= 1.0:
            image_tensor = image_tensor * 255.0
            
    elif isinstance(image, torch.Tensor):
        image_tensor = image.clone()
        # Ensure float
        if image_tensor.dtype != torch.float32:
            image_tensor = image_tensor.float()
        
        # Handle different input ranges
        if image_tensor.min() < 0:
            # Input is in [-1, 1] range (e.g., from VAE output)
            # Convert to [0, 255]
            image_tensor = ((image_tensor + 1) / 2) * 255.0
        elif image_tensor.max() <= 1.0:
            # Input is in [0, 1] range
            image_tensor = image_tensor * 255.0
        # else: already in [0, 255] range
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    # Step 2: Ensure we have (C, H, W) format
    if len(image_tensor.shape) == 2:
        image_tensor = image_tensor.unsqueeze(0)  # (H, W) -> (1, H, W)
    elif len(image_tensor.shape) == 4:
        # Batch dimension present, handle it separately
        pass
    
    # Step 3: Convert RGB to grayscale if needed (using luminance weights)
    if image_tensor.shape[0] == 3 or (len(image_tensor.shape) == 4 and image_tensor.shape[1] == 3):
        if len(image_tensor.shape) == 3:  # (3, H, W)
            # Standard RGB to grayscale conversion
            image_tensor = (
                0.299 * image_tensor[0:1] + 
                0.587 * image_tensor[1:2] + 
                0.114 * image_tensor[2:3]
            )
        else:  # (B, 3, H, W)
            image_tensor = (
                0.299 * image_tensor[:, 0:1] + 
                0.587 * image_tensor[:, 1:2] + 
                0.114 * image_tensor[:, 2:3]
            )
    
    # Step 4: Resize if needed
    if len(image_tensor.shape) == 3:  # (1, H, W)
        if image_tensor.shape[1] != target_size or image_tensor.shape[2] != target_size:
            image_tensor = torch.nn.functional.interpolate(
                image_tensor.unsqueeze(0),
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
    else:  # (B, 1, H, W)
        if image_tensor.shape[2] != target_size or image_tensor.shape[3] != target_size:
            image_tensor = torch.nn.functional.interpolate(
                image_tensor,
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=False
            )
    
    # Step 5: Apply TorchXRayVision normalization
    # Formula: (img / 255) * 2048 - 1024
    # This maps [0, 255] → [-1024, 1024]
    image_tensor = (image_tensor / 255.0) * 2048.0 - 1024.0
    
    return image_tensor


def batch_images(
    images: List[Union[Image.Image, torch.Tensor, np.ndarray]],
    target_size: int = 224,
    device: str = 'cuda',
) -> torch.Tensor:
    """
    Batch process multiple images for XRV.
    
    Args:
        images: List of images
        target_size: Target size for resizing
        device: Device to place tensors on
        
    Returns:
        Batched tensor of shape (B, 1, H, W)
    """
    processed = []
    for img in images:
        img_tensor = preprocess_for_xrv(img, target_size)
        # Ensure (1, H, W)
        if len(img_tensor.shape) == 2:
            img_tensor = img_tensor.unsqueeze(0)
        processed.append(img_tensor)
    
    # Stack into batch
    batch = torch.stack(processed, dim=0)
    return batch.to(device)


def compute_statistics(values: List[float]) -> dict:
    """
    Compute summary statistics for a list of values.
    
    Args:
        values: List of numeric values
        
    Returns:
        Dictionary with mean, std, min, max, median
    """
    if not values:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'median': 0.0,
            'count': 0,
        }
    
    values_tensor = torch.tensor(values, dtype=torch.float32)
    
    return {
        'mean': values_tensor.mean().item(),
        'std': values_tensor.std().item(),
        'min': values_tensor.min().item(),
        'max': values_tensor.max().item(),
        'median': values_tensor.median().item(),
        'count': len(values),
    }


def create_comparison_grid(
    original: Image.Image,
    counterfactual: Image.Image,
    direction: str,
    labels: bool = True,
    padding: int = 10,
    label_height: int = 30,
) -> Image.Image:
    """
    Create a side-by-side comparison grid of original and counterfactual images.
    
    Args:
        original: Original image (PIL)
        counterfactual: Counterfactual image (PIL)
        direction: Edit direction ('increase' or 'decrease')
        labels: Whether to add text labels above images
        padding: Padding between images in pixels
        label_height: Height reserved for labels if enabled
        
    Returns:
        Combined grid image (PIL)
    """
    # Ensure both images are the same size
    if original.size != counterfactual.size:
        # Resize counterfactual to match original
        counterfactual = counterfactual.resize(original.size, Image.Resampling.LANCZOS)
    
    width, height = original.size
    
    # Calculate grid dimensions
    header = label_height if labels else 0
    grid_width = width * 2 + padding
    grid_height = height + header
    
    # Create grid canvas (white background)
    grid = Image.new('RGB', (grid_width, grid_height), color=(255, 255, 255))
    
    # Convert images to RGB if needed
    if original.mode != 'RGB':
        original = original.convert('RGB')
    if counterfactual.mode != 'RGB':
        counterfactual = counterfactual.convert('RGB')
    
    # Paste images
    grid.paste(original, (0, header))
    grid.paste(counterfactual, (width + padding, header))
    
    # Add labels if requested
    if labels:
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(grid)
            
            # Try to use a system font, fall back to default
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            except (OSError, IOError):
                try:
                    font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", 16)
                except (OSError, IOError):
                    font = ImageFont.load_default()
            
            # Draw labels centered above each image
            orig_label = "Original"
            cf_label = f"Counterfactual ({direction})"
            
            # Get text bounding boxes for centering
            orig_bbox = draw.textbbox((0, 0), orig_label, font=font)
            cf_bbox = draw.textbbox((0, 0), cf_label, font=font)
            
            orig_text_width = orig_bbox[2] - orig_bbox[0]
            cf_text_width = cf_bbox[2] - cf_bbox[0]
            
            # Center labels above each image
            orig_x = (width - orig_text_width) // 2
            cf_x = width + padding + (width - cf_text_width) // 2
            text_y = (label_height - 16) // 2  # Vertically center
            
            draw.text((orig_x, text_y), orig_label, fill=(0, 0, 0), font=font)
            draw.text((cf_x, text_y), cf_label, fill=(0, 0, 0), font=font)
            
        except ImportError:
            # If ImageDraw is not available, skip labels
            pass
    
    return grid



