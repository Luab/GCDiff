"""
Image-based metrics for perceptual and pixel-level quality.
"""

import torch
import torch.nn.functional as F
import logging
from typing import List, Dict, Union
from PIL import Image
import numpy as np

from .utils import compute_statistics

logger = logging.getLogger(__name__)


class ImageMetrics:
    """
    Compute image-based quality metrics.
    
    Metrics:
    - LPIPS: Learned Perceptual Image Patch Similarity (perceptual distance)
    - SSIM: Structural Similarity Index
    - PSNR: Peak Signal-to-Noise Ratio
    - L1: Mean absolute error
    - L2 (MSE): Mean squared error
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        lpips_net: str = 'alex',  # 'alex', 'vgg', or 'squeeze'
    ):
        """
        Initialize image metrics.
        
        Args:
            device: Device for computation
            lpips_net: Network backbone for LPIPS ('alex' is faster, 'vgg' more accurate)
        """
        self.device = device
        self.lpips_net_type = lpips_net
        
        # Lazy load LPIPS (only if needed)
        self._lpips_model = None
    
    @property
    def lpips_model(self):
        """Lazy load LPIPS model."""
        if self._lpips_model is None:
            try:
                import lpips
                logger.info(f"Loading LPIPS model with {self.lpips_net_type} backbone")
                self._lpips_model = lpips.LPIPS(net=self.lpips_net_type).to(self.device)
                self._lpips_model.eval()
            except ImportError:
                raise ImportError(
                    "lpips not installed. Install with: pip install lpips"
                )
        return self._lpips_model
    
    def _to_tensor(
        self,
        image: Union[Image.Image, torch.Tensor, np.ndarray],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Convert image to tensor in [-1, 1] range.
        
        Args:
            image: Input image
            normalize: Whether to normalize to [-1, 1]
            
        Returns:
            Tensor of shape (1, C, H, W) or (C, H, W)
        """
        if isinstance(image, Image.Image):
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)
            
            if len(image_tensor.shape) == 2:
                image_tensor = image_tensor.unsqueeze(0)  # (H, W) -> (1, H, W)
            else:
                image_tensor = image_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
                
        elif isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image.astype(np.float32))
            if image.max() > 1.0:
                image_tensor = image_tensor / 255.0
                
        elif isinstance(image, torch.Tensor):
            image_tensor = image.clone().float()
            if image_tensor.max() > 1.0:
                image_tensor = image_tensor / 255.0
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Ensure 3 channels for LPIPS (replicate grayscale if needed)
        if len(image_tensor.shape) == 2:
            image_tensor = image_tensor.unsqueeze(0)  # (H, W) -> (1, H, W)
        
        if image_tensor.shape[0] == 1:
            image_tensor = image_tensor.repeat(3, 1, 1)  # (1, H, W) -> (3, H, W)
        
        # Normalize to [-1, 1] for LPIPS
        if normalize and image_tensor.min() >= 0:
            image_tensor = image_tensor * 2 - 1
        
        return image_tensor.to(self.device)
    
    @torch.no_grad()
    def compute_lpips(
        self,
        image1: Union[Image.Image, torch.Tensor, np.ndarray],
        image2: Union[Image.Image, torch.Tensor, np.ndarray],
    ) -> float:
        """
        Compute LPIPS (Learned Perceptual Image Patch Similarity).
        Lower is better (more similar).
        
        Args:
            image1: First image
            image2: Second image
            
        Returns:
            LPIPS distance (float)
        """
        img1_tensor = self._to_tensor(image1, normalize=True)
        img2_tensor = self._to_tensor(image2, normalize=True)
        
        # Ensure batch dimension
        if len(img1_tensor.shape) == 3:
            img1_tensor = img1_tensor.unsqueeze(0)
        if len(img2_tensor.shape) == 3:
            img2_tensor = img2_tensor.unsqueeze(0)
        
        # Compute LPIPS
        lpips_value = self.lpips_model(img1_tensor, img2_tensor)
        
        return lpips_value.item()
    
    @torch.no_grad()
    def compute_ssim(
        self,
        image1: Union[Image.Image, torch.Tensor, np.ndarray],
        image2: Union[Image.Image, torch.Tensor, np.ndarray],
        window_size: int = 11,
    ) -> float:
        """
        Compute SSIM (Structural Similarity Index).
        Higher is better (more similar), range [-1, 1], typically [0, 1].
        
        Args:
            image1: First image
            image2: Second image
            window_size: Size of Gaussian window
            
        Returns:
            SSIM value (float)
        """
        img1_tensor = self._to_tensor(image1, normalize=False)
        img2_tensor = self._to_tensor(image2, normalize=False)
        
        # Ensure [0, 1] range
        if img1_tensor.min() < 0:
            img1_tensor = (img1_tensor + 1) / 2
        if img2_tensor.min() < 0:
            img2_tensor = (img2_tensor + 1) / 2
        
        # Ensure batch dimension
        if len(img1_tensor.shape) == 3:
            img1_tensor = img1_tensor.unsqueeze(0)
        if len(img2_tensor.shape) == 3:
            img2_tensor = img2_tensor.unsqueeze(0)
        
        # Convert to grayscale for SSIM (average channels)
        if img1_tensor.shape[1] == 3:
            img1_tensor = img1_tensor.mean(dim=1, keepdim=True)
        if img2_tensor.shape[1] == 3:
            img2_tensor = img2_tensor.mean(dim=1, keepdim=True)
        
        # Compute SSIM using pytorch implementation
        ssim_value = self._ssim(img1_tensor, img2_tensor, window_size=window_size)
        
        return ssim_value.item()
    
    def _ssim(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        window_size: int = 11,
        size_average: bool = True,
    ) -> torch.Tensor:
        """
        SSIM implementation in PyTorch.
        
        Args:
            img1: (B, 1, H, W)
            img2: (B, 1, H, W)
            window_size: Gaussian window size
            size_average: Return mean SSIM over batch
            
        Returns:
            SSIM value
        """
        channel = img1.shape[1]
        
        # Create Gaussian window
        sigma = 1.5
        gauss = torch.Tensor([
            np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
            for x in range(window_size)
        ])
        gauss = gauss / gauss.sum()
        
        # Create 2D Gaussian window
        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        window = window.to(img1.device)
        
        # Compute SSIM
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    @torch.no_grad()
    def compute_psnr(
        self,
        image1: Union[Image.Image, torch.Tensor, np.ndarray],
        image2: Union[Image.Image, torch.Tensor, np.ndarray],
        max_value: float = 1.0,
    ) -> float:
        """
        Compute PSNR (Peak Signal-to-Noise Ratio).
        Higher is better (more similar).
        
        Args:
            image1: First image
            image2: Second image
            max_value: Maximum pixel value (1.0 for normalized images)
            
        Returns:
            PSNR value in dB (float)
        """
        img1_tensor = self._to_tensor(image1, normalize=False)
        img2_tensor = self._to_tensor(image2, normalize=False)
        
        # Ensure [0, 1] range
        if img1_tensor.min() < 0:
            img1_tensor = (img1_tensor + 1) / 2
        if img2_tensor.min() < 0:
            img2_tensor = (img2_tensor + 1) / 2
        
        mse = F.mse_loss(img1_tensor, img2_tensor)
        
        if mse == 0:
            return float('inf')
        
        psnr = 20 * torch.log10(torch.tensor(max_value) / torch.sqrt(mse))
        
        return psnr.item()
    
    @torch.no_grad()
    def compute_l1(
        self,
        image1: Union[Image.Image, torch.Tensor, np.ndarray],
        image2: Union[Image.Image, torch.Tensor, np.ndarray],
    ) -> float:
        """
        Compute L1 distance (Mean Absolute Error).
        Lower is better (more similar).
        
        Args:
            image1: First image
            image2: Second image
            
        Returns:
            L1 distance (float)
        """
        img1_tensor = self._to_tensor(image1, normalize=False)
        img2_tensor = self._to_tensor(image2, normalize=False)
        
        # Ensure [0, 1] range
        if img1_tensor.min() < 0:
            img1_tensor = (img1_tensor + 1) / 2
        if img2_tensor.min() < 0:
            img2_tensor = (img2_tensor + 1) / 2
        
        l1 = F.l1_loss(img1_tensor, img2_tensor)
        
        return l1.item()
    
    @torch.no_grad()
    def compute_l2(
        self,
        image1: Union[Image.Image, torch.Tensor, np.ndarray],
        image2: Union[Image.Image, torch.Tensor, np.ndarray],
    ) -> float:
        """
        Compute L2 distance (Mean Squared Error).
        Lower is better (more similar).
        
        Args:
            image1: First image
            image2: Second image
            
        Returns:
            L2 distance (float)
        """
        img1_tensor = self._to_tensor(image1, normalize=False)
        img2_tensor = self._to_tensor(image2, normalize=False)
        
        # Ensure [0, 1] range
        if img1_tensor.min() < 0:
            img1_tensor = (img1_tensor + 1) / 2
        if img2_tensor.min() < 0:
            img2_tensor = (img2_tensor + 1) / 2
        
        l2 = F.mse_loss(img1_tensor, img2_tensor)
        
        return l2.item()
    
    def compute_all(
        self,
        image1: Union[Image.Image, torch.Tensor, np.ndarray],
        image2: Union[Image.Image, torch.Tensor, np.ndarray],
        metrics: List[str] = ['lpips', 'ssim', 'psnr', 'l1', 'l2'],
    ) -> Dict[str, float]:
        """
        Compute all requested metrics.
        
        Args:
            image1: First image
            image2: Second image
            metrics: List of metric names to compute
            
        Returns:
            Dictionary with metric values
        """
        results = {}
        
        if 'lpips' in metrics:
            results['lpips'] = self.compute_lpips(image1, image2)
        
        if 'ssim' in metrics:
            results['ssim'] = self.compute_ssim(image1, image2)
        
        if 'psnr' in metrics:
            results['psnr'] = self.compute_psnr(image1, image2)
        
        if 'l1' in metrics:
            results['l1'] = self.compute_l1(image1, image2)
        
        if 'l2' in metrics:
            results['l2'] = self.compute_l2(image1, image2)
        
        return results
    
    def evaluate_batch(
        self,
        images1: List,
        images2: List,
        metrics: List[str] = ['lpips', 'ssim', 'psnr', 'l1'],
    ) -> Dict:
        """
        Evaluate a batch of image pairs.
        
        Args:
            images1: List of first images
            images2: List of second images
            metrics: List of metric names to compute
            
        Returns:
            Dictionary with aggregated and per-image metrics
        """
        assert len(images1) == len(images2), "Mismatched batch sizes"
        
        logger.info(f"Computing image metrics for {len(images1)} pairs")
        
        # Compute per-image metrics
        per_image = []
        metric_values = {m: [] for m in metrics}
        
        for i, (img1, img2) in enumerate(zip(images1, images2)):
            img_metrics = self.compute_all(img1, img2, metrics)
            img_metrics['image_idx'] = i
            per_image.append(img_metrics)
            
            # Collect values for aggregation
            for m in metrics:
                if m in img_metrics:
                    metric_values[m].append(img_metrics[m])
        
        # Compute aggregated statistics
        aggregated = {}
        for m in metrics:
            if metric_values[m]:
                aggregated[m] = compute_statistics(metric_values[m])
        
        return {
            'aggregated': aggregated,
            'per_image': per_image,
        }



