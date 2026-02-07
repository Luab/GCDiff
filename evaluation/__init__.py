"""
Evaluation module for measuring edit quality on chest X-ray images.

Uses torchxrayvision for classifier-based metrics (flip rates, confidence)
and standard metrics for image quality (LPIPS, SSIM, etc.).
"""

from .evaluator import EditEvaluator
from .classifier_metrics import ClassifierMetrics, MultiClassifierMetrics
from .image_metrics import ImageMetrics
from .frd_metrics import FRDMetrics, compute_frd_for_output_folder

__all__ = [
    'EditEvaluator', 
    'ClassifierMetrics', 
    'MultiClassifierMetrics',
    'ImageMetrics', 
    'FRDMetrics', 
    'compute_frd_for_output_folder',
]



