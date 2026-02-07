"""
Classifier-based metrics using TorchXRayVision models.
"""

import torch
import logging
from typing import List, Dict, Optional, Union, Tuple
from PIL import Image
import numpy as np

from .utils import batch_images_for_model, MODEL_PREPROCESS_CONFIGS

logger = logging.getLogger(__name__)


class ClassifierMetrics:
    """
    Compute classifier-based metrics using TorchXRayVision.
    
    Metrics:
    - Target flip rate: % of edits where target pathology prediction changed as intended
    - Non-target preservation rate: % of non-target pathologies that remained unchanged
    - Confidence delta: Change in prediction confidence for each pathology
    - Unintended flips: List of pathologies that flipped unintentionally
    """
    
    # TorchXRayVision pathologies (DenseNet-121 trained on multiple datasets)
    PATHOLOGIES = [
        'Atelectasis',
        'Consolidation',
        'Infiltration',
        'Pneumothorax',
        'Edema',
        'Emphysema',
        'Fibrosis',
        'Effusion',
        'Pneumonia',
        'Pleural_Thickening',
        'Cardiomegaly',
        'Nodule',
        'Mass',
        'Hernia',
        'Lung Lesion',
        'Fracture',
        'Lung Opacity',
        'Enlarged Cardiomediastinum',
    ]
    
    # Supported model names
    SUPPORTED_MODELS = [
        'densenet121-res224-all',
        'densenet121-res224-nih',
        'densenet121-res224-pc',
        'densenet121-res224-chex',
        'densenet121-res224-mimic_nb',
        'densenet121-res224-mimic_ch',
        'resnet50-res512-all',
        'jfhealthcare',
    ]
    
    def __init__(
        self,
        model_name: str = 'densenet121-res224-all',
        device: str = 'cuda',
        confidence_threshold: float = 0.01,
        flip_threshold: float = 0.5,
    ):
        """
        Initialize classifier metrics.
        
        Args:
            model_name: TorchXRayVision model name
                - 'densenet121-res224-all' (default, recommended)
                - 'densenet121-res224-nih', 'densenet121-res224-pc', etc.
                - 'resnet50-res512-all'
                - 'jfhealthcare' (5 pathologies, 512x512)
            device: Device for inference
            confidence_threshold: Minimum confidence change to count as significant
            flip_threshold: Threshold for binary classification (0.5 standard)
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.flip_threshold = flip_threshold
        self.model_name = model_name
        
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}. Supported: {self.SUPPORTED_MODELS}")
        
        if model_name not in MODEL_PREPROCESS_CONFIGS:
            raise ValueError(f"No preprocessing config for model: {model_name}")
        
        logger.info(f"Loading TorchXRayVision model: {model_name}")
        
        try:
            import torchxrayvision as xrv
        except ImportError:
            raise ImportError(
                "torchxrayvision not installed. Install with: pip install torchxrayvision"
            )
        
        # Get preprocessing config
        self._preprocess_config = MODEL_PREPROCESS_CONFIGS[model_name]
        self.input_size = self._preprocess_config['target_size']
        
        # Load model based on type
        if model_name == 'jfhealthcare':
            self.model = xrv.baseline_models.jfhealthcare.DenseNet()
            self._use_inner_model = True  # Flag to bypass wrapper's forward()
        elif model_name.startswith('resnet'):
            self.model = xrv.models.ResNet(weights=model_name)
            self._use_inner_model = False
        else:
            # DenseNet variants
            self.model = xrv.models.DenseNet(weights=model_name)
            self._use_inner_model = False
        
        self.model = self.model.to(device)
        self.model.eval()
        
        # Store pathology names from model
        self.pathologies = list(self.model.pathologies)
        
        logger.info(f"Model loaded successfully. Pathologies: {len(self.pathologies)}")
        logger.info(f"  Input size: {self.input_size}, Channels: {self._preprocess_config['channels']}")
    
    @torch.no_grad()
    def predict(
        self,
        images: Union[List, torch.Tensor],
        input_range: Tuple[float, float] = (-1, 1),
    ) -> torch.Tensor:
        """
        Get classifier predictions for images.
        
        Uses model-specific preprocessing to avoid inefficient double normalization/resize.
        
        Args:
            images: List of images or batched tensor
            input_range: Expected input range for tensors (default: (-1, 1) from VAE output)
            
        Returns:
            Predictions tensor of shape (B, num_pathologies)
        """
        # Use model-specific preprocessing
        if isinstance(images, list):
            batch = batch_images_for_model(
                images, 
                self.model_name, 
                self.device,
                input_range=input_range,
            )
        else:
            # If already a tensor, assume it's preprocessed for this model
            batch = images.to(self.device)
            if len(batch.shape) == 3:
                batch = batch.unsqueeze(0)
        
        # Get predictions
        if self._use_inner_model:
            # Bypass jfhealthcare wrapper - input is already [-2,2], 3ch, 512x512
            # The wrapper's forward() would do: repeat(1,3,1,1), resize, /512
            # We've already done this in preprocessing, so call inner model directly
            y, _ = self.model.model(batch)
            y = torch.cat(y, 1)
            if self.model.apply_sigmoid:
                y = torch.sigmoid(y)
            predictions = y
        else:
            # Standard models - use normal forward()
            predictions = self.model(batch)
                
        return predictions
    
    def compute_flip_rate(
        self,
        preds_before: torch.Tensor,
        preds_after: torch.Tensor,
        target_pathology: Optional[str] = None,
        direction: Optional[str] = None,  # 'increase' or 'decrease'
    ) -> Dict:
        """
        Compute flip rates and confidence changes.
        
        Args:
            preds_before: Predictions before edit (B, num_pathologies)
            preds_after: Predictions after edit (B, num_pathologies)
            target_pathology: Target pathology to track (optional)
            direction: Expected direction of change ('increase' or 'decrease')
            
        Returns:
            Dictionary with flip rate statistics
        """
        batch_size = preds_before.shape[0]
        num_pathologies = preds_before.shape[1]
        
        # Compute binary predictions
        binary_before = (preds_before > self.flip_threshold).float()
        binary_after = (preds_after > self.flip_threshold).float()
        
        # Detect flips (any change in binary prediction)
        flips = (binary_before != binary_after).float()  # (B, num_pathologies)
        
        # Compute confidence changes
        confidence_deltas = preds_after - preds_before  # (B, num_pathologies)
        
        # Significant changes (above threshold)
        significant_changes = (torch.abs(confidence_deltas) > self.confidence_threshold).float()
        
        results = {
            'total_images': batch_size,
            'total_pathologies': num_pathologies,
        }
        
        # Per-pathology statistics
        per_pathology = {}
        for i, pathology_name in enumerate(self.pathologies):
            pathology_flips = flips[:, i].sum().item()
            pathology_significant = significant_changes[:, i].sum().item()
            
            per_pathology[pathology_name] = {
                'flip_count': pathology_flips,
                'flip_rate': pathology_flips / batch_size,
                'significant_change_count': pathology_significant,
                'significant_change_rate': pathology_significant / batch_size,
                'mean_confidence_delta': confidence_deltas[:, i].mean().item(),
                'mean_pred_before': preds_before[:, i].mean().item(),
                'mean_pred_after': preds_after[:, i].mean().item(),
            }
        
        results['per_pathology'] = per_pathology
        
        # Target pathology specific metrics
        if target_pathology:
            if target_pathology not in self.pathologies:
                logger.warning(f"Target pathology '{target_pathology}' not in model pathologies")
                results['target_metrics'] = None
            else:
                target_idx = self.pathologies.index(target_pathology)
                
                target_flips = flips[:, target_idx]
                target_deltas = confidence_deltas[:, target_idx]
                
                # Check if flip is in the intended direction
                if direction == 'increase':
                    intended_flips = (target_deltas > self.confidence_threshold).float()
                elif direction == 'decrease':
                    intended_flips = (target_deltas < -self.confidence_threshold).float()
                else:
                    # Any significant change
                    intended_flips = (torch.abs(target_deltas) > self.confidence_threshold).float()
                
                results['target_metrics'] = {
                    'pathology': target_pathology,
                    'direction': direction,
                    'flip_rate': target_flips.mean().item(),
                    'intended_flip_rate': intended_flips.mean().item(),
                    'mean_confidence_delta': target_deltas.mean().item(),
                    'median_confidence_delta': target_deltas.median().item(),
                }
        
        # Non-target preservation (if target specified)
        if target_pathology and target_pathology in self.pathologies:
            target_idx = self.pathologies.index(target_pathology)
            
            # Mask out target pathology
            non_target_mask = torch.ones(num_pathologies, dtype=torch.bool)
            non_target_mask[target_idx] = False
            
            non_target_flips = flips[:, non_target_mask]
            non_target_preserved = (non_target_flips == 0).float()
            
            # Per-image preservation rate
            preservation_per_image = non_target_preserved.mean(dim=1)  # (B,)
            
            results['non_target_preservation'] = {
                'mean_rate': preservation_per_image.mean().item(),
                'median_rate': preservation_per_image.median().item(),
                'perfect_preservation_count': (preservation_per_image == 1.0).sum().item(),
                'perfect_preservation_rate': (preservation_per_image == 1.0).float().mean().item(),
            }
            
            # Identify which pathologies flipped unintentionally
            unintended_flip_counts = non_target_flips.sum(dim=0)
            unintended_pathologies = []
            
            non_target_pathologies = [p for p in self.pathologies if p != target_pathology]
            for i, count in enumerate(unintended_flip_counts):
                if count > 0:
                    unintended_pathologies.append({
                        'pathology': non_target_pathologies[i],
                        'flip_count': count.item(),
                        'flip_rate': (count / batch_size).item(),
                    })
            
            results['unintended_flips'] = sorted(
                unintended_pathologies, 
                key=lambda x: x['flip_rate'], 
                reverse=True
            )
        
        return results
    
    def evaluate_pair(
        self,
        image_before: Union[Image.Image, torch.Tensor, np.ndarray],
        image_after: Union[Image.Image, torch.Tensor, np.ndarray],
        target_pathology: Optional[str] = None,
        direction: Optional[str] = None,
    ) -> Dict:
        """
        Evaluate a single image pair.
        
        Args:
            image_before: Original image
            image_after: Edited image
            target_pathology: Target pathology for editing
            direction: Expected direction ('increase' or 'decrease')
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Get predictions
        pred_before = self.predict([image_before])
        pred_after = self.predict([image_after])
        
        # Compute flip rates
        results = self.compute_flip_rate(
            pred_before,
            pred_after,
            target_pathology=target_pathology,
            direction=direction,
        )
        
        # Add per-image details
        results['predictions_before'] = {
            p: pred_before[0, i].item() 
            for i, p in enumerate(self.pathologies)
        }
        results['predictions_after'] = {
            p: pred_after[0, i].item() 
            for i, p in enumerate(self.pathologies)
        }
        
        return results
    
    def evaluate_batch(
        self,
        images_before: List,
        images_after: List,
        target_pathology: Optional[str] = None,
        direction: Optional[str] = None,
    ) -> Dict:
        """
        Evaluate a batch of image pairs.
        
        Args:
            images_before: List of original images
            images_after: List of edited images
            target_pathology: Target pathology for editing
            direction: Expected direction ('increase' or 'decrease')
            
        Returns:
            Dictionary with aggregated and per-image metrics
        """
        assert len(images_before) == len(images_after), "Mismatched batch sizes"
        
        logger.info(f"Evaluating batch of {len(images_before)} image pairs")
        
        # Get predictions for all images
        preds_before = self.predict(images_before)
        preds_after = self.predict(images_after)
        
        # Compute aggregated metrics
        aggregated = self.compute_flip_rate(
            preds_before,
            preds_after,
            target_pathology=target_pathology,
            direction=direction,
        )
        
        # Compute per-image metrics
        per_image = []
        for i in range(len(images_before)):
            img_results = {
                'image_idx': i,
                'predictions_before': {
                    p: preds_before[i, j].item() 
                    for j, p in enumerate(self.pathologies)
                },
                'predictions_after': {
                    p: preds_after[i, j].item() 
                    for j, p in enumerate(self.pathologies)
                },
                'confidence_deltas': {
                    p: (preds_after[i, j] - preds_before[i, j]).item()
                    for j, p in enumerate(self.pathologies)
                },
            }
            
            # Target-specific metrics
            if target_pathology and target_pathology in self.pathologies:
                target_idx = self.pathologies.index(target_pathology)
                delta = preds_after[i, target_idx] - preds_before[i, target_idx]
                
                img_results['target_flip'] = abs(delta.item()) > self.confidence_threshold
                img_results['target_delta'] = delta.item()
                
                # Count non-target flips
                non_target_flips = 0
                for j, p in enumerate(self.pathologies):
                    if j != target_idx:
                        if abs(preds_after[i, j] - preds_before[i, j]) > self.confidence_threshold:
                            non_target_flips += 1
                
                img_results['non_target_flips'] = non_target_flips
            
            per_image.append(img_results)
        
        return {
            'aggregated': aggregated,
            'per_image': per_image,
        }


class MultiClassifierMetrics:
    """
    Wrapper for running multiple classifiers in parallel with separate preprocessing per model.
    
    Each model gets its own preprocessing pipeline to avoid inefficient double normalization/resize.
    Returns per-model results for comparison.
    
    Usage:
        multi_clf = MultiClassifierMetrics(
            model_names=['densenet121-res224-all', 'jfhealthcare'],
            device='cuda',
        )
        
        # Get predictions from all classifiers
        results = multi_clf.predict_all([image1, image2, ...])
        # Returns: {'densenet121-res224-all': tensor, 'jfhealthcare': tensor}
        
        # Evaluate batch with all classifiers
        results = multi_clf.evaluate_batch(images_before, images_after, target_pathology='Cardiomegaly')
        # Returns: {'densenet121-res224-all': {...}, 'jfhealthcare': {...}}
    """
    
    def __init__(
        self,
        model_names: List[str] = None,
        device: str = 'cuda',
        confidence_threshold: float = 0.1,
        flip_threshold: float = 0.5,
    ):
        """
        Initialize multi-classifier metrics.
        
        Args:
            model_names: List of model names to use. Default: ['densenet121-res224-all', 'jfhealthcare']
            device: Device for inference
            confidence_threshold: Minimum confidence change to count as significant
            flip_threshold: Threshold for binary classification
        """
        if model_names is None:
            model_names = ['densenet121-res224-all', 'jfhealthcare']
        
        self.model_names = model_names
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.flip_threshold = flip_threshold
        
        logger.info(f"Initializing MultiClassifierMetrics with models: {model_names}")
        
        # Initialize each classifier
        self.classifiers: Dict[str, ClassifierMetrics] = {}
        for name in model_names:
            self.classifiers[name] = ClassifierMetrics(
                model_name=name,
                device=device,
                confidence_threshold=confidence_threshold,
                flip_threshold=flip_threshold,
            )
        
        logger.info(f"MultiClassifierMetrics initialized with {len(self.classifiers)} classifiers")
    
    def get_all_pathologies(self) -> Dict[str, List[str]]:
        """Get pathologies supported by each classifier."""
        return {name: clf.pathologies for name, clf in self.classifiers.items()}
    
    def get_common_pathologies(self) -> List[str]:
        """Get pathologies supported by ALL classifiers."""
        if not self.classifiers:
            return []
        
        pathology_sets = [set(clf.pathologies) for clf in self.classifiers.values()]
        common = pathology_sets[0]
        for ps in pathology_sets[1:]:
            common = common.intersection(ps)
        
        return list(common)
    
    @torch.no_grad()
    def predict_all(
        self,
        images: List,
        input_range: Tuple[float, float] = (-1, 1),
    ) -> Dict[str, torch.Tensor]:
        """
        Get predictions from all classifiers.
        
        Each model preprocesses the images separately to avoid quality loss.
        
        Args:
            images: List of images
            input_range: Expected input range for tensors
            
        Returns:
            Dict mapping model_name -> predictions tensor
        """
        results = {}
        for name, classifier in self.classifiers.items():
            results[name] = classifier.predict(images, input_range=input_range)
        return results
    
    def evaluate_pair(
        self,
        image_before: Union[Image.Image, torch.Tensor, np.ndarray],
        image_after: Union[Image.Image, torch.Tensor, np.ndarray],
        target_pathology: Optional[str] = None,
        direction: Optional[str] = None,
    ) -> Dict[str, Dict]:
        """
        Evaluate a single image pair with all classifiers.
        
        Args:
            image_before: Original image
            image_after: Edited image
            target_pathology: Target pathology for editing
            direction: Expected direction ('increase' or 'decrease')
            
        Returns:
            Dict mapping model_name -> evaluation results
        """
        results = {}
        for name, classifier in self.classifiers.items():
            # Skip if target pathology not supported by this classifier
            if target_pathology and target_pathology not in classifier.pathologies:
                logger.info(f"Skipping {name}: '{target_pathology}' not in pathologies")
                results[name] = {
                    'skipped': True,
                    'reason': f"Target pathology '{target_pathology}' not supported",
                    'supported_pathologies': classifier.pathologies,
                }
                continue
            
            results[name] = classifier.evaluate_pair(
                image_before,
                image_after,
                target_pathology=target_pathology,
                direction=direction,
            )
        
        return results
    
    def evaluate_batch(
        self,
        images_before: List,
        images_after: List,
        target_pathology: Optional[str] = None,
        direction: Optional[str] = None,
    ) -> Dict[str, Dict]:
        """
        Evaluate a batch of image pairs with all classifiers.
        
        Each classifier preprocesses separately to avoid quality loss from double resize/normalize.
        
        Args:
            images_before: List of original images
            images_after: List of edited images
            target_pathology: Target pathology for editing
            direction: Expected direction ('increase' or 'decrease')
            
        Returns:
            Dict mapping model_name -> evaluation results
            Example: {'densenet121-res224-all': {...}, 'jfhealthcare': {...}}
        """
        assert len(images_before) == len(images_after), "Mismatched batch sizes"
        
        logger.info(f"Evaluating batch of {len(images_before)} pairs with {len(self.classifiers)} classifiers")
        
        results = {}
        for name, classifier in self.classifiers.items():
            # Check if target pathology is supported
            if target_pathology and target_pathology not in classifier.pathologies:
                logger.info(f"Skipping target metrics for {name}: '{target_pathology}' not in pathologies")
                # Still evaluate, but without target-specific metrics
                results[name] = classifier.evaluate_batch(
                    images_before,
                    images_after,
                    target_pathology=None,  # Don't track target
                    direction=None,
                )
                results[name]['target_not_supported'] = True
                results[name]['supported_pathologies'] = classifier.pathologies
            else:
                results[name] = classifier.evaluate_batch(
                    images_before,
                    images_after,
                    target_pathology=target_pathology,
                    direction=direction,
                )
        
        return results
    
    def compute_flip_rate(
        self,
        preds_before: Dict[str, torch.Tensor],
        preds_after: Dict[str, torch.Tensor],
        target_pathology: Optional[str] = None,
        direction: Optional[str] = None,
    ) -> Dict[str, Dict]:
        """
        Compute flip rates for all classifiers given pre-computed predictions.
        
        Args:
            preds_before: Dict of model_name -> predictions before edit
            preds_after: Dict of model_name -> predictions after edit
            target_pathology: Target pathology to track
            direction: Expected direction of change
            
        Returns:
            Dict mapping model_name -> flip rate statistics
        """
        results = {}
        for name, classifier in self.classifiers.items():
            if name not in preds_before or name not in preds_after:
                logger.warning(f"Missing predictions for {name}")
                continue
            
            # Check if target pathology is supported
            tp = target_pathology if target_pathology in classifier.pathologies else None
            dr = direction if tp else None
            
            results[name] = classifier.compute_flip_rate(
                preds_before[name],
                preds_after[name],
                target_pathology=tp,
                direction=dr,
            )
            
            if target_pathology and target_pathology not in classifier.pathologies:
                results[name]['target_not_supported'] = True
        
        return results



