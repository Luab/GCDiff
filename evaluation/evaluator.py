"""
Main evaluator for edit quality assessment.
"""

import logging
from typing import List, Dict, Optional, Union
from PIL import Image
import torch
import numpy as np

from .classifier_metrics import ClassifierMetrics, MultiClassifierMetrics
from .image_metrics import ImageMetrics

logger = logging.getLogger(__name__)


class EditEvaluator:
    """
    Comprehensive evaluator for image editing quality.
    
    Combines classifier-based metrics (flip rates, confidence) with
    image quality metrics (LPIPS, SSIM, etc.).
    
    Usage:
        evaluator = EditEvaluator(device='cuda')
        
        # Single pair
        results = evaluator.evaluate_pair(
            original_image,
            edited_image,
            target_pathology='Cardiomegaly',
            direction='increase'
        )
        
        # Batch evaluation
        results = evaluator.evaluate_batch(
            original_images,
            edited_images,
            target_pathology='Cardiomegaly',
            compute_image_metrics=True
        )
    """
    
    def __init__(
        self,
        classifier_model: Union[str, List[str]] = 'densenet121-res224-all',
        device: str = 'cuda',
        confidence_threshold: float = 0.1,
        lpips_net: str = 'alex',
        wandb_log: bool = False,
    ):
        """
        Initialize evaluator.
        
        Args:
            classifier_model: TorchXRayVision model name or list of model names
                - Single model: 'densenet121-res224-all', 'jfhealthcare', etc.
                - Multiple models: ['densenet121-res224-all', 'jfhealthcare']
            device: Device for computation
            confidence_threshold: Minimum confidence change for significance
            lpips_net: LPIPS backbone ('alex' or 'vgg')
            wandb_log: Whether to log to wandb
        """
        self.device = device
        self.wandb_log = wandb_log
        self._multi_classifier_mode = isinstance(classifier_model, list)
        
        # Initialize classifier metrics
        logger.info("Initializing classifier metrics...")
        if self._multi_classifier_mode:
            logger.info(f"Multi-classifier mode with models: {classifier_model}")
            self.classifier_metrics = MultiClassifierMetrics(
                model_names=classifier_model,
                device=device,
                confidence_threshold=confidence_threshold,
            )
        else:
            self.classifier_metrics = ClassifierMetrics(
                model_name=classifier_model,
                device=device,
                confidence_threshold=confidence_threshold,
            )
        
        # Initialize image metrics
        logger.info("Initializing image metrics...")
        self.image_metrics = ImageMetrics(
            device=device,
            lpips_net=lpips_net,
        )
        
        logger.info("Evaluator initialized successfully")
    
    def evaluate_pair(
        self,
        image_before: Union[Image.Image, torch.Tensor, np.ndarray],
        image_after: Union[Image.Image, torch.Tensor, np.ndarray],
        target_pathology: Optional[str] = None,
        direction: Optional[str] = None,
        compute_image_metrics: bool = True,
        image_metrics_list: List[str] = ['lpips', 'ssim', 'psnr', 'l1'],
    ) -> Dict:
        """
        Evaluate a single image pair.
        
        Args:
            image_before: Original image
            image_after: Edited image
            target_pathology: Target pathology for editing (e.g., 'Cardiomegaly')
            direction: Expected direction ('increase' or 'decrease')
            compute_image_metrics: Whether to compute image quality metrics
            image_metrics_list: Which image metrics to compute
            
        Returns:
            Dictionary with all evaluation results
        """
        logger.info("Evaluating single image pair")
        
        results = {}
        
        # Classifier-based metrics
        logger.debug("Computing classifier metrics...")
        classifier_results = self.classifier_metrics.evaluate_pair(
            image_before,
            image_after,
            target_pathology=target_pathology,
            direction=direction,
        )
        results['classifier'] = classifier_results
        
        # Image quality metrics
        if compute_image_metrics:
            logger.debug("Computing image metrics...")
            image_results = self.image_metrics.compute_all(
                image_before,
                image_after,
                metrics=image_metrics_list,
            )
            results['image'] = image_results
        
        # Log to wandb if enabled
        if self.wandb_log:
            self._log_to_wandb(results, prefix='single_pair')
        
        return results
    
    def evaluate_batch(
        self,
        images_before: List,
        images_after: List,
        target_pathology: Optional[str] = None,
        direction: Optional[str] = None,
        compute_image_metrics: bool = True,
        image_metrics_list: List[str] = ['lpips', 'ssim', 'l1'],
    ) -> Dict:
        """
        Evaluate a batch of image pairs.
        
        Args:
            images_before: List of original images
            images_after: List of edited images
            target_pathology: Target pathology for editing
            direction: Expected direction ('increase' or 'decrease')
            compute_image_metrics: Whether to compute image quality metrics
            image_metrics_list: Which image metrics to compute
            
        Returns:
            Dictionary with aggregated and per-image results
        """
        assert len(images_before) == len(images_after), "Mismatched batch sizes"
        
        batch_size = len(images_before)
        logger.info(f"Evaluating batch of {batch_size} image pairs")
        
        results = {
            'batch_size': batch_size,
            'target_pathology': target_pathology,
            'direction': direction,
        }
        
        # Classifier-based metrics
        logger.info("Computing classifier metrics for batch...")
        classifier_results = self.classifier_metrics.evaluate_batch(
            images_before,
            images_after,
            target_pathology=target_pathology,
            direction=direction,
        )
        results['classifier'] = classifier_results
        
        # Image quality metrics
        if compute_image_metrics:
            logger.info("Computing image metrics for batch...")
            image_results = self.image_metrics.evaluate_batch(
                images_before,
                images_after,
                metrics=image_metrics_list,
            )
            results['image'] = image_results
        
        # Log to wandb if enabled
        if self.wandb_log:
            self._log_to_wandb(results, prefix='batch')
        
        # Log summary
        self._log_summary(results)
        
        return results
    
    def _log_summary(self, results: Dict):
        """Log summary statistics to logger."""
        logger.info("=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        
        if 'classifier' in results:
            classifier_results = results['classifier']
            
            if self._multi_classifier_mode:
                # Multi-classifier mode: results keyed by model name
                for model_name, model_results in classifier_results.items():
                    logger.info(f"\n[{model_name}]")
                    self._log_single_classifier_summary(model_results)
            else:
                # Single classifier mode
                self._log_single_classifier_summary(classifier_results)
        
        # Image metrics
        if 'image' in results and 'aggregated' in results['image']:
            logger.info("\nImage Quality Metrics:")
            for metric_name, stats in results['image']['aggregated'].items():
                logger.info(f"  {metric_name.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        logger.info("=" * 60)
    
    def _log_single_classifier_summary(self, results: Dict):
        """Log summary for a single classifier's results."""
        if results.get('target_not_supported'):
            logger.info("  Target pathology not supported by this model")
            return
        
        if results.get('skipped'):
            logger.info(f"  Skipped: {results.get('reason', 'Unknown')}")
            return
        
        agg = results.get('aggregated', results)
        
        # Target metrics
        if agg.get('target_metrics'):
            tm = agg['target_metrics']
            logger.info(f"  Target Pathology: {tm['pathology']}")
            logger.info(f"    Flip Rate: {tm['flip_rate']:.2%}")
            logger.info(f"    Intended Flip Rate: {tm['intended_flip_rate']:.2%}")
            logger.info(f"    Mean Confidence Delta: {tm['mean_confidence_delta']:+.3f}")
        
        # Non-target preservation
        if agg.get('non_target_preservation'):
            ntp = agg['non_target_preservation']
            logger.info(f"  Non-Target Preservation:")
            logger.info(f"    Mean Rate: {ntp['mean_rate']:.2%}")
            logger.info(f"    Perfect Preservation Rate: {ntp['perfect_preservation_rate']:.2%}")
        
        # Unintended flips
        if agg.get('unintended_flips'):
            logger.info(f"  Unintended Flips: {len(agg['unintended_flips'])} pathologies")
            for flip in agg['unintended_flips'][:3]:  # Top 3
                logger.info(f"    - {flip['pathology']}: {flip['flip_rate']:.2%}")
    
    def _log_to_wandb(self, results: Dict, prefix: str = ''):
        """Log results to wandb if enabled."""
        try:
            import wandb
            
            if not wandb.run:
                logger.warning("wandb logging enabled but no active run")
                return
            
            log_dict = {}
            
            # Classifier metrics
            if 'classifier' in results:
                if 'aggregated' in results['classifier']:
                    agg = results['classifier']['aggregated']
                    
                    if agg.get('target_metrics'):
                        tm = agg['target_metrics']
                        log_dict.update({
                            f'{prefix}/target_flip_rate': tm['flip_rate'],
                            f'{prefix}/target_intended_flip_rate': tm['intended_flip_rate'],
                            f'{prefix}/target_confidence_delta': tm['mean_confidence_delta'],
                        })
                    
                    if agg.get('non_target_preservation'):
                        ntp = agg['non_target_preservation']
                        log_dict.update({
                            f'{prefix}/non_target_preservation_mean': ntp['mean_rate'],
                            f'{prefix}/non_target_preservation_perfect': ntp['perfect_preservation_rate'],
                        })
            
            # Image metrics
            if 'image' in results:
                if isinstance(results['image'], dict) and 'aggregated' in results['image']:
                    for metric_name, stats in results['image']['aggregated'].items():
                        log_dict[f'{prefix}/image_{metric_name}_mean'] = stats['mean']
                        log_dict[f'{prefix}/image_{metric_name}_std'] = stats['std']
                else:
                    # Single image case
                    for metric_name, value in results['image'].items():
                        log_dict[f'{prefix}/image_{metric_name}'] = value
            
            if log_dict:
                wandb.log(log_dict)
                logger.debug(f"Logged {len(log_dict)} metrics to wandb")
                
        except ImportError:
            logger.warning("wandb not installed, skipping wandb logging")
        except Exception as e:
            logger.error(f"Error logging to wandb: {e}")
    
    @property
    def pathologies(self) -> Union[List[str], Dict[str, List[str]]]:
        """
        Get list of pathologies from classifier(s).
        
        Returns:
            - Single classifier mode: List of pathology names
            - Multi-classifier mode: Dict mapping model_name -> List of pathologies
        """
        if self._multi_classifier_mode:
            return self.classifier_metrics.get_all_pathologies()
        else:
            return self.classifier_metrics.pathologies
    
    @property
    def common_pathologies(self) -> List[str]:
        """
        Get pathologies supported by ALL classifiers.
        
        Useful in multi-classifier mode to find common targets.
        """
        if self._multi_classifier_mode:
            return self.classifier_metrics.get_common_pathologies()
        else:
            return self.classifier_metrics.pathologies
