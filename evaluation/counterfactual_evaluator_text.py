"""
Counterfactual evaluation for text-conditioned image generation.

Evaluates how well RadEdit responds to text prompt edits by:
1. Using text-based counterfactual generators to produce edited images
2. Measuring pathology flip rates and image quality

Uses RadEdit's native BioViL-T text conditioning (no ControlNet/graphs).
This serves as a baseline comparison for graph-based counterfactual generation.

Supports methods:
- text_ddim: DDIM inversion + text reconstruction
- text_ddpm: DDPM "Edit Friendly" inversion + text reconstruction
"""

import sys
import os
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from counterfactual.text_inversion import (
    TextDDIMInversionGenerator,
    TextDDPMInversionGenerator,
    AVAILABLE_TEMPLATE_SETS,
)
from counterfactual.base import BaseCounterfactualGenerator
from data_loader import CheXpertGraphDataset, get_dataloaders
from evaluation.evaluator import EditEvaluator
from evaluation.classifier_metrics import ClassifierMetrics
from evaluation.utils import create_comparison_grid
from evaluation.frd_metrics import FRDMetrics

logger = logging.getLogger(__name__)

# Generator registry for text-based methods
TEXT_GENERATOR_REGISTRY = {
    'text_ddim': TextDDIMInversionGenerator,
    'text_ddpm': TextDDPMInversionGenerator,
    'ddim': TextDDIMInversionGenerator,  # Alias
    'ddpm': TextDDPMInversionGenerator,  # Alias
    # Fixed methods with independent noise sampling (per "Edit Friendly DDPM" paper Eq. 6)
    'text_ddpm_fixed': TextDDPMInversionGenerator,
    'ddpm_fixed': TextDDPMInversionGenerator,
}

# Methods that require use_independent_noise=True
FIXED_METHODS = {'text_ddpm_fixed', 'ddpm_fixed'}


class TextCounterfactualEvaluator:
    """
    Evaluates text-based counterfactual image generation quality.
    
    This class orchestrates the evaluation pipeline:
    1. Iterates through a dataloader
    2. Uses a text-based generator to produce counterfactual images
    3. Computes metrics comparing original vs counterfactual
    
    Args:
        generator: Text-based counterfactual generator instance
        device: Device for evaluation metrics computation
        classifier_models: Single model name or list of models for multi-classifier evaluation
    """
    
    def __init__(
        self,
        generator: BaseCounterfactualGenerator,
        device: str = 'cuda',
        classifier_models: Union[str, List[str]] = 'densenet121-res224-all',
    ):
        self.generator = generator
        self.device = device
        
        logger.info("Initializing TextCounterfactualEvaluator...")
        
        # Initialize edit evaluator for metrics (supports multi-classifier)
        self.edit_evaluator = EditEvaluator(
            classifier_model=classifier_models,
            device=device,
        )
        
        logger.info("TextCounterfactualEvaluator initialized successfully")
    
    def evaluate(
        self,
        dataloader: DataLoader,
        target_pathology: str,
        num_batches: Optional[int] = None,
        save_images: bool = False,
        output_dir: Optional[str] = None,
    ) -> Dict:
        """
        Run counterfactual evaluation on a dataloader.
        
        Args:
            dataloader: DataLoader with text_prompt available
            target_pathology: Target pathology to edit
            num_batches: Number of batches to evaluate (None = all)
            save_images: Whether to save generated images
            output_dir: Directory to save images (if save_images=True)
            
        Returns:
            Dictionary with evaluation results
        """
        if target_pathology not in ClassifierMetrics.PATHOLOGIES:
            raise ValueError(f"Unknown pathology: {target_pathology}. "
                           f"Available: {ClassifierMetrics.PATHOLOGIES}")
        
        logger.info(f"Starting text-based counterfactual evaluation for {target_pathology}")
        
        if save_images and output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            # Create subdirectories for organized output
            (output_path / "original").mkdir(exist_ok=True)
            (output_path / "reconstructed").mkdir(exist_ok=True)
            (output_path / "increase").mkdir(exist_ok=True)
            (output_path / "decrease").mkdir(exist_ok=True)
            (output_path / "grids").mkdir(exist_ok=True)
        
        # Collect all results
        all_original_images = []
        all_reconstructed_images = []
        all_counterfactual_images = []
        all_directions = []
        all_source_prompts = []
        all_target_prompts = []
        total_edit_stats = {'added': 0, 'removed': 0}
        has_reconstruction = False  # Track if generator provides reconstructions
        
        # Determine number of batches
        total_batches = len(dataloader) if num_batches is None else min(num_batches, len(dataloader))
        
        logger.info(f"Processing {total_batches} batches...")
        
        sample_idx = 0
        for batch_idx, batch in enumerate(tqdm(dataloader, total=total_batches, desc="Generating text counterfactuals")):
            if num_batches is not None and batch_idx >= num_batches:
                break
            
            # Generate counterfactuals using the text generator
            result = self.generator.generate_batch(batch, target_pathology)
            
            # Collect results
            all_original_images.extend(result['original_images'])
            all_counterfactual_images.extend(result['counterfactual_images'])
            all_directions.extend(result['directions'])
            
            # Collect reconstructed images if available (for inversion-based methods)
            if 'reconstructed_images' in result:
                all_reconstructed_images.extend(result['reconstructed_images'])
                has_reconstruction = True
            
            # Collect metadata
            metadata = result['metadata']
            all_source_prompts.extend(metadata.get('source_prompts', []))
            all_target_prompts.extend(metadata.get('target_prompts', []))
            total_edit_stats['added'] += metadata['edit_stats']['added']
            total_edit_stats['removed'] += metadata['edit_stats']['removed']
            
            # Save images if requested
            if save_images and output_dir:
                reconstructed_list = result.get('reconstructed_images', [None] * len(result['original_images']))
                for j, (orig, recon, cf, direction) in enumerate(zip(
                    result['original_images'],
                    reconstructed_list,
                    result['counterfactual_images'],
                    result['directions']
                )):
                    # Save individual images to organized folders
                    orig.save(output_path / "original" / f"sample_{sample_idx:04d}.png")
                    if recon is not None:
                        recon.save(output_path / "reconstructed" / f"sample_{sample_idx:04d}.png")
                    cf.save(output_path / direction / f"sample_{sample_idx:04d}.png")
                    
                    # Create and save comparison grid
                    grid = create_comparison_grid(orig, cf, direction)
                    grid.save(output_path / "grids" / f"sample_{sample_idx:04d}_{direction}.png")
                    
                    sample_idx += 1
            else:
                sample_idx += len(result['original_images'])
        
        # Compute evaluation metrics
        logger.info("Computing evaluation metrics...")
        
        num_samples = len(all_original_images)
        
        results = {
            'target_pathology': target_pathology,
            'num_samples': num_samples,
            'edit_stats': total_edit_stats,
            'has_reconstruction': has_reconstruction,
        }
        
        # Compute reconstruction metrics if available (separates inversion error from edit effect)
        if has_reconstruction and len(all_reconstructed_images) == num_samples:
            logger.info("Computing inversion fidelity metrics (Original vs Reconstructed)...")
            inversion_metrics = self.edit_evaluator.image_metrics.evaluate_batch(
                all_original_images,
                all_reconstructed_images,
                metrics=['lpips', 'ssim', 'l1'],
            )
            results['inversion_fidelity'] = inversion_metrics
            
            logger.info("Computing pure edit effect metrics (Reconstructed vs Counterfactual)...")
            edit_effect_metrics = self.edit_evaluator.image_metrics.evaluate_batch(
                all_reconstructed_images,
                all_counterfactual_images,
                metrics=['lpips', 'ssim', 'l1'],
            )
            results['edit_effect'] = edit_effect_metrics
        
        # Evaluate increase and decrease separately
        for direction in ['increase', 'decrease']:
            direction_mask = [d == direction for d in all_directions]
            orig_subset = [img for img, m in zip(all_original_images, direction_mask) if m]
            cf_subset = [img for img, m in zip(all_counterfactual_images, direction_mask) if m]
            
            if len(orig_subset) > 0:
                logger.info(f"Evaluating {len(orig_subset)} samples with direction={direction}")
                eval_results = self.edit_evaluator.evaluate_batch(
                    images_before=orig_subset,
                    images_after=cf_subset,
                    target_pathology=target_pathology,
                    direction=direction,
                    compute_image_metrics=True,
                )
                results[f'evaluation_{direction}'] = eval_results
        
        # Combined evaluation (all samples)
        logger.info(f"Computing combined evaluation for {num_samples} samples")
        combined_eval = self.edit_evaluator.evaluate_batch(
            images_before=all_original_images,
            images_after=all_counterfactual_images,
            target_pathology=target_pathology,
            direction=None,
            compute_image_metrics=True,
        )
        results['evaluation_combined'] = combined_eval
        
        # Compute FRD (distribution-level) metrics if images were saved
        if save_images and output_dir:
            logger.info("Computing FRD (Frechet Radiomic Distance) metrics...")
            try:
                frd_metrics = FRDMetrics(parallelize=True, force_recompute=False)
                if frd_metrics.is_available:
                    frd_results = {}
                    
                    # Build comparison folders
                    original_folder = str(output_path / "original")
                    comparison_folders = {}
                    for name in ["increase", "decrease", "reconstructed"]:
                        folder = output_path / name
                        if folder.exists() and any(folder.iterdir()):
                            comparison_folders[name] = str(folder)
                    
                    # Compute FRD for each comparison
                    if comparison_folders:
                        frd_results = frd_metrics.compute_all_comparisons(
                            original_folder=original_folder,
                            comparison_folders=comparison_folders,
                        )
                        results['frd'] = frd_results
                        logger.info(f"FRD results: {frd_results}")
                    else:
                        logger.warning("No comparison folders found for FRD computation")
                else:
                    logger.warning("FRD computation skipped - dependencies not available")
            except Exception as e:
                logger.error(f"FRD computation failed: {e}")
        
        # Log summary
        self._log_summary(results)
        
        return results
    
    def _log_summary(self, results: Dict):
        """Log evaluation summary."""
        logger.info("=" * 70)
        logger.info("TEXT COUNTERFACTUAL EVALUATION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Target Pathology: {results['target_pathology']}")
        logger.info(f"Total Samples: {results['num_samples']}")
        logger.info(f"Edit Stats: Added={results['edit_stats']['added']}, "
                   f"Removed={results['edit_stats']['removed']}")
        
        # Log per-direction results
        for direction in ['increase', 'decrease']:
            key = f'evaluation_{direction}'
            if key in results and results[key]:
                eval_data = results[key]
                if 'classifier' in eval_data and 'aggregated' in eval_data['classifier']:
                    agg = eval_data['classifier']['aggregated']
                    if agg.get('target_metrics'):
                        tm = agg['target_metrics']
                        logger.info(f"\n{direction.upper()} Direction:")
                        logger.info(f"  Target Flip Rate: {tm['flip_rate']:.2%}")
                        logger.info(f"  Intended Flip Rate: {tm['intended_flip_rate']:.2%}")
                        logger.info(f"  Mean Confidence Delta: {tm['mean_confidence_delta']:+.4f}")
        
        # Log separated image quality metrics if reconstruction is available
        if results.get('has_reconstruction') and 'inversion_fidelity' in results:
            logger.info("\n" + "-" * 50)
            logger.info("IMAGE QUALITY BREAKDOWN")
            logger.info("-" * 50)
            
            # Inversion fidelity (Original vs Reconstructed)
            inv = results['inversion_fidelity'].get('aggregated', {})
            logger.info("Inversion Fidelity (Original vs Reconstructed):")
            logger.info(f"  LPIPS: {inv.get('lpips', {}).get('mean', 0):.4f} "
                       f"± {inv.get('lpips', {}).get('std', 0):.4f}")
            logger.info(f"  SSIM:  {inv.get('ssim', {}).get('mean', 0):.4f} "
                       f"± {inv.get('ssim', {}).get('std', 0):.4f}")
            logger.info(f"  L1:    {inv.get('l1', {}).get('mean', 0):.4f} "
                       f"± {inv.get('l1', {}).get('std', 0):.4f}")
            
            # Pure edit effect (Reconstructed vs Counterfactual)
            edit = results['edit_effect'].get('aggregated', {})
            logger.info("Pure Edit Effect (Reconstructed vs Counterfactual):")
            logger.info(f"  LPIPS: {edit.get('lpips', {}).get('mean', 0):.4f} "
                       f"± {edit.get('lpips', {}).get('std', 0):.4f}")
            logger.info(f"  SSIM:  {edit.get('ssim', {}).get('mean', 0):.4f} "
                       f"± {edit.get('ssim', {}).get('std', 0):.4f}")
            logger.info(f"  L1:    {edit.get('l1', {}).get('mean', 0):.4f} "
                       f"± {edit.get('l1', {}).get('std', 0):.4f}")
            
            # Total change (Original vs Counterfactual) - from combined evaluation
            if 'evaluation_combined' in results and 'image' in results['evaluation_combined']:
                total = results['evaluation_combined']['image'].get('aggregated', {})
                logger.info("Total Change (Original vs Counterfactual):")
                logger.info(f"  LPIPS: {total.get('lpips', {}).get('mean', 0):.4f} "
                           f"± {total.get('lpips', {}).get('std', 0):.4f}")
                logger.info(f"  SSIM:  {total.get('ssim', {}).get('mean', 0):.4f} "
                           f"± {total.get('ssim', {}).get('std', 0):.4f}")
                logger.info(f"  L1:    {total.get('l1', {}).get('mean', 0):.4f} "
                           f"± {total.get('l1', {}).get('std', 0):.4f}")
        
        # Log FRD (distribution-level) metrics
        if 'frd' in results and results['frd']:
            logger.info("\n" + "-" * 50)
            logger.info("FRD (Frechet Radiomic Distance) - Distribution Level")
            logger.info("-" * 50)
            for comparison, frd_value in results['frd'].items():
                if frd_value is not None:
                    logger.info(f"  {comparison}: {frd_value:.4f}")
                else:
                    logger.info(f"  {comparison}: N/A")
        
        logger.info("=" * 70)


def collate_fn(batch):
    """
    Custom collate function for text-based evaluation.
    
    Handles text prompts and labels as lists (not tensors).
    """
    from torch.utils.data._utils.collate import default_collate
    
    # Separate text prompts
    text_prompts = [item.pop('text_prompt') if 'text_prompt' in item else '' for item in batch]
    
    # Separate labels if present (dict objects can't be collated)
    labels = [item.pop('labels') if 'labels' in item else None for item in batch]
    has_labels = any(l is not None for l in labels)
    
    # Remove graph if present (not needed for text-based)
    for item in batch:
        if 'graph' in item:
            item.pop('graph')
    
    # Default collate for tensors
    collated = default_collate(batch)
    
    # Add text prompts back
    collated['text_prompt'] = text_prompts
    
    # Add labels back if any were present
    if has_labels:
        collated['labels'] = labels
    
    return collated


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate text-based counterfactual image generation (RadEdit baseline)"
    )
    
    # Method selection
    parser.add_argument("--method", type=str, default="text_ddpm",
                        choices=list(TEXT_GENERATOR_REGISTRY.keys()),
                        help="Counterfactual generation method. "
                             "text_ddim: DDIM inversion + text reconstruction. "
                             "text_ddpm: DDPM 'Edit Friendly' inversion (recommended). "
                             "Default: text_ddpm")
    
    # DDPM-specific arguments
    parser.add_argument("--skip", type=int, default=36,
                        help="Skip parameter for DDPM inversion (0-50). "
                             "Higher = more fidelity, less edit strength. "
                             "Only used with text_ddpm. Default: 36")
    
    # Template selection for ablation
    parser.add_argument("--template_set", type=str, default="default",
                        choices=AVAILABLE_TEMPLATE_SETS,
                        help="Template set for prompt generation. "
                             "Options: default (structured), freeform (simple), detailed (clinical). "
                             f"Available: {AVAILABLE_TEMPLATE_SETS}. Default: default")
    
    # Dataset arguments
    parser.add_argument("--csv_path", type=str,
                        default="/mnt/data/diffusion_graph/reports_processed.csv",
                        help="Path to reports CSV")
    parser.add_argument("--embeddings_path", type=str,
                        default="/mnt/data/diffusion_graph/new_embeddings_768.pkl",
                        help="Path to embeddings pickle")
    parser.add_argument("--image_root", type=str,
                        default="/mnt/data/CheXpert/PNG",
                        help="Root directory for images")
    parser.add_argument("--labels_path", type=str,
                        default=None,
                        help="Path to JSON file with ground truth pathology labels")
    
    # Evaluation arguments
    parser.add_argument("--target_pathology", type=str, default="Cardiomegaly",
                        help="Target pathology to evaluate")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to evaluate (None = all)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for generation")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    
    # Generation arguments
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=None,
                        help="Classifier-free guidance scale. "
                             "Default: 1.0 for ddim, 15.0 for ddpm")
    
    # Model arguments
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for computation")
    
    # Output arguments
    parser.add_argument("--save_images", action="store_true",
                        help="Save generated images")
    parser.add_argument("--output_dir", type=str, default="outputs/counterfactual_text",
                        help="Output directory for images and results")
    
    # Classifier arguments
    parser.add_argument("--classifier_models", type=str, nargs='+',
                        default=['densenet121-res224-all'],
                        help="Classifier model(s) for evaluation. "
                             "Single model or multiple for multi-classifier mode. "
                             "Options: densenet121-res224-all, jfhealthcare, resnet50-res512-all. "
                             "Example: --classifier_models densenet121-res224-all jfhealthcare")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Prepare generator kwargs
    generator_kwargs = {
        'device': args.device,
        'num_inference_steps': args.num_inference_steps,
        'template_set': args.template_set,
    }
    
    # Set method-specific defaults
    method = args.method.lower()
    if method in ['text_ddpm', 'ddpm', 'text_ddpm_fixed', 'ddpm_fixed']:
        generator_kwargs['skip'] = args.skip
        if args.guidance_scale is None:
            generator_kwargs['guidance_scale'] = 15.0
        else:
            generator_kwargs['guidance_scale'] = args.guidance_scale
        
        # Enable independent noise for fixed methods
        if method in FIXED_METHODS:
            generator_kwargs['use_independent_noise'] = True
            logger.info(f"Using DDPM_fixed method with independent noise, skip={args.skip}, guidance_scale={generator_kwargs['guidance_scale']}, template_set={args.template_set}")
        else:
            logger.info(f"Using DDPM method with skip={args.skip}, guidance_scale={generator_kwargs['guidance_scale']}, template_set={args.template_set}")
    else:  # text_ddim
        if args.guidance_scale is None:
            generator_kwargs['guidance_scale'] = 1.0
        else:
            generator_kwargs['guidance_scale'] = args.guidance_scale
        generator_kwargs['inversion_guidance_scale'] = 1.0
        logger.info(f"Using DDIM method with guidance_scale={generator_kwargs['guidance_scale']}, template_set={args.template_set}")
    
    # Initialize generator
    logger.info(f"Creating {args.method} generator...")
    generator_class = TEXT_GENERATOR_REGISTRY[method]
    generator = generator_class(**generator_kwargs)
    
    # Determine classifier model(s)
    classifier_models = args.classifier_models
    if len(classifier_models) == 1:
        classifier_models = classifier_models[0]  # Single model, not list
    logger.info(f"Using classifier model(s): {classifier_models}")
    
    # Initialize evaluator
    evaluator = TextCounterfactualEvaluator(
        generator=generator,
        device=args.device,
        classifier_models=classifier_models,
    )
    
    # Create dataset (no graphs needed)
    logger.info("Loading dataset...")
    if args.labels_path:
        logger.info(f"Using labels from: {args.labels_path}")
    dataset = CheXpertGraphDataset(
        csv_path=args.csv_path,
        embeddings_path=args.embeddings_path,
        image_root=args.image_root,
        split="valid",
        embedding_dim="original",
        load_graphs=False,  # Not needed for text-based
        labels_path=args.labels_path,
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    logger.info(f"Dataset size: {len(dataset)}, Batch size: {args.batch_size}")
    
    # Convert num_samples to num_batches
    num_batches = None
    if args.num_samples is not None:
        num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
        logger.info(f"Evaluating {args.num_samples} samples ({num_batches} batches)")
    
    # Run evaluation
    results = evaluator.evaluate(
        dataloader=dataloader,
        target_pathology=args.target_pathology,
        num_batches=num_batches,
        save_images=args.save_images,
        output_dir=args.output_dir,
    )
    
    # Add method info to results
    results['method'] = args.method
    results['template_set'] = args.template_set
    results['method_params'] = {
        'num_inference_steps': args.num_inference_steps,
        'guidance_scale': generator_kwargs['guidance_scale'],
        'template_set': args.template_set,
    }
    if method in ['text_ddpm', 'ddpm']:
        results['method_params']['skip'] = args.skip
    
    # Save results
    if args.output_dir:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        results_json = convert_numpy(results)
        
        with open(output_path / "results.json", 'w') as f:
            json.dump(results_json, f, indent=2)
        
        logger.info(f"Results saved to {output_path / 'results.json'}")


if __name__ == "__main__":
    main()

