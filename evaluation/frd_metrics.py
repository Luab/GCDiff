"""
FRD (Frechet Radiomic Distance) metrics for medical image distribution comparison.

FRD is a distribution-level metric that compares radiomic feature distributions
between two sets of images. Unlike LPIPS/SSIM (per-image-pair metrics), FRD measures
how similar the overall distribution of medical imaging features is between folders.

Based on: https://github.com/RichardObi/frd-score (v1)
Paper: https://arxiv.org/abs/2412.01496
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Add frd-score to path for imports
FRD_PATH = Path(__file__).parent.parent / "frd-score" / "frd_v1"
if FRD_PATH.exists():
    sys.path.insert(0, str(FRD_PATH))


class FRDMetrics:
    """
    Compute Frechet Radiomic Distance between image folders.
    
    FRD measures the similarity between two distributions of radiomic features
    extracted from medical images. Lower FRD indicates more similar distributions.
    
    Usage:
        frd_metrics = FRDMetrics()
        
        # Compute FRD between two folders
        frd_score = frd_metrics.compute_frd(
            folder1="/path/to/original",
            folder2="/path/to/generated"
        )
        
        # Compute FRD for multiple comparisons
        results = frd_metrics.compute_all_comparisons(
            original_folder="/path/to/original",
            comparison_folders={
                "increase": "/path/to/increase",
                "decrease": "/path/to/decrease",
            }
        )
    """
    
    def __init__(
        self,
        parallelize: bool = True,
        num_workers: int = 8,
        force_recompute: bool = False,
        radiomics_fname: str = "radiomics.csv",
    ):
        """
        Initialize FRD metrics.
        
        Args:
            parallelize: Whether to use parallel feature extraction
            num_workers: Number of workers for parallel extraction
            force_recompute: Whether to recompute radiomics even if cached
            radiomics_fname: Filename for cached radiomic features
        """
        self.parallelize = parallelize
        self.num_workers = num_workers
        self.force_recompute = force_recompute
        self.radiomics_fname = radiomics_fname
        
        # Verify FRD dependencies are available
        self._verify_dependencies()
    
    def _verify_dependencies(self):
        """Verify that FRD dependencies are installed."""
        try:
            from src.radiomics_utils import (
                compute_and_save_imagefolder_radiomics,
                compute_and_save_imagefolder_radiomics_parallel,
                convert_radiomic_dfs_to_vectors,
            )
            from src.utils import frechet_distance
            self._compute_radiomics = compute_and_save_imagefolder_radiomics
            self._compute_radiomics_parallel = compute_and_save_imagefolder_radiomics_parallel
            self._convert_to_vectors = convert_radiomic_dfs_to_vectors
            self._frechet_distance = frechet_distance
            logger.info("FRD dependencies loaded successfully from frd-score/frd_v1")
        except ImportError as e:
            logger.warning(
                f"FRD dependencies not available: {e}. "
                "Install with: pip install SimpleITK pyradiomics multiprocess scipy"
            )
            self._compute_radiomics = None
            self._compute_radiomics_parallel = None
            self._convert_to_vectors = None
            self._frechet_distance = None
    
    @property
    def is_available(self) -> bool:
        """Check if FRD computation is available."""
        return self._frechet_distance is not None
    
    def _extract_radiomics(self, image_folder: str) -> Optional[pd.DataFrame]:
        """
        Extract radiomic features from an image folder.
        
        Args:
            image_folder: Path to folder containing images
            
        Returns:
            DataFrame with radiomic features, or None if extraction fails
        """
        if not self.is_available:
            logger.error("FRD dependencies not available")
            return None
        
        radiomics_path = os.path.join(image_folder, self.radiomics_fname)
        
        # Check if radiomics are already computed
        if not self.force_recompute and os.path.exists(radiomics_path):
            logger.info(f"Loading cached radiomics from {radiomics_path}")
            return pd.read_csv(radiomics_path)
        
        # Compute radiomics
        logger.info(f"Computing radiomics for {image_folder}...")
        try:
            if self.parallelize:
                df = self._compute_radiomics_parallel(
                    image_folder,
                    radiomics_fname=self.radiomics_fname,
                    num_workers=self.num_workers,
                )
            else:
                df = self._compute_radiomics(
                    image_folder,
                    radiomics_fname=self.radiomics_fname,
                )
            logger.info(f"Computed radiomics for {len(df)} images")
            return df
        except Exception as e:
            logger.error(f"Failed to compute radiomics: {e}")
            return None
    
    def compute_frd(
        self,
        folder1: str,
        folder2: str,
        match_sample_count: bool = True,
    ) -> Optional[float]:
        """
        Compute FRD between two image folders.
        
        Args:
            folder1: Path to first image folder (reference)
            folder2: Path to second image folder (comparison)
            match_sample_count: Whether to match sample counts via random sampling
            
        Returns:
            FRD score (log of Frechet distance), or None if computation fails.
            Lower FRD indicates more similar distributions.
        """
        if not self.is_available:
            logger.error("FRD computation not available - missing dependencies")
            return None
        
        # Validate folders exist
        if not os.path.isdir(folder1):
            logger.error(f"Folder does not exist: {folder1}")
            return None
        if not os.path.isdir(folder2):
            logger.error(f"Folder does not exist: {folder2}")
            return None
        
        # Extract radiomics
        df1 = self._extract_radiomics(folder1)
        df2 = self._extract_radiomics(folder2)
        
        if df1 is None or df2 is None:
            logger.error("Failed to extract radiomics from one or both folders")
            return None
        
        if len(df1) == 0 or len(df2) == 0:
            logger.error("One or both folders have no valid images")
            return None
        
        # Convert to feature vectors
        try:
            feats1, feats2 = self._convert_to_vectors(
                df1, df2,
                match_sample_count=match_sample_count,
            )
        except Exception as e:
            logger.error(f"Failed to convert radiomics to vectors: {e}")
            return None
        
        # Compute Frechet distance
        try:
            fd = self._frechet_distance(feats1, feats2)
            frd = float(np.log(fd))
            logger.info(f"FRD({folder1} vs {folder2}) = {frd:.4f}")
            return frd
        except Exception as e:
            logger.error(f"Failed to compute Frechet distance: {e}")
            return None
    
    def compute_all_comparisons(
        self,
        original_folder: str,
        comparison_folders: Dict[str, str],
    ) -> Dict[str, Optional[float]]:
        """
        Compute FRD between original folder and multiple comparison folders.
        
        Args:
            original_folder: Path to original/reference images
            comparison_folders: Dict mapping names to folder paths
                e.g., {"increase": "/path/to/increase", "decrease": "/path/to/decrease"}
                
        Returns:
            Dictionary with FRD scores for each comparison
        """
        results = {}
        
        for name, folder in comparison_folders.items():
            if os.path.isdir(folder):
                frd = self.compute_frd(original_folder, folder)
                results[f"original_vs_{name}"] = frd
            else:
                logger.warning(f"Skipping {name}: folder does not exist at {folder}")
                results[f"original_vs_{name}"] = None
        
        return results


def compute_frd_for_output_folder(
    output_dir: str,
    force_recompute: bool = False,
    parallelize: bool = True,
) -> Dict[str, Optional[float]]:
    """
    Convenience function to compute FRD for a standard counterfactual output folder.
    
    Expects folder structure:
        output_dir/
            original/
            increase/
            decrease/
            reconstructed/  (optional)
    
    Args:
        output_dir: Path to output directory
        force_recompute: Whether to recompute radiomics even if cached
        parallelize: Whether to use parallel extraction
        
    Returns:
        Dictionary with FRD scores:
            - original_vs_increase
            - original_vs_decrease
            - original_vs_reconstructed (if exists)
    """
    output_path = Path(output_dir)
    
    original_folder = output_path / "original"
    if not original_folder.exists():
        logger.error(f"Original folder not found: {original_folder}")
        return {}
    
    # Build comparison folders dict
    comparison_folders = {}
    for name in ["increase", "decrease", "reconstructed"]:
        folder = output_path / name
        if folder.exists() and any(folder.iterdir()):  # Check folder exists and has files
            comparison_folders[name] = str(folder)
    
    if not comparison_folders:
        logger.warning("No comparison folders found")
        return {}
    
    # Compute FRD
    frd_metrics = FRDMetrics(
        force_recompute=force_recompute,
        parallelize=parallelize,
    )
    
    return frd_metrics.compute_all_comparisons(
        original_folder=str(original_folder),
        comparison_folders=comparison_folders,
    )

