"""
Abstract base class for counterfactual generators.

Defines the interface that all counterfactual generation methods must implement,
enabling easy swapping of generation strategies in the evaluation pipeline.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from PIL import Image


class BaseCounterfactualGenerator(ABC):
    """
    Abstract base class for counterfactual image generators.
    
    All counterfactual generators must implement the `generate_batch` method
    which takes a batch from a dataloader and produces counterfactual images.
    
    Example implementations:
    - GraphCounterfactualGenerator: Edits graphs and re-encodes
    - TextCounterfactualGenerator: Modifies text prompts
    - LatentCounterfactualGenerator: Manipulates latent space directly
    """
    
    @abstractmethod
    def generate_batch(
        self,
        batch: Dict[str, Any],
        target_pathology: str,
    ) -> Dict[str, Any]:
        """
        Generate counterfactual images for a batch.
        
        Args:
            batch: Dictionary from dataloader containing:
                - 'image': torch.Tensor [B, 3, H, W] - Original images
                - 'graph_embedding': torch.Tensor [B, 768] or [B, 128, 768]
                - 'graph': List[nx.DiGraph] (if load_graphs=True)
                - 'text_prompt': List[str]
                - 'original_idx': List[int]
            target_pathology: Target pathology to modify (e.g., 'Cardiomegaly')
            
        Returns:
            Dictionary containing:
                - 'original_images': List[PIL.Image] - Original images as PIL
                - 'counterfactual_images': List[PIL.Image] - Generated counterfactuals
                - 'directions': List[str] - "increase" or "decrease" per sample
                - 'metadata': Dict - Additional info (e.g., embedding similarity)
        """
        pass
    
    @property
    @abstractmethod
    def supported_pathologies(self) -> List[str]:
        """Return list of pathologies this generator can target."""
        pass
    
    def validate_batch(self, batch: Dict[str, Any]) -> None:
        """
        Validate that batch contains required keys.
        
        Args:
            batch: Batch dictionary from dataloader
            
        Raises:
            ValueError: If required keys are missing
        """
        required_keys = ['image']
        for key in required_keys:
            if key not in batch:
                raise ValueError(f"Batch missing required key: '{key}'")
    
    def validate_pathology(self, pathology: str) -> None:
        """
        Validate that pathology is supported.
        
        Args:
            pathology: Target pathology name
            
        Raises:
            ValueError: If pathology is not supported
        """
        if pathology not in self.supported_pathologies:
            raise ValueError(
                f"Unsupported pathology: '{pathology}'. "
                f"Supported: {self.supported_pathologies}"
            )

