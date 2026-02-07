"""
Counterfactual generation module.

Provides different methods for generating counterfactual medical images.

Graph-based generators (use ControlNet with graph embeddings):
1. FromScratchGenerator (alias: GraphCounterfactualGenerator)
   - Generates new images from random noise with edited graph conditioning
   - No structure preservation from original image
   - Fastest method

2. DDIMInversionGenerator
   - Uses deterministic DDIM inversion to map original image to noise
   - Reconstructs with edited graph conditioning via ControlNet
   - Preserves structure, but may accumulate inversion errors

3. DDPMInversionGenerator
   - Uses stochastic DDPM "edit-friendly" forward diffusion
   - Skip parameter controls fidelity vs edit strength
   - Best balance of structure preservation and edit flexibility
   - Based on "An Edit Friendly DDPM Noise Space" (CVPR 2024)

Text-based generators (use RadEdit native text conditioning, no ControlNet):
4. TextDDIMInversionGenerator
   - DDIM inversion with source text prompt
   - Reconstructs with template-based target text prompt
   - Baseline for comparison with graph-based methods

5. TextDDPMInversionGenerator
   - DDPM "edit-friendly" inversion
   - Reconstructs with template-based target text prompt
   - Best text-based baseline
"""

from .base import BaseCounterfactualGenerator
from .from_scratch import FromScratchGenerator, GraphCounterfactualGenerator
from .ddim_inversion import DDIMInversionGenerator
from .ddpm_inversion import DDPMInversionGenerator
from .text_inversion import TextDDIMInversionGenerator, TextDDPMInversionGenerator
from .hybrid_inversion import HybridDDPMGenerator

__all__ = [
    # Base
    'BaseCounterfactualGenerator',
    # Graph-based
    'FromScratchGenerator',
    'GraphCounterfactualGenerator',  # Backward compatibility alias
    'DDIMInversionGenerator',
    'DDPMInversionGenerator',
    # Text-based
    'TextDDIMInversionGenerator',
    'TextDDPMInversionGenerator',
    # Hybrid (graph + text)
    'HybridDDPMGenerator',
]

# Method name to class mapping for graph-based CLI
GENERATOR_REGISTRY = {
    'from_scratch': FromScratchGenerator,
    'ddim': DDIMInversionGenerator,
    'ddpm': DDPMInversionGenerator,
    # Aliases
    'ddim_inversion': DDIMInversionGenerator,
    'ddpm_inversion': DDPMInversionGenerator,
    'ddpm_edit': DDPMInversionGenerator,
    'our_inv': DDPMInversionGenerator,  # Match DDPM_inversion repo naming
    # Hybrid (graph + text)
    'hybrid': HybridDDPMGenerator,
    'hybrid_ddpm': HybridDDPMGenerator,
    # Fixed methods with independent noise sampling (per "Edit Friendly DDPM" paper Eq. 6)
    # These are handled specially in get_generator() to pass use_independent_noise=True
    'ddpm_fixed': DDPMInversionGenerator,
    'hybrid_fixed': HybridDDPMGenerator,
}

# Method name to class mapping for text-based CLI
TEXT_GENERATOR_REGISTRY = {
    'text_ddim': TextDDIMInversionGenerator,
    'text_ddpm': TextDDPMInversionGenerator,
    # Aliases without prefix
    'ddim': TextDDIMInversionGenerator,
    'ddpm': TextDDPMInversionGenerator,
    # Fixed methods with independent noise sampling (per "Edit Friendly DDPM" paper Eq. 6)
    # These are handled specially in get_text_generator() to pass use_independent_noise=True
    'text_ddpm_fixed': TextDDPMInversionGenerator,
    'ddpm_fixed': TextDDPMInversionGenerator,
}

# Methods that require use_independent_noise=True
FIXED_METHODS = {'ddpm_fixed', 'hybrid_fixed', 'text_ddpm_fixed'}


def get_generator(method: str, **kwargs) -> BaseCounterfactualGenerator:
    """
    Factory function to get a graph-based counterfactual generator by method name.
    
    Args:
        method: Generator method name ('from_scratch', 'ddim', 'ddpm', 'ddpm_fixed', 'hybrid_fixed')
        **kwargs: Arguments to pass to generator constructor
            - controlnet_path: Path to ControlNet checkpoint
            - embedder_path: Path to graph embedder pickle
            - device, num_inference_steps, guidance_scale, etc.
        
    Returns:
        Initialized generator instance
        
    Example:
        >>> generator = get_generator('ddpm', 
        ...     controlnet_path='checkpoints/controlnet.pth',
        ...     embedder_path='embedder.pkl',
        ...     skip=36,
        ... )
        >>> generator_fixed = get_generator('ddpm_fixed', ...)  # Uses independent noise
    """
    method_lower = method.lower()
    if method_lower not in GENERATOR_REGISTRY:
        available = list(set(GENERATOR_REGISTRY.keys()))
        raise ValueError(
            f"Unknown method: '{method}'. Available: {available}"
        )
    
    # Enable independent noise for *_fixed methods
    if method_lower in FIXED_METHODS:
        kwargs['use_independent_noise'] = True
    
    generator_class = GENERATOR_REGISTRY[method_lower]
    return generator_class(**kwargs)


def get_text_generator(method: str, **kwargs) -> BaseCounterfactualGenerator:
    """
    Factory function to get a text-based counterfactual generator by method name.
    
    Args:
        method: Generator method name ('text_ddim', 'text_ddpm', 'text_ddpm_fixed', 'ddim', 'ddpm', 'ddpm_fixed')
        **kwargs: Arguments to pass to generator constructor
            - device, num_inference_steps, guidance_scale, skip (for ddpm), etc.
        
    Returns:
        Initialized text generator instance
        
    Example:
        >>> generator = get_text_generator('text_ddpm', 
        ...     skip=36,
        ...     guidance_scale=15.0,
        ... )
        >>> generator_fixed = get_text_generator('text_ddpm_fixed', ...)  # Uses independent noise
    """
    method_lower = method.lower()
    if method_lower not in TEXT_GENERATOR_REGISTRY:
        available = list(set(TEXT_GENERATOR_REGISTRY.keys()))
        raise ValueError(
            f"Unknown text method: '{method}'. Available: {available}"
        )
    
    # Enable independent noise for *_fixed methods
    if method_lower in FIXED_METHODS:
        kwargs['use_independent_noise'] = True
    
    generator_class = TEXT_GENERATOR_REGISTRY[method_lower]
    return generator_class(**kwargs)
