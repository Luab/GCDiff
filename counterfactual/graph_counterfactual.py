"""
Backward compatibility module.

The original GraphCounterfactualGenerator has been refactored to:
- FromScratchGenerator: Generates new images from random noise (this file's original behavior)
- DDIMInversionGenerator: Uses DDIM inversion for structure preservation
- DDPMInversionGenerator: Uses DDPM "Edit Friendly" inversion (recommended)

This module re-exports GraphCounterfactualGenerator as an alias to FromScratchGenerator.
"""

# Re-export for backward compatibility
from .from_scratch import FromScratchGenerator, GraphCounterfactualGenerator

__all__ = ['GraphCounterfactualGenerator', 'FromScratchGenerator']
