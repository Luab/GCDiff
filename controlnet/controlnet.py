"""
ControlNet implementation for RadEdit with graph conditioning.

Uses diffusers' ControlNetModel.from_unet() to create a proper ControlNet
with zero spatial conditioning and graph embeddings via cross-attention.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, Dict, Any
from diffusers import UNet2DConditionModel
from diffusers.models.controlnets.controlnet import ControlNetModel


class GraphControlNet(nn.Module):
    """
    Wrapper around diffusers ControlNetModel for graph conditioning.
    
    Uses zero spatial conditioning and graph embeddings via cross-attention.
    This matches the baseline ControlNet architecture exactly, but without
    spatial image conditioning.
    
    Conditioning Strategy
    ---------------------
    Current Implementation:
        - ControlNet: Graph embeddings via encoder_hidden_states
        - UNet: Encoded empty prompt ("" through BioViL-T, not zeros)
        - Control signal comes from ControlNet residual connections
        - Empty prompt maintains proper text encoder output distribution
    
    Alternative Strategies (modify training/inference scripts):
    
    1. Text + Graph (Dual Modality):
       - ControlNet: Graph embeddings
       - UNet: Text embeddings from clinical prompts
       - Pros: Leverages RadEdit's text training, semantic + structural
       - Implementation: Encode batch['text_prompt'] and pass to UNet
    
    2. Dual Graph Conditioning:
       - ControlNet: Graph embeddings
       - UNet: Same graph embeddings
       - Pros: Both networks see graph structure
       - Cons: Out-of-distribution for UNet (trained on text)
       - Implementation: Pass graph_embeddings to both
    
    3. Concatenated Text + Graph:
       - Concat text and graph: [B, 256, 768]
       - Pass to UNet cross-attention
       - Cons: 2x sequence length, may need adjustments
    
    4. Learned Adapter:
       - Train adapter to merge text + graph → [B, 128, 768]
       - Pass merged embeddings to UNet
       - Pros: Optimal fusion, maintains sequence length
    
    5. Zero Embeddings (not recommended):
       - torch.zeros((B, 128, 768))
       - Simpler but out-of-distribution
       - Use encoded empty prompt instead (current)
    
    See README_controlnet.md for detailed comparison and usage examples.
    
    Embedding Expansion Strategies
    -------------------------------
    Controls how graph embeddings are expanded from [B, 768] to [B, 128, 768]:
    
    - "precomputed": Expects [B, 128, 768] precomputed embeddings (baseline)
    - "linear": Trainable nn.Linear(768, 128*768) expansion (~98M params)
    - "repeat": Simple repetition/stacking (no additional params)
    
    Args:
        unet: RadEdit UNet2DConditionModel to create ControlNet from
        embedding_strategy: Strategy for expanding embeddings ("precomputed", "linear", "repeat")
    """
    
    def __init__(self, unet: UNet2DConditionModel, embedding_strategy: str = "precomputed"):
        super().__init__()
        
        # Validate embedding strategy
        valid_strategies = ["precomputed", "linear", "repeat"]
        if embedding_strategy not in valid_strategies:
            raise ValueError(f"embedding_strategy must be one of {valid_strategies}, got {embedding_strategy}")
        
        self.embedding_strategy = embedding_strategy
        
        print(f"🔧 Initializing GraphControlNet from UNet (strategy={embedding_strategy})...")
        
        # Create ControlNet from UNet (copies encoder architecture + weights)
        self.controlnet = ControlNetModel.from_unet(
            unet=unet,
            conditioning_channels=3,  # Dummy spatial channels (will be zero)
            load_weights_from_unet=True,  # Initialize with UNet encoder weights
        )
        
        print(f"✅ ControlNet created from UNet")
        
        # Freeze the spatial conditioning embedding (we don't use it)
        # The zero-initialized conv_out will always output zeros for zero input
        for param in self.controlnet.controlnet_cond_embedding.parameters():
            param.requires_grad = False
        
        print(f"   Frozen spatial conditioning embedding")
        
        # Create expansion layer for "linear" strategy
        self.expansion_layer = None
        if self.embedding_strategy == "linear":
            self.expansion_layer = nn.Linear(768, 128 * 768)
            print(f"   Added trainable expansion layer: {sum(p.numel() for p in self.expansion_layer.parameters()):,} params")
        
        print(f"   Total ControlNet parameters: {sum(p.numel() for p in self.controlnet.parameters()):,}")
        print(f"   Trainable parameters: {sum(p.numel() for p in self.controlnet.parameters() if p.requires_grad):,}")
        if self.expansion_layer is not None:
            print(f"   Expansion layer parameters: {sum(p.numel() for p in self.expansion_layer.parameters()):,}")
    
    def forward(
        self,
        noisy_latents: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        graph_embeddings: torch.Tensor,
        conditioning_scale: float = 1.0,
        return_dict: bool = True,
    ):
        """
        Forward pass through ControlNet with graph conditioning.
        
        Args:
            noisy_latents: [B, 4, H, W] - Noisy latent images
            timestep: [B] or scalar - Diffusion timestep
            graph_embeddings: [B, 768] or [B, 128, 768] - Graph embeddings
                - For "linear" or "repeat": [B, 768]
                - For "precomputed": [B, 128, 768]
            conditioning_scale: Scale factor for ControlNet outputs
            return_dict: Whether to return dict or tuple
        
        Returns:
            ControlNetOutput or tuple:
                - down_block_res_samples: List of tensors from down blocks
                - mid_block_res_sample: Tensor from mid block
        """
        B = noisy_latents.shape[0]
        device = noisy_latents.device
        dtype = noisy_latents.dtype
        
        # Expand embeddings based on strategy
        if self.embedding_strategy == "precomputed":
            # Expect [B, 128, 768] - already expanded
            if graph_embeddings.dim() != 3 or graph_embeddings.shape[1:] != (128, 768):
                raise ValueError(f"precomputed strategy expects [B, 128, 768], got {graph_embeddings.shape}")
            expanded_embeddings = graph_embeddings
        
        elif self.embedding_strategy == "linear":
            # Expect [B, 768] - expand with trainable linear layer
            if graph_embeddings.dim() != 2 or graph_embeddings.shape[1] != 768:
                raise ValueError(f"linear strategy expects [B, 768], got {graph_embeddings.shape}")
            # [B, 768] -> [B, 128*768] -> [B, 128, 768]
            expanded = self.expansion_layer(graph_embeddings)
            expanded_embeddings = expanded.view(B, 128, 768)
        
        elif self.embedding_strategy == "repeat":
            # Expect [B, 768] - simple repetition
            if graph_embeddings.dim() != 2 or graph_embeddings.shape[1] != 768:
                raise ValueError(f"repeat strategy expects [B, 768], got {graph_embeddings.shape}")
            # [B, 768] -> [B, 1, 768] -> [B, 128, 768]
            expanded_embeddings = graph_embeddings.unsqueeze(1).repeat(1, 128, 1)
        
        # Create zero spatial conditioning
        # Since controlnet_cond_embedding.conv_out is zero-initialized,
        # zero input → zero output → no spatial bias
        controlnet_cond = torch.zeros(
            B, 3, 512, 512,
            device=device,
            dtype=dtype
        )
        
        # Forward pass with graph embeddings in cross-attention
        return self.controlnet(
            sample=noisy_latents,
            timestep=timestep,
            encoder_hidden_states=expanded_embeddings,  # Graph conditioning via cross-attn!
            controlnet_cond=controlnet_cond,  # Zero spatial conditioning
            conditioning_scale=conditioning_scale,
            return_dict=return_dict,
        )
    
    def get_trainable_parameters(self):
        """Return trainable parameters (ControlNet + expansion layer if present)."""
        params = [p for p in self.controlnet.parameters() if p.requires_grad]
        if self.expansion_layer is not None:
            params.extend([p for p in self.expansion_layer.parameters() if p.requires_grad])
        return params
    
    def get_num_trainable_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.get_trainable_parameters())


def test_graph_controlnet():
    """Test GraphControlNet initialization and forward pass."""
    import os
    
    print("=" * 80)
    print("Testing GraphControlNet")
    print("=" * 80)
    
    # Set proxy for model download
   # os.environ['HTTP_PROXY'] = "http://admin:Haruhi_123@192.168.1.1:1338"
    #os.environ['HTTPS_PROXY'] = "http://admin:Haruhi_123@192.168.1.1:1338"
    
    print("\nLoading RadEdit UNet...")
    unet = UNet2DConditionModel.from_pretrained(
        "microsoft/radedit",
        subfolder="unet",
    )
    
    print(f"UNet loaded:")
    print(f"  - Config: {unet.config}")
    print(f"  - Total parameters: {sum(p.numel() for p in unet.parameters()):,}")
    
    print("\nTesting all embedding strategies...")
    batch_size = 2
    noisy_latents = torch.randn(batch_size, 4, 64, 64)  # 512/8 = 64
    timestep = torch.tensor([500, 500])
    
    for strategy in ["precomputed", "linear", "repeat"]:
        print(f"\n--- Testing strategy: {strategy} ---")
        controlnet = GraphControlNet(unet, embedding_strategy=strategy)
        
        # Prepare embeddings based on strategy
        if strategy == "precomputed":
            graph_embeddings = torch.randn(batch_size, 128, 768)
        else:  # linear or repeat
            graph_embeddings = torch.randn(batch_size, 768)
        
        print(f"  Input shapes:")
        print(f"    - noisy_latents: {noisy_latents.shape}")
        print(f"    - timestep: {timestep.shape}")
        print(f"    - graph_embeddings: {graph_embeddings.shape}")
        
        with torch.no_grad():
            outputs = controlnet(
                noisy_latents=noisy_latents,
                timestep=timestep,
                graph_embeddings=graph_embeddings,
                conditioning_scale=1.0,
                return_dict=True,
            )
        
        print(f"  Outputs:")
        print(f"    - down_block_res_samples: {len(outputs.down_block_res_samples)} tensors")
        print(f"    - mid_block_res_sample: {outputs.mid_block_res_sample.shape}")
        
        # Check that outputs are initially near zero (due to zero initialization)
        mid_abs_mean = outputs.mid_block_res_sample.abs().mean().item()
        print(f"  Mid block abs mean: {mid_abs_mean:.6f}")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_graph_controlnet()
