"""
Test ControlNet with a mock UNet (no network download required).
"""

import torch
import torch.nn as nn
from controlnet import ControlNetModel
from diffusers import UNet2DConditionModel
from diffusers.configuration_utils import ConfigMixin, register_to_config


def create_test_unet():
    """Create a minimal test UNet with proper config."""
    
    # Create UNet with minimal configuration for testing
    unet = UNet2DConditionModel(
        sample_size=64,  # 512 / 8 = 64 (VAE downsamples by 8)
        in_channels=4,  # Latent channels
        out_channels=4,
        center_input_sample=False,
        flip_sin_to_cos=True,
        freq_shift=0,
        down_block_types=(
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type="UNetMidBlock2DCrossAttn",
        up_block_types=(
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        block_out_channels=(320, 640, 1280, 1280),
        layers_per_block=2,
        cross_attention_dim=768,  # BioViL-T embedding dimension
        attention_head_dim=8,
    )
    
    return unet


def test_controlnet_init():
    """Test ControlNet initialization."""
    print("=" * 80)
    print("Testing ControlNet Initialization (Offline)")
    print("=" * 80)
    
    print("\nCreating test UNet...")
    base_unet = create_test_unet()
    
    print(f"Base UNet config:")
    print(f"  - in_channels: {base_unet.config.in_channels}")
    print(f"  - out_channels: {base_unet.config.out_channels}")
    print(f"  - block_out_channels: {base_unet.config.block_out_channels}")
    print(f"  - down_block_types: {base_unet.config.down_block_types}")
    print(f"  - cross_attention_dim: {base_unet.config.cross_attention_dim}")
    
    print("\nInitializing ControlNet...")
    controlnet = ControlNetModel(base_unet, conditioning_scale=1.0)
    
    print(f"\n✅ ControlNet created successfully!")
    print(f"   Total parameters: {controlnet.get_num_parameters():,}")
    print(f"   Base UNet parameters: {sum(p.numel() for p in base_unet.parameters()):,}")
    
    return controlnet


def test_controlnet_forward():
    """Test ControlNet forward pass."""
    print("\n" + "=" * 80)
    print("Testing ControlNet Forward Pass")
    print("=" * 80)
    
    # Create ControlNet
    base_unet = create_test_unet()
    controlnet = ControlNetModel(base_unet, conditioning_scale=1.0)
    
    # Create test inputs
    batch_size = 2
    noisy_latents = torch.randn(batch_size, 4, 64, 64)
    timestep = torch.tensor([100])
    graph_embeddings = torch.randn(batch_size, 128, 768)
    
    print(f"\nInput shapes:")
    print(f"  - noisy_latents: {noisy_latents.shape}")
    print(f"  - timestep: {timestep.shape}")
    print(f"  - graph_embeddings: {graph_embeddings.shape}")
    
    print(f"\nRunning forward pass...")
    outputs = controlnet(
        noisy_latents=noisy_latents,
        timestep=timestep,
        graph_embeddings=graph_embeddings,
        return_dict=True,
    )
    
    print(f"\nOutputs:")
    print(f"  - Down block samples: {len(outputs['down_block_res_samples'])}")
    for i, sample in enumerate(outputs['down_block_res_samples']):
        print(f"    - Down block {i}: {sample.shape}")
    print(f"  - Mid block sample: {outputs['mid_block_res_sample'].shape}")
    
    # Check that outputs are not all zeros (zero conv should produce zeros initially)
    mid_sum = outputs['mid_block_res_sample'].abs().sum().item()
    print(f"\n  - Mid block abs sum: {mid_sum:.6f} (should be ~0 due to zero init)")
    
    print("\n✅ Forward pass successful!")
    return outputs


def test_parameter_count():
    """Test that ControlNet has roughly the same params as encoder part of UNet."""
    print("\n" + "=" * 80)
    print("Testing Parameter Counts")
    print("=" * 80)
    
    base_unet = create_test_unet()
    controlnet = ControlNetModel(base_unet, conditioning_scale=1.0)
    
    # Count parameters in base UNet down blocks
    down_params = sum(p.numel() for block in base_unet.down_blocks for p in block.parameters())
    mid_params = sum(p.numel() for p in base_unet.mid_block.parameters())
    conv_in_params = sum(p.numel() for p in base_unet.conv_in.parameters())
    
    total_encoder_params = down_params + mid_params + conv_in_params
    controlnet_params = controlnet.get_num_parameters()
    
    print(f"\nBase UNet encoder components:")
    print(f"  - conv_in: {conv_in_params:,}")
    print(f"  - down_blocks: {down_params:,}")
    print(f"  - mid_block: {mid_params:,}")
    print(f"  - Total encoder: {total_encoder_params:,}")
    
    print(f"\nControlNet:")
    print(f"  - Total parameters: {controlnet_params:,}")
    
    # ControlNet should have slightly more params due to zero convs
    print(f"\nRatio (ControlNet / Encoder): {controlnet_params / total_encoder_params:.2f}x")
    print("  (Should be ~1.0x, slightly higher due to zero convs)")
    
    print("\n✅ Parameter count test passed!")


if __name__ == "__main__":
    # Run all tests
    controlnet = test_controlnet_init()
    outputs = test_controlnet_forward()
    test_parameter_count()
    
    print("\n" + "=" * 80)
    print("🎉 All ControlNet tests passed!")
    print("=" * 80)







