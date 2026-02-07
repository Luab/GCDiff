"""
Dissect and showcase diffusion guidance mechanics - RadEdit Model.

This script breaks down the RadEdit diffusion process to show:
1. Explicit prompt embedding generation with BioViL-T and shapes
2. Conditioning vector structure
3. Classifier-free guidance step-by-step
4. Manual denoising loop with detailed logging
5. Intermediate results visualization
"""

import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)
from transformers import AutoModel, AutoTokenizer
from PIL import Image
from tqdm.auto import tqdm
from typing import Optional

# Set HTTP proxy for downloading models
os.environ['HTTP_PROXY'] = "http://admin:Haruhi_123@192.168.1.1:1338"
os.environ['HTTPS_PROXY'] = "http://admin:Haruhi_123@192.168.1.1:1338"


class RadEditGuidanceDissector:
    """Dissect and visualize RadEdit diffusion guidance mechanics."""
    
    def __init__(
        self,
        device: str = "cpu" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        print(f"🔧 Loading RadEdit model components...")
        self.device = device
        self.dtype = dtype
        
        # Load UNet from RadEdit
        print("  - Loading UNet from microsoft/radedit...")
        self.unet = UNet2DConditionModel.from_pretrained(
            "microsoft/radedit", 
            subfolder="unet",
            torch_dtype=dtype,
        )
        self.unet = self.unet.to(device)
        
        # Load SDXL VAE
        print("  - Loading SDXL-VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sdxl-vae",
            torch_dtype=dtype,
        )
        self.vae = self.vae.to(device)
        
        # Load BioViL-T text encoder and tokenizer
        print("  - Loading BioViL-T text encoder...")
        self.text_encoder = AutoModel.from_pretrained(
            "microsoft/BiomedVLP-BioViL-T",
            trust_remote_code=True,
        )
        self.text_encoder = self.text_encoder.to(device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/BiomedVLP-BioViL-T",
            model_max_length=128,
            trust_remote_code=True,
        )
        
        # Setup scheduler
        self.scheduler = DDIMScheduler(
            beta_schedule="linear",
            clip_sample=False,
            prediction_type="epsilon",
            timestep_spacing="trailing",
            steps_offset=1,
        )
        
        print("✅ RadEdit model loaded\n")
        print("Model Architecture:")
        print("  - Text Encoder: BioViL-T (medical domain-specific)")
        print("  - VAE: SDXL-VAE (high quality)")
        print("  - UNet: RadEdit custom (trained on chest X-rays)")
        print()
    
    def dissect_text_encoding(self, prompt: str, verbose: bool = True):
        """
        Dissect the text encoding process with BioViL-T and show all intermediate shapes.
        
        Returns:
            embeddings: Final text embeddings [batch_size, seq_len, hidden_dim]
        """
        if verbose:
            print("=" * 80)
            print("📝 TEXT ENCODING DISSECTION (BioViL-T)")
            print("=" * 80)
            print(f"Input prompt: '{prompt}'")
            print()
        
        # Step 1: Tokenization
        text_inputs = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device)
        
        if verbose:
            print("Step 1: Tokenization (WordPiece)")
            print(f"  input_ids shape: {input_ids.shape}")
            print(f"  attention_mask shape: {attention_mask.shape}")
            print(f"  Max sequence length: {self.tokenizer.model_max_length}")
            print(f"  Number of actual tokens: {attention_mask.sum().item()}")
            print(f"  Token IDs: {input_ids[0][:15].tolist()}... (first 15)")
            
            # Decode tokens
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            print(f"  Decoded tokens (first 15): {tokens[:15]}")
            print(f"  Special tokens: [CLS]=101, [SEP]=102, [PAD]=0")
            print()
        
        # Step 2: Text encoding with BioViL-T
        with torch.no_grad():
            encoder_output = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            # Extract last_hidden_state from BioViL-T output
            if hasattr(encoder_output, 'last_hidden_state'):
                embeddings = encoder_output.last_hidden_state
            elif isinstance(encoder_output, tuple):
                embeddings = encoder_output[0]
            else:
                embeddings = encoder_output
        
        if verbose:
            print("Step 2: Text Encoding (BioViL-T / CXR-BERT)")
            print(f"  Model: BiomedVLP-BioViL-T")
            print(f"  Architecture: BERT-based, trained on MIMIC-CXR reports")
            print(f"  Output embeddings shape: {embeddings.shape}")
            print(f"    [batch_size, sequence_length, hidden_dimension]")
            print(f"    [{embeddings.shape[0]}, {embeddings.shape[1]}, {embeddings.shape[2]}]")
            print(f"  Embedding statistics:")
            print(f"    Mean: {embeddings.mean().item():.6f}")
            print(f"    Std:  {embeddings.std().item():.6f}")
            print(f"    Min:  {embeddings.min().item():.6f}")
            print(f"    Max:  {embeddings.max().item():.6f}")
            
            # Analyze per-position statistics
            token_norms = embeddings.norm(dim=-1)[0]
            print(f"  Per-token L2 norms:")
            print(f"    [CLS] token: {token_norms[0].item():.4f}")
            if attention_mask.sum() > 2:
                print(f"    First content token: {token_norms[1].item():.4f}")
            print(f"    Mean across sequence: {token_norms.mean().item():.4f}")
            print()
        
        return embeddings
    
    def dissect_classifier_free_guidance(
        self,
        conditional_prompt: str,
        guidance_scale: float = 7.5,
    ):
        """
        Dissect classifier-free guidance setup for RadEdit.
        
        Shows how conditional and unconditional embeddings are combined.
        """
        print("=" * 80)
        print("🎯 CLASSIFIER-FREE GUIDANCE DISSECTION (RadEdit)")
        print("=" * 80)
        print(f"Guidance scale: {guidance_scale}")
        print(f"Text encoder: BioViL-T (medical domain-specific)")
        print()
        
        # Get conditional embeddings
        print("Conditional embeddings (with clinical prompt):")
        cond_embeds = self.dissect_text_encoding(conditional_prompt, verbose=True)
        
        print("-" * 80)
        
        # Get unconditional embeddings
        print("Unconditional embeddings (empty prompt):")
        uncond_embeds = self.dissect_text_encoding("", verbose=True)
        
        print("-" * 80)
        print("\nStep 3: Concatenation for CFG")
        combined_embeds = torch.cat([uncond_embeds, cond_embeds])
        print(f"  Combined shape: {combined_embeds.shape}")
        print(f"    [batch_size * 2, sequence_length, hidden_dimension]")
        print(f"    [{combined_embeds.shape[0]}, {combined_embeds.shape[1]}, {combined_embeds.shape[2]}]")
        print(f"  First half  -> unconditional (empty prompt)")
        print(f"  Second half -> conditional (clinical finding)")
        print()
        
        # Compare embeddings
        embedding_diff = (cond_embeds - uncond_embeds).abs().mean().item()
        cosine_sim = torch.nn.functional.cosine_similarity(
            cond_embeds.flatten(), uncond_embeds.flatten(), dim=0
        ).item()
        print(f"  Mean absolute difference: {embedding_diff:.6f}")
        print(f"  Cosine similarity: {cosine_sim:.6f}")
        print()
        
        print("Step 4: Classifier-Free Guidance Formula")
        print(f"  noise_pred = uncond_pred + {guidance_scale} * (cond_pred - uncond_pred)")
        print()
        print(f"  Interpretation:")
        print(f"    • guidance_scale = 1.0: Pure conditional (just use the prompt)")
        print(f"    • guidance_scale > 1.0: Amplify prompt influence (more faithful)")
        print(f"    • guidance_scale = 7.5: RadEdit default (strong guidance)")
        print(f"    • guidance_scale = 0.0: Ignore prompt entirely")
        print()
        
        return cond_embeds, uncond_embeds, combined_embeds
    
    def dissect_denoising_step(
        self,
        latents: torch.Tensor,
        timestep: int,
        text_embeddings: torch.Tensor,
        guidance_scale: float = 7.5,
        step_num: int = 0,
        verbose: bool = True,
    ):
        """
        Dissect a single denoising step in detail.
        
        Shows latent shapes, UNet inputs/outputs, and guidance application.
        """
        if verbose:
            print("=" * 80)
            print(f"🔄 DENOISING STEP #{step_num} DISSECTION (t={timestep})")
            print("=" * 80)
        
        # Step 1: Prepare latents for CFG
        latent_model_input = torch.cat([latents] * 2)
        
        if verbose:
            print("Step 1: Prepare UNet inputs")
            print(f"  Original latents shape: {latents.shape}")
            print(f"    [batch, channels, height, width]")
            print(f"    [{latents.shape[0]}, {latents.shape[1]}, {latents.shape[2]}, {latents.shape[3]}]")
            print(f"  Duplicated for CFG: {latent_model_input.shape}")
            print(f"  Timestep: {timestep}")
            print(f"  Text embeddings: {text_embeddings.shape}")
            print()
        
        # Step 2: UNet forward pass
        with torch.no_grad():
            noise_pred = self.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=text_embeddings,
            ).sample
        
        if verbose:
            print("Step 2: UNet forward pass")
            print(f"  Input shapes:")
            print(f"    latents: {latent_model_input.shape}")
            print(f"    timestep: scalar ({timestep})")
            print(f"    encoder_hidden_states: {text_embeddings.shape}")
            print(f"  Output noise prediction shape: {noise_pred.shape}")
            print(f"  Noise statistics:")
            print(f"    Mean: {noise_pred.mean().item():.6f}")
            print(f"    Std:  {noise_pred.std().item():.6f}")
            print()
        
        # Step 3: Split predictions
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        
        if verbose:
            print("Step 3: Split predictions")
            print(f"  Unconditional noise: {noise_pred_uncond.shape}")
            print(f"    Mean: {noise_pred_uncond.mean().item():.6f}")
            print(f"  Conditional noise: {noise_pred_text.shape}")
            print(f"    Mean: {noise_pred_text.mean().item():.6f}")
            print(f"  Difference magnitude: {(noise_pred_text - noise_pred_uncond).abs().mean().item():.6f}")
            print()
        
        # Step 4: Apply guidance
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        
        if verbose:
            print("Step 4: Apply classifier-free guidance")
            print(f"  Formula: uncond + {guidance_scale} * (cond - uncond)")
            print(f"  Guided noise shape: {noise_pred.shape}")
            print(f"  Guided noise statistics:")
            print(f"    Mean: {noise_pred.mean().item():.6f}")
            print(f"    Std:  {noise_pred.std().item():.6f}")
            print()
        
        # Step 5: Scheduler step
        output = self.scheduler.step(noise_pred, timestep, latents)
        latents = output.prev_sample
        
        if verbose:
            print("Step 5: DDIM scheduler step")
            print(f"  Input noisy latents: {latents.shape}")
            print(f"  Output denoised latents: {latents.shape}")
            print(f"  Latent statistics:")
            print(f"    Mean: {latents.mean().item():.6f}")
            print(f"    Std:  {latents.std().item():.6f}")
            print()
        
        return latents, noise_pred
    
    def generate_with_dissection(
        self,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: int = 42,
        show_every_n_steps: int = 10,
    ):
        """
        Generate image with full dissection of the process.
        
        Shows detailed information at key steps and saves intermediate results.
        """
        print("=" * 80)
        print("🚀 GENERATION WITH FULL DISSECTION")
        print("=" * 80)
        print(f"Prompt: '{prompt}'")
        print(f"Steps: {num_inference_steps}")
        print(f"Guidance: {guidance_scale}")
        print(f"Seed: {seed}")
        print()
        
        # Step 1: Prepare embeddings
        cond_embeds, uncond_embeds, combined_embeds = self.dissect_classifier_free_guidance(
            prompt, guidance_scale
        )
        
        # Step 2: Initialize latents
        print("=" * 80)
        print("🎲 LATENT INITIALIZATION (SDXL-VAE)")
        print("=" * 80)
        
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Get latent dimensions from VAE
        latent_channels = self.vae.config.latent_channels
        height = width = 512 // 8  # VAE downsamples by 8x
        
        latents = torch.randn(
            (1, latent_channels, height, width),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        
        print(f"  VAE: SDXL-VAE (stabilityai/sdxl-vae)")
        print(f"  Random latents shape: {latents.shape}")
        print(f"    [batch, channels, height, width]")
        print(f"    [{latents.shape[0]}, {latents.shape[1]}, {latents.shape[2]}, {latents.shape[3]}]")
        print(f"  Latent space: {latent_channels} channels, {height}x{width} spatial resolution")
        print(f"  Image space: {height*8}x{width*8} pixels (8x upsampling by SDXL-VAE)")
        print(f"  Note: SDXL-VAE provides better reconstruction quality than SD1.5 VAE")
        print(f"  Initial noise statistics:")
        print(f"    Mean: {latents.mean().item():.6f}")
        print(f"    Std:  {latents.std().item():.6f}")
        print()
        
        # Step 3: Setup scheduler
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        
        # Scale initial noise
        latents = latents * self.scheduler.init_noise_sigma
        
        print(f"  Scaled by scheduler (init_noise_sigma={self.scheduler.init_noise_sigma:.4f})")
        print(f"    Mean: {latents.mean().item():.6f}")
        print(f"    Std:  {latents.std().item():.6f}")
        print()
        
        # Step 4: Denoising loop
        print("=" * 80)
        print("🔄 DENOISING LOOP")
        print("=" * 80)
        print(f"Total steps: {num_inference_steps}")
        print(f"Showing details every {show_every_n_steps} steps")
        print()
        
        intermediate_latents = []
        
        for i, t in enumerate(tqdm(timesteps, desc="Denoising")):
            # Show detailed dissection for selected steps
            verbose = (i % show_every_n_steps == 0) or (i == len(timesteps) - 1)
            
            latents, noise_pred = self.dissect_denoising_step(
                latents=latents,
                timestep=t,
                text_embeddings=combined_embeds,
                guidance_scale=guidance_scale,
                step_num=i,
                verbose=verbose,
            )
            
            # Store intermediate results
            if i % show_every_n_steps == 0:
                intermediate_latents.append((i, latents.clone()))
        
        # Step 5: Decode to image
        print("=" * 80)
        print("🎨 SDXL-VAE DECODING")
        print("=" * 80)
        
        latents = latents / self.vae.config.scaling_factor
        print(f"  Input latents shape: {latents.shape}")
        print(f"  Unscaled by VAE scaling factor ({self.vae.config.scaling_factor})")
        print(f"  VAE decoder: SDXL-VAE (improved quality over SD1.5)")
        print()
        
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        
        print(f"  Decoded image shape: {image.shape}")
        print(f"    [batch, channels, height, width]")
        print(f"    [{image.shape[0]}, {image.shape[1]}, {image.shape[2]}, {image.shape[3]}]")
        print(f"  Output: RGB chest X-ray (3 channels)")
        print(f"  Image value range: [-1, 1] (before normalization)")
        print(f"  Image statistics:")
        print(f"    Mean: {image.mean().item():.6f}")
        print(f"    Std:  {image.std().item():.6f}")
        print(f"    Min:  {image.min().item():.6f}")
        print(f"    Max:  {image.max().item():.6f}")
        print()
        
        # Convert to PIL
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).astype(np.uint8)[0]
        pil_image = Image.fromarray(image)
        
        print("=" * 80)
        print("✅ GENERATION COMPLETE")
        print("=" * 80)
        
        return pil_image, intermediate_latents
    
    def visualize_intermediate_steps(
        self,
        intermediate_latents,
        save_path: str = "diffusion_progression.png",
    ):
        """Visualize intermediate denoising steps."""
        print(f"\n📊 Visualizing {len(intermediate_latents)} intermediate steps...")
        
        n_steps = len(intermediate_latents)
        fig, axes = plt.subplots(2, (n_steps + 1) // 2, figsize=(4 * ((n_steps + 1) // 2), 8))
        axes = axes.flatten()
        
        for idx, (step_num, latent) in enumerate(intermediate_latents):
            # Decode latent
            with torch.no_grad():
                latent_scaled = latent / self.vae.config.scaling_factor
                img = self.vae.decode(latent_scaled).sample
            
            # Convert to displayable format
            img = (img / 2 + 0.5).clamp(0, 1)
            img = img.cpu().squeeze().numpy()
            if len(img.shape) == 3:
                img = img.transpose(1, 2, 0)
            
            axes[idx].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
            axes[idx].set_title(f"Step {step_num}", fontsize=10)
            axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(len(intermediate_latents), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', format='png')
        print(f"💾 Saved progression to {save_path}")
        plt.close(fig)


def main():
    """Run dissection demo on RadEdit."""
    
    print("=" * 80)
    print("🔬 RadEdit Diffusion Guidance Dissection Demo")
    print("=" * 80)
    print()
    
    # Initialize dissector
    dissector = RadEditGuidanceDissector(
        device="cpu" if torch.cuda.is_available() else "cpu",
    )
    
    # Test prompts - RadEdit uses clinical impression style
    test_prompts = [
        "No acute cardiopulmonary process",
        "Small right-sided pleural effusion",
        "Cardiomegaly",
    ]
    
    # Use first prompt
    prompt = test_prompts[0]
    
    print("=" * 80)
    print(f"🎬 Generating chest X-ray with dissection")
    print(f"   Prompt: '{prompt}'")
    print("=" * 80)
    print()
    
    # Generate with full dissection
    image, intermediate_latents = dissector.generate_with_dissection(
        prompt=prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        seed=42,
        show_every_n_steps=10,  # Show details every 10 steps
    )
    
    # Save final image
    image.save("radedit_dissection_final.png")
    print(f"\n💾 Saved final image to radedit_dissection_final.png")
    
    # Visualize progression
    dissector.visualize_intermediate_steps(
        intermediate_latents,
        save_path="radedit_diffusion_progression.png",
    )
    
    print("\n" + "=" * 80)
    print("✅ RadEdit Dissection Complete!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - radedit_dissection_final.png: Final generated chest X-ray")
    print("  - radedit_diffusion_progression.png: Denoising progression grid")
    print()
    print("Key insights from RadEdit:")
    print("  ✓ BioViL-T uses medical domain-specific BERT tokenization")
    print("  ✓ Embeddings are [batch, 128, 768] (longer context than CLIP)")
    print("  ✓ SDXL-VAE provides higher quality image reconstruction")
    print("  ✓ UNet trained specifically on chest X-ray datasets")


if __name__ == "__main__":
    main()

