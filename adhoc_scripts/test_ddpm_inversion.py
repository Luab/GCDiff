"""
Quick test for DDPM inversion on Microsoft RadEdit chest X-ray model.

Protocol:
1. Generate synthetic CXR with initial prompt
2. Run DDPM inversion (iterative optimization)
3. Reconstruct with different target prompt
4. Visualize comparison grid with pixelwise difference

Note: DDPM inversion is more challenging than DDIM due to stochasticity.
We use an iterative optimization approach to find the noise that reconstructs the image.
"""

import os
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving
import matplotlib.pyplot as plt
import numpy as np
from diffusers import (
    AutoencoderKL, 
    DDPMScheduler, 
    StableDiffusionPipeline, 
    UNet2DConditionModel
)
from transformers import AutoModel, AutoTokenizer
from PIL import Image
from tqdm.auto import tqdm

# Set HTTP proxy for downloading models
os.environ['HTTP_PROXY'] = "http://admin:Haruhi_123@192.168.1.1:1338"
os.environ['HTTPS_PROXY'] = "http://admin:Haruhi_123@192.168.1.1:1338"


class RadEditDDPMInversionTester:
    """Quick tester for DDPM inversion on Microsoft RadEdit model."""
    
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
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
        
        # Load SDXL VAE
        print("  - Loading SDXL-VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sdxl-vae",
            torch_dtype=dtype,
        )
        
        # Load BioViL-T text encoder and tokenizer
        print("  - Loading BioViL-T text encoder...")
        self.text_encoder = AutoModel.from_pretrained(
            "microsoft/BiomedVLP-BioViL-T",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/BiomedVLP-BioViL-T",
            model_max_length=128,
            trust_remote_code=True,
        )
        
        # Setup DDPM scheduler
        self.scheduler = DDPMScheduler(
            beta_schedule="linear",
            clip_sample=False,
            prediction_type="epsilon",
            timestep_spacing="trailing",
            steps_offset=1,
        )
        
        # Create generation pipeline
        self.pipe = StableDiffusionPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.scheduler,
            safety_checker=None,
            requires_safety_checker=False,
            feature_extractor=None,
        )
        self.pipe = self.pipe.to(device)
        
        print("✅ RadEdit model loaded successfully")
    
    def generate_image(self, prompt: str, num_inference_steps: int = 50, seed: int = 42):
        """Generate synthetic chest X-ray using DDPM sampling."""
        print(f"🖼️  Generating image: '{prompt}'")
        
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        with torch.no_grad():
            image = self.pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=7.5,
                generator=generator,
            ).images[0]
        
        return image
    
    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode PIL image to latent space."""
        if isinstance(image, Image.Image):
            # Convert PIL to tensor
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np).unsqueeze(0)
            
            # Handle grayscale
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(-1)
            
            # Permute to (B, C, H, W) and convert to RGB if needed
            image_tensor = image_tensor.permute(0, 3, 1, 2)
            if image_tensor.shape[1] == 1:
                image_tensor = image_tensor.repeat(1, 3, 1, 1)
            
            # Normalize to [-1, 1]
            image_tensor = (image_tensor - 0.5) * 2
        else:
            image_tensor = image
        
        image_tensor = image_tensor.to(self.device, dtype=self.dtype)
        
        with torch.no_grad():
            latents = self.vae.encode(image_tensor).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        
        return latents
    
    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to image space."""
        latents = latents / self.vae.config.scaling_factor
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        return image
    
    def _get_text_embeddings(self, prompt: str) -> torch.Tensor:
        """Get text embeddings from BioViL-T."""
        text_inputs = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        with torch.no_grad():
            # BioViL-T requires attention_mask
            embeddings = self.text_encoder(
                input_ids=text_inputs.input_ids.to(self.device),
                attention_mask=text_inputs.attention_mask.to(self.device),
            )
            # Extract the text embeddings (last_hidden_state)
            if hasattr(embeddings, 'last_hidden_state'):
                embeddings = embeddings.last_hidden_state
            elif isinstance(embeddings, tuple):
                embeddings = embeddings[0]
        
        return embeddings
    
    def ddpm_inversion(
        self,
        image: Image.Image,
        prompt: str,
        num_inference_steps: int = 50,
    ):
        """Perform DDPM inversion using forward diffusion process.
        
        Uses the forward (noising) process of DDPM to add noise step by step.
        This follows: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        
        Args:
            image: Input image to invert
            prompt: Text prompt for inversion (unused in forward process, kept for API consistency)
            num_inference_steps: Number of diffusion steps
        """
        print(f"🔄 Running DDPM inversion (forward diffusion)...")
        print(f"   Diffusion steps: {num_inference_steps}")
        
        # Encode image to latents
        latents = self._encode_image(image)
        
        # Setup scheduler
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = reversed(self.scheduler.timesteps)  # Reverse for forward process
        
        # Get noise schedule parameters
        alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        
        # Sample random noise
        noise = torch.randn_like(latents, device=self.device, dtype=self.dtype)
        
        trajectory = [latents.clone()]
        
        # Forward diffusion: progressively add noise
        # Using the closed-form formula: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        for i, t in enumerate(tqdm(list(timesteps), desc="Adding noise")):
            # Get alpha_bar for this timestep
            alpha_bar_t = alphas_cumprod[t]
            
            # Apply forward diffusion
            with torch.no_grad():
                noisy_latents = (
                    torch.sqrt(alpha_bar_t) * latents + 
                    torch.sqrt(1 - alpha_bar_t) * noise
                )
            
            trajectory.append(noisy_latents.clone())
        
        # Return the fully noised latents
        final_noise = trajectory[-1]
        
        print(f"✅ Inversion complete. Final noise std: {final_noise.std().item():.4f}")
        return final_noise, trajectory
    
    def reconstruct_from_noise(
        self,
        noise: torch.Tensor,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
    ):
        """Reconstruct image from inverted noise using target prompt."""
        print(f"🎨 Reconstructing with prompt: '{prompt}'")
        
        # Setup scheduler
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        
        # Get text embeddings
        text_embeds = self._get_text_embeddings(prompt)
        uncond_embeds = self._get_text_embeddings("")
        
        latents = noise.clone()
        
        for t in tqdm(timesteps, desc="Denoising"):
            # Expand latents for classifier-free guidance
            latent_input = torch.cat([latents] * 2)
            text_input = torch.cat([uncond_embeds, text_embeds])
            
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_input, t, encoder_hidden_states=text_input
                ).sample
            
            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            
            # DDPM step
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decode to image
        image = self._decode_latents(latents)
        return image
    
    def run_test(
        self,
        source_prompt: str,
        target_prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: int = 42,
    ):
        """
        Full test protocol:
        1. Generate image with source prompt
        2. Invert it using forward DDPM diffusion
        3. Reconstruct with same prompt (test fidelity)
        4. Reconstruct with target prompt (test editing)
        
        Args:
            source_prompt: Initial prompt for generation
            target_prompt: Target prompt for editing
            num_inference_steps: Number of DDPM steps
            guidance_scale: Guidance scale for reconstruction
            seed: Random seed
        """
        print("=" * 80)
        print("🧪 Starting DDPM Inversion Test (RadEdit)")
        print(f"   Source: '{source_prompt}'")
        print(f"   Target: '{target_prompt}'")
        print(f"   Reconstruction guidance: {guidance_scale}")
        print("=" * 80)
        
        # Step 1: Generate original
        original_image = self.generate_image(
            source_prompt, 
            num_inference_steps=num_inference_steps,
            seed=seed
        )
        
        # Step 2: Invert using forward diffusion
        noise, trajectory = self.ddpm_inversion(
            original_image,
            source_prompt,
            num_inference_steps=num_inference_steps,
        )
        
        # Step 3: Reconstruct with same prompt (fidelity test)
        reconstructed = self.reconstruct_from_noise(
            noise,
            source_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        
        # Step 4: Reconstruct with target prompt (editing test)
        edited = self.reconstruct_from_noise(
            noise,
            target_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        
        # Visualize
        self.visualize_results(
            original_image,
            reconstructed,
            edited,
            source_prompt,
            target_prompt,
        )
        
        return original_image, reconstructed, edited, trajectory
    
    def visualize_results(
        self,
        original,
        reconstructed,
        edited,
        source_prompt: str,
        target_prompt: str,
        save_path: str = "ddpm_inversion_results.png",
    ):
        """Visualize comparison grid with pixelwise difference."""
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        
        def tensor_to_np(tensor):
            """Convert tensor to numpy for display."""
            if isinstance(tensor, Image.Image):
                return np.array(tensor)
            
            img = tensor.squeeze().cpu().numpy()
            if len(img.shape) == 3:  # (C, H, W)
                img = img.transpose(1, 2, 0)  # (H, W, C)
            # Denormalize from [-1, 1] to [0, 1]
            img = np.clip((img + 1) / 2, 0, 1)
            return img
        
        # Convert images to numpy
        original_np = tensor_to_np(original)
        reconstructed_np = tensor_to_np(reconstructed)
        edited_np = tensor_to_np(edited)
        
        # Compute pixelwise difference
        diff = np.abs(edited_np - reconstructed_np)
        
        images = [original_np, reconstructed_np, edited_np, diff]
        titles = [
            f"Original\n'{source_prompt}'",
            f"Reconstructed (same prompt)\n'{source_prompt}'",
            f"Edited (target prompt)\n'{target_prompt}'",
            f"Pixelwise Difference\n(|Edited - Reconstructed|)"
        ]
        cmaps = ['gray', 'gray', 'gray', 'hot']  # Use 'hot' colormap for difference
        
        for i, (img, title, cmap) in enumerate(zip(images, titles, cmaps)):
            if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
                im = axes[i].imshow(img.squeeze(), cmap=cmap)
            else:
                im = axes[i].imshow(img)
            
            axes[i].set_title(title, fontsize=10, wrap=True)
            axes[i].axis('off')
            
            # Add colorbar for difference map
            if i == 3:
                cbar = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
                cbar.set_label('Absolute Difference', rotation=270, labelpad=15)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', format='png')
        print(f"💾 Results saved to {save_path}")
        print(f"   Mean absolute difference: {diff.mean():.4f}")
        print(f"   Max absolute difference: {diff.max():.4f}")
        
        # Also compute reconstruction error
        recon_diff = np.abs(original_np - reconstructed_np)
        print(f"   Reconstruction error (mean): {recon_diff.mean():.4f}")
        print(f"   Reconstruction error (max): {recon_diff.max():.4f}")
        plt.close(fig)


def main():
    """Run quick test."""
    
    # Initialize tester
    tester = RadEditDDPMInversionTester(
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # Define test cases - RadEdit uses clinical impression style prompts
    # DDPM inversion uses forward diffusion (adding noise)
    test_cases = [
        {
            "source": "Small right-sided pleural effusion",
            "target": "No acute cardiopulmonary process",
            "steps": 50,
            "guidance": 7.5,
            "seed": 42,
        },
        {
            "source": "Bilateral pleural effusions",
            "target": "Large right-sided pleural effusion",
            "steps": 50,
            "guidance": 7.5,
            "seed": 123,
        },
        {
            "source": "Cardiomegaly",
            "target": "No acute cardiopulmonary process",
            "steps": 50,
            "guidance": 7.5,
            "seed": 456,
        },
    ]
    
    # Run first test
    test = test_cases[0]
    original, reconstructed, edited, trajectory = tester.run_test(
        source_prompt=test["source"],
        target_prompt=test["target"],
        num_inference_steps=test["steps"],
        guidance_scale=test["guidance"],
        seed=test["seed"],
    )
    
    print("✅ Test complete!")
    print(f"   Trajectory length: {len(trajectory)}")
    print(f"   Final noise shape: {trajectory[-1].shape}")
    print(f"   Final noise std: {trajectory[-1].std().item():.4f}")


if __name__ == "__main__":
    main()

