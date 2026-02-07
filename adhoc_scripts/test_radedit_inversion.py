"""
Quick test for DDIM inversion on Microsoft RadEdit chest X-ray model.

Protocol:
1. Generate synthetic CXR with initial prompt
2. Run DDIM inversion 
3. Reconstruct with different target prompt
4. Visualize comparison grid with pixelwise difference
"""

import os
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving
import matplotlib.pyplot as plt
import numpy as np
from diffusers import (
    AutoencoderKL, 
    DDIMScheduler, 
    DDIMInverseScheduler,
    StableDiffusionPipeline, 
    UNet2DConditionModel
)
from transformers import AutoModel, AutoTokenizer
from PIL import Image
from tqdm.auto import tqdm

# Set HTTP proxy for downloading models
os.environ['HTTP_PROXY'] = "http://admin:Haruhi_123@192.168.1.1:1338"
os.environ['HTTPS_PROXY'] = "http://admin:Haruhi_123@192.168.1.1:1338"


class RadEditInversionTester:
    """Quick tester for DDIM inversion on Microsoft RadEdit model."""
    
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
        
        # Setup DDIM schedulers for inversion
        self.scheduler = DDIMScheduler(
            beta_schedule="linear",
            clip_sample=False,
            prediction_type="epsilon",
            timestep_spacing="trailing",
            steps_offset=1,
        )
        self.inv_scheduler = DDIMInverseScheduler(
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
    
    def generate_image(self, prompt: str, num_inference_steps: int = 100, seed: int = 42):
        """Generate synthetic chest X-ray."""
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
    
    def ddim_inversion(
        self,
        image: Image.Image,
        prompt: str,
        num_inference_steps: int = 100,
        guidance_scale: float = 1.0,
    ):
        """Perform DDIM inversion to get noise trajectory.
        
        Note: For accurate inversion, guidance_scale should match reconstruction.
        However, using guidance_scale=1.0 (no CFG) often works better for inversion.
        """
        print(f"🔄 Running DDIM inversion ({num_inference_steps} steps, guidance={guidance_scale})...")
        
        # Encode image to latents
        latents = self._encode_image(image)
        
        # Setup inverse scheduler
        self.inv_scheduler.set_timesteps(num_inference_steps)
        timesteps = self.inv_scheduler.timesteps
        
        # Get text embeddings
        text_embeds = self._get_text_embeddings(prompt)
        uncond_embeds = self._get_text_embeddings("")
        
        trajectory = [latents.clone()]
        
        for t in tqdm(timesteps, desc="Inverting"):
            with torch.no_grad():
                if guidance_scale == 1.0:
                    # No classifier-free guidance
                    noise_pred = self.unet(
                        latents, t, encoder_hidden_states=text_embeds
                    ).sample
                else:
                    # Use classifier-free guidance during inversion
                    latent_input = torch.cat([latents] * 2)
                    text_input = torch.cat([uncond_embeds, text_embeds])
                    
                    noise_pred = self.unet(
                        latent_input, t, encoder_hidden_states=text_input
                    ).sample
                    
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                
                latents = self.inv_scheduler.step(noise_pred, t, latents).prev_sample
            
            trajectory.append(latents.clone())
        
        print(f"✅ Inversion complete. Final noise std: {latents.std().item():.4f}")
        return latents, trajectory
    
    def reconstruct_from_noise(
        self,
        noise: torch.Tensor,
        prompt: str,
        num_inference_steps: int = 100,
        guidance_scale: float = 7.5,
    ):
        """Reconstruct image from inverted noise using target prompt."""
        print(f"🎨 Reconstructing with prompt: '{prompt}'")
        
        # Setup forward scheduler
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
            
            # DDIM step
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decode to image
        image = self._decode_latents(latents)
        return image
    
    def run_test(
        self,
        source_prompt: str,
        target_prompt: str,
        num_inference_steps: int = 100,
        guidance_scale: float = 1.0,
        inversion_guidance_scale: float = 1.0,
        seed: int = 42,
    ):
        """
        Full test protocol:
        1. Generate image with source prompt
        2. Invert it (using inversion_guidance_scale)
        3. Reconstruct with same prompt (test fidelity, using guidance_scale)
        4. Reconstruct with target prompt (test editing, using guidance_scale)
        
        Note: For best reconstruction, use guidance_scale=1.0 (no CFG).
        For editing, you can increase guidance_scale for stronger effects.
        """
        print("=" * 80)
        print("🧪 Starting DDIM Inversion Test (RadEdit)")
        print(f"   Source: '{source_prompt}'")
        print(f"   Target: '{target_prompt}'")
        print(f"   Inversion guidance: {inversion_guidance_scale}")
        print(f"   Reconstruction guidance: {guidance_scale}")
        print("=" * 80)
        
        # Step 1: Generate original
        original_image = self.generate_image(
            source_prompt, 
            num_inference_steps=num_inference_steps,
            seed=seed
        )
        
        # Step 2: Invert (use inversion_guidance_scale)
        noise, trajectory = self.ddim_inversion(
            original_image,
            source_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=inversion_guidance_scale,
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
        save_path: str = "radedit_inversion_results.png",
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
        plt.close(fig)


def main():
    """Run quick test."""
    
    # Initialize tester
    tester = RadEditInversionTester(
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # Define test cases - RadEdit uses clinical impression style prompts
    # Note: Using guidance_scale=1.0 for better reconstruction fidelity
    test_cases = [
        {
            "source": "Small right-sided pleural effusion",
            "target": "No acute cardiopulmonary process",
            "steps": 100,
            "guidance": 1.0,  # No CFG for accurate inversion
            "inversion_guidance": 1.0,
            "seed": 42,
        },
        {
            "source": "Bilateral pleural effusions",
            "target": "Large right-sided pleural effusion",
            "steps": 100,
            "guidance": 1.0,
            "inversion_guidance": 1.0,
            "seed": 123,
        },
        {
            "source": "Cardiomegaly",
            "target": "No acute cardiopulmonary process",
            "steps": 100,
            "guidance": 1.0,
            "inversion_guidance": 1.0,
            "seed": 456,
        },
    ]
    
    # Run first test
    test = test_cases[2]
    original, reconstructed, edited, trajectory = tester.run_test(
        source_prompt=test["source"],
        target_prompt=test["target"],
        num_inference_steps=test["steps"],
        guidance_scale=test["guidance"],
        inversion_guidance_scale=test["inversion_guidance"],
        seed=test["seed"],
    )
    
    print("✅ Test complete!")
    print(f"   Trajectory length: {len(trajectory)}")
    print(f"   Final noise shape: {trajectory[-1].shape}")


if __name__ == "__main__":
    main()

