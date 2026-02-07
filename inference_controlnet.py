"""
Inference script for generating images with ControlNet graph conditioning.

Loads trained GraphControlNet and generates chest X-rays conditioned on
medical report graph embeddings.
"""

import argparse
from pathlib import Path
import torch
from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm
import numpy as np

from controlnet import GraphControlNet


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images with graph-conditioned ControlNet")
    
    # Model arguments
    parser.add_argument("--pretrained_model", type=str, default="microsoft/radedit",
                        help="Pretrained RadEdit model name")
    parser.add_argument("--controlnet_path", type=str, required=True,
                        help="Path to trained ControlNet checkpoint")
    parser.add_argument("--conditioning_scale", type=float, default=1.0,
                        help="ControlNet conditioning scale")
    
    # Generation arguments
    parser.add_argument("--graph_embeddings", type=str, required=True,
                        help="Path to graph embeddings HDF5 file")
    parser.add_argument("--graph_indices", type=int, nargs="+", default=[0, 1, 2, 3],
                        help="Indices of graph embeddings to use")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Classifier-free guidance scale")
    parser.add_argument("--height", type=int, default=512,
                        help="Output image height")
    parser.add_argument("--width", type=int, default=512,
                        help="Output image width")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="outputs/controlnet",
                        help="Output directory for generated images")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    print("Loading models...")
    
    # Load UNet
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model,
        subfolder="unet",
    ).to(device)
    unet.eval()
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sdxl-vae",
    ).to(device)
    vae.eval()
    
    # Load text encoder for empty prompt embeddings
    print("Loading BioViL-T text encoder...")
    text_encoder = AutoModel.from_pretrained(
        "microsoft/BiomedVLP-BioViL-T",
        trust_remote_code=True,
    ).to(device)
    text_encoder.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedVLP-BioViL-T",
        model_max_length=128,
        trust_remote_code=True,
    )
    
    # Precompute empty prompt embeddings
    print("Precomputing empty prompt embeddings...")
    with torch.no_grad():
        text_inputs = tokenizer(
            [""],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        empty_prompt_embeds = text_encoder(
            input_ids=text_inputs.input_ids.to(device),
            attention_mask=text_inputs.attention_mask.to(device),
        ).last_hidden_state  # [1, 128, 768]
    
    # Create and load ControlNet
    print(f"Loading ControlNet from {args.controlnet_path}...")
    controlnet = GraphControlNet(unet)
    
    checkpoint = torch.load(args.controlnet_path, map_location=device)
    controlnet.load_state_dict(checkpoint)
    controlnet = controlnet.to(device)
    controlnet.eval()
    
    print("✓ Models loaded successfully")
    
    # Setup scheduler
    scheduler = DDIMScheduler(
        beta_schedule="linear",
        clip_sample=False,
        prediction_type="epsilon",
        timestep_spacing="trailing",
        steps_offset=1,
    )
    scheduler.set_timesteps(args.num_inference_steps)
    
    # Load graph embeddings
    print(f"\nLoading graph embeddings from {args.graph_embeddings}...")
    import h5py
    with h5py.File(args.graph_embeddings, 'r') as f:
        all_embeddings = f['embeddings'][:]
    print(f"Loaded {len(all_embeddings)} graph embeddings with shape {all_embeddings[0].shape}")
    
    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # Generate images for each graph embedding
    print(f"\nGenerating {len(args.graph_indices)} images...")
    
    for idx in tqdm(args.graph_indices, desc="Generating"):
        if idx >= len(all_embeddings):
            print(f"Warning: Index {idx} out of range, skipping")
            continue
        
        # Get graph embedding
        graph_embedding = torch.from_numpy(all_embeddings[idx]).unsqueeze(0).to(device)  # [1, 128, 768]
        
        # Prepare latents
        latent_channels = unet.config.in_channels
        latent_height = args.height // 8
        latent_width = args.width // 8
        
        latents = torch.randn(
            (1, latent_channels, latent_height, latent_width),
            device=device,
            dtype=torch.float32,
        )
        
        # Scaling for initial latents
        latents = latents * scheduler.init_noise_sigma
        
        # Denoising loop
        with torch.no_grad():
            for t in tqdm(scheduler.timesteps, desc=f"Image {idx}", leave=False):
                # Get ControlNet outputs
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents=latents,
                    timestep=t,
                    graph_embeddings=graph_embedding,
                    conditioning_scale=args.conditioning_scale,
                    return_dict=False,
                )
                
                # Use encoded empty prompt for UNet (precomputed)
                empty_embeddings = empty_prompt_embeds.to(dtype=latents.dtype)
                
                # Predict noise with UNet + ControlNet
                if args.guidance_scale > 1.0:
                    # Classifier-free guidance
                    # Note: With empty UNet prompt, CFG acts on ControlNet residuals
                    latent_input = torch.cat([latents] * 2)
                    
                    # Both get empty prompt (unconditional for CFG on ControlNet scale)
                    encoder_hidden_states = torch.cat([empty_embeddings, empty_embeddings])
                    
                    # Duplicate ControlNet outputs
                    down_block_res_samples_dup = [torch.cat([s] * 2) for s in down_block_res_samples]
                    mid_block_res_sample_dup = torch.cat([mid_block_res_sample] * 2)
                    
                    # Predict noise
                    noise_pred = unet(
                        latent_input,
                        t,
                        encoder_hidden_states=encoder_hidden_states,
                        down_block_additional_residuals=down_block_res_samples_dup,
                        mid_block_additional_residual=mid_block_res_sample_dup,
                        return_dict=False,
                    )[0]
                    
                    # Perform guidance
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + args.guidance_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )
                else:
                    # No classifier-free guidance
                    noise_pred = unet(
                        latents,
                        t,
                        encoder_hidden_states=empty_embeddings,  # Encoded "" prompt
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        return_dict=False,
                    )[0]
                
                # Compute previous latents
                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        # Decode latents to image
        with torch.no_grad():
            latents = latents / vae.config.scaling_factor
            image = vae.decode(latents, return_dict=False)[0]
        
        # Convert to PIL Image
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).round().astype("uint8")
        pil_image = Image.fromarray(image)
        
        # Save image
        output_path = output_dir / f"graph_{idx:04d}_scale_{args.conditioning_scale:.1f}.png"
        pil_image.save(output_path)
        print(f"✓ Saved: {output_path}")
    
    print(f"\n✅ Generation complete! Images saved to {output_dir}")


if __name__ == "__main__":
    main()

