"""
Training script for ControlNet with graph conditioning on RadEdit.

Trains GraphControlNet to condition chest X-ray generation on medical report
graph embeddings while keeping the base RadEdit UNet frozen.

Conditioning Strategy
---------------------
UNet (frozen): 
    - Receives encoded empty prompt ("" through BioViL-T)
    - Maintains in-distribution behavior (proper text encoder output)
    - Control signal comes from ControlNet residual connections

ControlNet (trainable):
    - Receives graph embeddings via cross-attention
    - Learns to translate graph structure to visual features
    - Outputs residual connections to guide UNet

Alternative Approaches
----------------------
See README_controlnet.md for other conditioning strategies including:
- Text + Graph (dual modality)
- Dual graph conditioning
- Concatenated embeddings
- Learned adapters
- Zero embeddings (not recommended)
"""

import os
import argparse
from pathlib import Path
from typing import Optional
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed

# Set tokenizers parallelism before DataLoader forks
# We only use tokenizer once for empty prompt precomputation, then DataLoader workers fork
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from data_loader import get_dataloaders
from controlnet import GraphControlNet


def get_empty_prompt_embeddings(text_encoder, tokenizer, batch_size, device, dtype):
    """
    Get embeddings for empty prompt (unconditional).
    
    This is more correct than zero tensors as it produces embeddings
    in the same distribution as the text encoder output.
    
    Args:
        text_encoder: BioViL-T text encoder
        tokenizer: BioViL-T tokenizer
        batch_size: Batch size to generate embeddings for
        device: Device to put tensors on
        dtype: Data type for tensors
    
    Returns:
        Empty prompt embeddings [batch_size, 128, 768]
    """
    text_inputs = tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    with torch.no_grad():
        embeddings = text_encoder(
            input_ids=text_inputs.input_ids.to(device),
            attention_mask=text_inputs.attention_mask.to(device),
        )
        if hasattr(embeddings, 'last_hidden_state'):
            embeddings = embeddings.last_hidden_state
        elif isinstance(embeddings, tuple):
            embeddings = embeddings[0]
    
    return embeddings.to(dtype)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ControlNet with graph conditioning")
    
    # Data arguments
    parser.add_argument("--csv_path", type=str, default="/home/luab/graph/reports_processed.csv",
                        help="Path to reports CSV")
    parser.add_argument("--graph_embeddings", type=str, default="/home/luab/graph/new_embeddings_expanded.h5",
                        help="Path to graph embeddings HDF5")
    parser.add_argument("--image_root", type=str, default="/mnt/data/CheXpert/PNG",
                        help="Root directory for CheXpert images")
    parser.add_argument("--text_field", type=str, default="section_impression",
                        choices=["section_impression", "section_findings"],
                        help="Text field to use for prompts")
    
    # Model arguments
    parser.add_argument("--pretrained_model", type=str, default="microsoft/radedit",
                        help="Pretrained RadEdit model name")
    parser.add_argument("--conditioning_scale", type=float, default=1.0,
                        help="ControlNet conditioning scale")
    parser.add_argument("--embedding_strategy", type=str, default="precomputed",
                        choices=["precomputed", "linear", "repeat"],
                        help="Embedding expansion strategy: precomputed (baseline), linear (trainable), repeat (simple)")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="checkpoints/controlnet",
                        help="Output directory for checkpoints")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size per device")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                        help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                        help="Adam beta2")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2,
                        help="Adam weight decay")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                        help="Adam epsilon")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of gradient accumulation steps")
    
    # Optimization arguments
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"],
                        help="Mixed precision training")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to save memory")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Maximum number of training samples (for testing)")
    
    # Logging arguments
    parser.add_argument("--wandb_project", type=str, default="controlnet-graph",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Weights & Biases run name")
    parser.add_argument("--log_every_n_steps", type=int, default=10,
                        help="Log every N steps")
    parser.add_argument("--save_every_n_epochs", type=int, default=1,
                        help="Save checkpoint every N epochs")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb" if args.wandb_project else None,
        project_dir=args.output_dir,
    )
    
    # Set seed
    if args.seed is not None:
        set_seed(args.seed)
    
    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize wandb
    if accelerator.is_main_process and args.wandb_project:
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=vars(args),
            init_kwargs={"wandb": {"name": args.wandb_run_name}},
        )
    
    # Load models
    accelerator.print("Loading RadEdit components...")
    
    # Load UNet
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model,
        subfolder="unet",
    )
    unet.requires_grad_(False)  # Freeze UNet
    unet.eval()
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sdxl-vae",
    )
    vae.requires_grad_(False)  # Freeze VAE
    vae.eval()
    
    # Load text encoder (BioViL-T) - not used during training but good for reference
    text_encoder = AutoModel.from_pretrained(
        "microsoft/BiomedVLP-BioViL-T",
        trust_remote_code=True,
    )
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedVLP-BioViL-T",
        model_max_length=128,
        trust_remote_code=True,
    )
    
    # Create ControlNet
    accelerator.print(f"Creating GraphControlNet with {args.embedding_strategy} strategy...")
    controlnet = GraphControlNet(unet, embedding_strategy=args.embedding_strategy)
    
    if args.gradient_checkpointing:
        controlnet.controlnet.enable_gradient_checkpointing()
    
    # Create noise scheduler
    noise_scheduler = DDPMScheduler(
        beta_schedule="linear",
        prediction_type="epsilon",
    )
    
    # Map embedding strategy to data format
    if args.embedding_strategy == "precomputed":
        # Use expanded embeddings [128, 768]
        embedding_dim = "expanded"
        embeddings_path = args.graph_embeddings  # new_embeddings_expanded.h5
    else:
        # Use original embeddings [768] for linear and repeat strategies
        embedding_dim = "original"
        embeddings_path = args.graph_embeddings
        accelerator.print(f"  Using original embeddings: {embeddings_path}")
    
    # Create dataloaders
    accelerator.print("Loading datasets...")
    train_loader, val_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        csv_path=args.csv_path,
        embeddings_path=embeddings_path,
        image_root=args.image_root,
        text_field=args.text_field,
        max_samples=args.max_train_samples,
        embedding_dim=embedding_dim,
    )
    
    accelerator.print(f"Train samples: {len(train_loader.dataset)}")
    accelerator.print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        controlnet.get_trainable_parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # Prepare for training
    controlnet, optimizer, train_loader, val_loader = accelerator.prepare(
        controlnet, optimizer, train_loader, val_loader
    )
    
    # Move frozen models to device
    unet = unet.to(accelerator.device)
    vae = vae.to(accelerator.device)
    
    # Precompute empty prompt embeddings (temporarily move text_encoder to GPU)
    accelerator.print("Precomputing empty prompt embeddings...")
    text_encoder = text_encoder.to(accelerator.device)
    empty_prompt_embeds_single = get_empty_prompt_embeddings(
        text_encoder, tokenizer, batch_size=1,
        device=accelerator.device, dtype=torch.float32
    )
    accelerator.print(f"  Empty prompt shape: {empty_prompt_embeds_single.shape}")
    
    # Free text encoder from GPU (we only needed it for precomputation)
    del text_encoder
    torch.cuda.empty_cache()
    accelerator.print("  Text encoder freed from GPU")
    
    # Calculate total training steps
    num_update_steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    max_train_steps = args.num_epochs * num_update_steps_per_epoch
    
    accelerator.print(f"\n{'='*80}")
    accelerator.print("Training Configuration:")
    accelerator.print(f"  Embedding strategy: {args.embedding_strategy}")
    accelerator.print(f"  Embedding dimension: {embedding_dim}")
    accelerator.print(f"  Number of training samples: {len(train_loader.dataset)}")
    accelerator.print(f"  Number of validation samples: {len(val_loader.dataset)}")
    accelerator.print(f"  Batch size per device: {args.batch_size}")
    accelerator.print(f"  Total batch size: {args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")
    accelerator.print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
    accelerator.print(f"  Total optimization steps: {max_train_steps}")
    accelerator.print(f"  Number of epochs: {args.num_epochs}")
    accelerator.print(f"  Learning rate: {args.learning_rate}")
    accelerator.print(f"  Mixed precision: {args.mixed_precision}")
    
    # Get unwrapped model to access methods
    unwrapped_controlnet = accelerator.unwrap_model(controlnet)
    accelerator.print(f"  Trainable parameters: {unwrapped_controlnet.get_num_trainable_parameters():,}")
    accelerator.print(f"{'='*80}\n")
    
    # Training loop
    global_step = 0
    
    for epoch in range(args.num_epochs):
        accelerator.print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        controlnet.train()
        
        progress_bar = tqdm(
            total=num_update_steps_per_epoch,
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch + 1}",
        )
        
        epoch_loss = 0.0
        
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(controlnet):
                # Get batch data
                images = batch['image']  # [B, 3, 512, 512]
                graph_embeddings = batch['graph_embedding']  # [B, 128, 768]
                
                # Encode images to latents
                with torch.no_grad():
                    latents = vae.encode(images).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                
                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # Sample timesteps
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,),
                    device=latents.device
                ).long()
                
                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get ControlNet outputs with graph conditioning
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents=noisy_latents,
                    timestep=timesteps,
                    graph_embeddings=graph_embeddings,  # Graph to ControlNet
                    conditioning_scale=args.conditioning_scale,
                    return_dict=False,
                )
                
                # Expand precomputed empty prompt embeddings to batch size
                empty_embeddings = empty_prompt_embeds_single.repeat(bsz, 1, 1).to(
                    device=noisy_latents.device,
                    dtype=noisy_latents.dtype
                )
                
                # Predict noise with UNet (frozen) + ControlNet outputs
                # UNet gets encoded empty prompt (not zeros), control from ControlNet residuals
                # Note: UNet parameters frozen, but gradients flow through to ControlNet outputs
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=empty_embeddings,  # Encoded "" prompt
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]
                
                # Compute loss
                loss = F.mse_loss(noise_pred, noise, reduction="mean")
                
                # Backward pass
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        controlnet.parameters(),
                        args.max_grad_norm
                    )
                
                optimizer.step()
                optimizer.zero_grad()
            
            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                epoch_loss += loss.detach().item()
                
                # Log metrics
                if global_step % args.log_every_n_steps == 0:
                    avg_loss = epoch_loss / (step + 1)
                    accelerator.log({
                        "train/loss": loss.item(),
                        "train/epoch": epoch,
                        "train/step": global_step,
                        "train/avg_loss": avg_loss,
                    }, step=global_step)
                    
                    progress_bar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "avg_loss": f"{avg_loss:.4f}",
                    })
        
        progress_bar.close()
        
        # Save checkpoint
        if (epoch + 1) % args.save_every_n_epochs == 0:
            if accelerator.is_main_process:
                checkpoint_path = Path(args.output_dir) / f"checkpoint-epoch-{epoch+1}"
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                
                # Save ControlNet
                unwrapped_controlnet = accelerator.unwrap_model(controlnet)
                torch.save(
                    unwrapped_controlnet.state_dict(),
                    checkpoint_path / "controlnet.pth"
                )
                
                # Save training state
                accelerator.save_state(checkpoint_path / "accelerator_state")
                
                accelerator.print(f"✓ Saved checkpoint to {checkpoint_path}")
        
        # Validation (optional, just for monitoring)
        if accelerator.is_main_process:
            controlnet.eval()
            val_loss = 0.0
            val_steps = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation", disable=True):
                    images = batch['image'].to(accelerator.device)
                    graph_embeddings = batch['graph_embedding'].to(accelerator.device)
                    
                    latents = vae.encode(images).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,),
                        device=latents.device
                    ).long()
                    
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    
                    down_block_res_samples, mid_block_res_sample = controlnet(
                        noisy_latents=noisy_latents,
                        timestep=timesteps,
                        graph_embeddings=graph_embeddings,
                        conditioning_scale=args.conditioning_scale,
                        return_dict=False,
                    )
                    
                    # Expand empty prompt embeddings for validation
                    empty_embeddings = empty_prompt_embeds_single.repeat(bsz, 1, 1).to(
                        device=noisy_latents.device,
                        dtype=noisy_latents.dtype
                    )
                    
                    noise_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=empty_embeddings,  # Encoded "" prompt
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        return_dict=False,
                    )[0]
                    
                    loss = F.mse_loss(noise_pred, noise, reduction="mean")
                    val_loss += loss.item()
                    val_steps += 1
                    
                    if val_steps >= 50:  # Limit validation steps
                        break
            
            if val_steps > 0:
                val_loss /= val_steps
                accelerator.log({
                    "val/loss": val_loss,
                    "val/epoch": epoch,
                }, step=global_step)
                accelerator.print(f"Validation loss: {val_loss:.4f}")
    
    # Save final checkpoint
    if accelerator.is_main_process:
        final_path = Path(args.output_dir) / "final"
        final_path.mkdir(parents=True, exist_ok=True)
        
        unwrapped_controlnet = accelerator.unwrap_model(controlnet)
        torch.save(
            unwrapped_controlnet.state_dict(),
            final_path / "controlnet.pth"
        )
        
        accelerator.print(f"\n✓ Training complete! Final model saved to {final_path}")
    
    accelerator.end_training()


if __name__ == "__main__":
    main()

