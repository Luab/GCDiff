from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    DDIMInverseScheduler,
    DDIMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


class MRINullTextInversion:
    """
    Null-text inversion for MRI counterfactual generation.

    Based on: "Null-text Inversion for Editing Real Images using Guided Diffusion Models"
    https://arxiv.org/abs/2211.09794
    """

    def __init__(
        self,
        model_path: str = "runwayml/stable-diffusion-v1-5",
        unet_path: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            model_path: Base model path
            unet_path: Path to trained UNet weights
            device: Device to run on
            dtype: Model dtype
        """
        self.device = device
        self.dtype = dtype

        # Load components
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_path, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_path, subfolder="text_encoder"
        )
        self.vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")

        if unet_path:
            self.unet = UNet2DConditionModel.from_pretrained(unet_path)
        else:
            self.unet = UNet2DConditionModel.from_pretrained(
                model_path, subfolder="unet"
            )

        # Use DDIM scheduler for inversion
        self.scheduler = DDIMScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
        # Dedicated inverse scheduler (diffusers >=0.24)
        self.inv_scheduler = DDIMInverseScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )

        # Move to device
        self.text_encoder.to(device, dtype=dtype)
        self.unet.to(device, dtype=dtype)
        self.vae.to(device, dtype=dtype)

        # Get null-text embedding
        self.null_text_embeds = self._get_text_embeddings("")

    def _get_text_embeddings(self, prompt: Union[str, List[str]]) -> torch.Tensor:
        """Return CLIP embeddings with leading batch dimension.

        Accepts either a single prompt (``str``) or an ``Iterable[str]`` whose length
        matches the desired batch size.  If a *single* prompt is provided but will
        be paired with batched latents later, the embedding is automatically
        repeated to match the latent batch dimension on-the-fly in the caller.
        """

        # Ensure list of prompts for tokenizer
        if isinstance(prompt, str):
            prompt_list = [prompt]
        else:
            prompt_list = list(prompt)  # convert any iterable to list

        text_inputs = self.tokenizer(
            prompt_list,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            embeddings = self.text_encoder(text_inputs.input_ids.to(self.device))[0]

        return embeddings

    def _encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to latent space using VAE."""
        with torch.no_grad():
            latents = self.vae.encode(
                image.to(self.device, dtype=self.dtype)
            ).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        return latents

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to image space."""
        latents = latents / self.vae.config.scaling_factor
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        return image

    def ddim_inversion(
        self,
        image: torch.Tensor,
        prompt: Union[str, List[str]],
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Perform DDIM inversion to get noise trajectory.
        """
        # Configure inverse scheduler timesteps
        self.inv_scheduler.set_timesteps(num_inference_steps)
        timesteps = self.inv_scheduler.timesteps  # already in forward order

        # Encode image to latent space (x_0)
        latents = self._encode_image(image)

        trajectory = [latents.clone()]

        text_embeds = self._get_text_embeddings(prompt)
        # Ensure embeddings batch matches latents batch
        if text_embeds.shape[0] == 1 and latents.shape[0] > 1:
            text_embeds = text_embeds.repeat(latents.shape[0], 1, 1)

        for t in tqdm(timesteps, desc="DDIM Inversion"):
            with torch.no_grad():
                noise_pred = self.unet(
                    latents, t, encoder_hidden_states=text_embeds
                ).sample
                latents = self.inv_scheduler.step(noise_pred, t, latents).prev_sample

            trajectory.append(latents.clone())

        return latents, trajectory

    def optimize_null_text(
        self,
        image: torch.Tensor,
        prompt: str,
        trajectory: List[torch.Tensor],
        num_optimization_steps: int = 20,
        num_inference_steps: int = 20,
        learning_rate: float = 1e-2,
        guidance_scale: float = 7.5,
        epsilon: float = 1e-5,
    ) -> List[torch.Tensor]:
        """
        Optimize timestep-specific null-text embeddings for accurate reconstruction.
        """
        # Get text embeddings
        text_embeds = self._get_text_embeddings(prompt)

        # ------------------------------------------------------------------
        # If batch size >1 delegate to the batched implementation for speed.
        # This preserves backwards-compatibility with existing single-slice code
        # while enabling efficient multi-slice optimisation.
        # ------------------------------------------------------------------
        if image.shape[0] > 1:
            return self.optimize_null_text_batched(
                image=image,
                prompt=prompt,
                trajectory=trajectory,
                num_optimization_steps=num_optimization_steps,
                num_inference_steps=num_inference_steps,
                learning_rate=learning_rate,
                guidance_scale=guidance_scale,
                epsilon=epsilon,
            )

        self.scheduler.set_timesteps(num_inference_steps)
        # Timesteps are already provided in descending order by diffusers
        timesteps = self.scheduler.timesteps

        # Prepare context (uncond + cond embeddings)
        context = torch.cat([self.null_text_embeds, text_embeds])
        uncond_embeddings, cond_embeddings = context.chunk(2)

        uncond_embeddings_list = []
        latent_cur = trajectory[-1].clone()  # Start from noise

        bar = tqdm(
            total=num_optimization_steps * num_inference_steps,
            desc="Optimizing null-text",
        )

        # Optimize for each timestep
        for i in range(num_inference_steps):
            # Warm-start: carry over previous timestep's optimized embedding.
            # This follows the original Null-Text Inversion algorithm and
            # ensures faster convergence.
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True

            # Adaptive learning rate (monotonically decreasing, never negative)
            adaptive_lr = learning_rate * max(0.0, 1.0 - i / float(num_inference_steps))
            optimizer = torch.optim.Adam([uncond_embeddings], lr=adaptive_lr)

            # Get target (previous latent in trajectory)
            latent_prev = trajectory[len(trajectory) - i - 2]
            t = timesteps[i]

            # Get conditional noise prediction (fixed)
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(
                    latent_cur, t, cond_embeddings
                )

            # Inner optimization loop for this timestep
            for j in range(num_optimization_steps):
                # Get unconditional noise prediction (trainable)

                noise_pred_uncond = self.get_noise_pred_single(
                    latent_cur, t, uncond_embeddings
                )

                # Apply guidance
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )

                # Predict previous latent
                latents_prev_rec = self.scheduler.step(
                    noise_pred, t, latent_cur
                ).prev_sample

                # Compute loss in latent space
                loss = F.mse_loss(latents_prev_rec, latent_prev)

                # Optimize
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping (helps stability for large LR)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    uncond_embeddings, max_norm=1.0
                )

                optimizer.step()

                loss_item = loss.item()
                bar.update()

                # Update tqdm postfix instead of spamming stdout
                if j % 10 == 0 or j == num_optimization_steps - 1:
                    bar.set_postfix({"ts": i, "loss": f"{loss_item:.4e}"})

                # Early stopping if converged
                if loss_item < epsilon + i * 2e-5:
                    break

            # Update progress bar for skipped steps
            for k in range(j + 1, num_optimization_steps):
                bar.update()

            # Store optimized embedding for this timestep
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())

            # Move to next timestep using optimized embedding
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(
                    latent_cur, t, context, guidance_scale=guidance_scale
                )

        bar.close()
        return uncond_embeddings_list

    # ------------------------------------------------------------------
    # Batched variant ---------------------------------------------------
    # ------------------------------------------------------------------
    def optimize_null_text_batched(
        self,
        image: torch.Tensor,
        prompt: Union[str, List[str]],
        trajectory: List[torch.Tensor],
        num_optimization_steps: int = 20,
        num_inference_steps: int = 20,
        learning_rate: float = 1e-2,
        guidance_scale: float = 7.5,
        epsilon: float = 1e-5,
    ) -> List[torch.Tensor]:
        """Batched optimisation of timestep-specific null-text embeddings.

        This version handles a batch of *B* images jointly. It returns a list of
        length `num_inference_steps`; each entry is a tensor of shape
        ``(B, 77, 768)`` containing the optimised unconditional embedding for
        that timestep.
        """

        B = image.shape[0]

        # -----------------------------------------
        # Prepare text embeddings (cond) and repeat null-embeds for B
        # -----------------------------------------
        text_embeds = self._get_text_embeddings(prompt)
        if text_embeds.shape[0] == 1 and B > 1:
            text_embeds = text_embeds.repeat(B, 1, 1)

        null_embeds = self.null_text_embeds.repeat(B, 1, 1)

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps  # descending order

        uncond_embeddings_list: List[torch.Tensor] = []

        # Latent at current timestep (x_T)
        latent_cur = trajectory[-1].clone()

        bar = tqdm(
            total=num_optimization_steps * num_inference_steps,
            desc="Optimizing null-text (batched)",
        )

        for i in range(num_inference_steps):
            # Create per-sample trainable copy of uncond embeddings
            uncond_embeddings = null_embeds.clone().detach()
            uncond_embeddings.requires_grad = True

            optimizer = torch.optim.Adam(
                [uncond_embeddings],
                lr=learning_rate * max(0.0, 1.0 - i / float(num_inference_steps)),
            )

            latent_prev = trajectory[len(trajectory) - i - 2]
            t = timesteps[i]

            # Conditional noise prediction (fixed)
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, text_embeds)

            for j in range(num_optimization_steps):
                noise_pred_uncond = self.get_noise_pred_single(
                    latent_cur, t, uncond_embeddings
                )

                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )

                latents_prev_rec = self.scheduler.step(
                    noise_pred, t, latent_cur
                ).prev_sample

                loss = F.mse_loss(latents_prev_rec, latent_prev)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(uncond_embeddings, 1.0)
                optimizer.step()

                bar.update()

                if loss.item() < epsilon + i * 2e-5:
                    # good enough → break early
                    # Advance bar for skipped iterations
                    bar.update(num_optimization_steps - j - 1)
                    break

            # Store embedding for this timestep
            uncond_embeddings_list.append(uncond_embeddings.detach())

            # Move latents to previous step using optimised embeddings
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, text_embeds])
                latent_cur = self.get_noise_pred(
                    latent_cur, t, context, guidance_scale=guidance_scale
                )

        bar.close()

        return uncond_embeddings_list

    def get_noise_pred(self, latents, t, context, guidance_scale: float = 7.5):
        latents_input = torch.cat([latents] * 2)
        noise_pred = self.unet(latents_input, t, encoder_hidden_states=context)[
            "sample"
        ]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_prediction_text - noise_pred_uncond
        )
        # Use scheduler's native step to compute previous latent sample
        latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        return latents

    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def edit_image(
        self,
        optimized_null_embeds: Union[torch.Tensor, List[torch.Tensor]],
        trajectory: List[torch.Tensor],
        source_prompt: str,
        target_prompt: str,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        edit_strength: float = 1.0,
    ) -> torch.Tensor:
        """
        Edit image using optimized null-text embedding(s).
        Handles both single embeddings and timestep-specific embeddings.
        """
        # Get text embeddings
        source_embeds = self._get_text_embeddings(source_prompt)
        target_embeds = self._get_text_embeddings(target_prompt)

        # Interpolate between source and target
        text_embeds = source_embeds + edit_strength * (target_embeds - source_embeds)

        # Start from the final noise
        latents = trajectory[-1].clone()

        # Timesteps are already descending
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # Check if we have timestep-specific embeddings
        is_timestep_specific = isinstance(optimized_null_embeds, list)

        # Denoising loop
        for i, t in enumerate(tqdm(timesteps, desc="Editing")):
            # Use timestep-specific embedding if available, otherwise use single embedding
            if is_timestep_specific:
                uncond_embeds = (
                    optimized_null_embeds[i]
                    if i < len(optimized_null_embeds)
                    else self.null_text_embeds
                )
            else:
                uncond_embeds = optimized_null_embeds

            # Predict noise
            latent_input = torch.cat([latents, latents])
            text_input = torch.cat([uncond_embeds, text_embeds])

            with torch.no_grad():
                noise_pred = self.unet(
                    latent_input, t, encoder_hidden_states=text_input
                ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # DDIM step
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # Decode to image
        image = self._decode_latents(latents)
        return image

    # --------------------------------------------------------------
    # Prompt-to-Prompt (PTP) editing – optional advanced backend
    # --------------------------------------------------------------
    def edit_image_ptp(
        self,
        optimized_null_embeds: Union[torch.Tensor, List[torch.Tensor]],
        trajectory: List[torch.Tensor],
        source_prompt: str,
        target_prompt: str,
        *,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        edit_strength: float = 1.0,
        cross_replace_steps: Union[float, dict] = 0.8,
        self_replace_steps: float = 0.4,
        local_blend_words: Optional[List[str]] = None,
        **extra,
    ) -> torch.Tensor:
        """ """
        pass

    def generate_counterfactual(
        self,
        image: torch.Tensor,
        source_prompt: str,
        target_prompt: str,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        edit_strength: float = 1.0,
        num_optimization_steps: int = 300,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Full pipeline: invert image and generate counterfactual.
        Returns: (reconstructed, edited, optimized_null_embeds, trajectory)
        """
        print("🔄 Step 1: DDIM Inversion")
        _, trajectory = self.ddim_inversion(image, source_prompt, num_inference_steps)

        print("🎯 Step 2: Null-text Optimization")
        optimized_null_embeds = self.optimize_null_text(
            image,
            source_prompt,
            trajectory,
            num_optimization_steps=num_optimization_steps,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        print("✨ Step 3: Reconstruction Test")
        reconstructed = self.edit_image(
            optimized_null_embeds,
            trajectory,
            source_prompt,
            source_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            edit_strength=0.0,
        )

        print("🎨 Step 4: Counterfactual Generation")
        edited = self.edit_image(
            optimized_null_embeds,
            trajectory,
            source_prompt,
            target_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            edit_strength=edit_strength,
        )

        return reconstructed, edited, optimized_null_embeds, trajectory


def plot_counterfactual_results(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    edited: torch.Tensor,
    source_prompt: str,
    target_prompt: str,
):
    """Plot comparison of original, reconstructed, and edited images."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Convert to numpy and handle channel dimension
    def tensor_to_np(tensor):
        img = tensor.squeeze().cpu().numpy()
        if len(img.shape) == 3:  # (C, H, W)
            img = img.transpose(1, 2, 0)  # (H, W, C)
        if img.shape[-1] == 3:  # RGB
            img = img.mean(axis=-1)  # Convert to grayscale
        return np.clip((img + 1) / 2, 0, 1)  # Normalize to [0, 1]

    images = [original, reconstructed, edited]
    titles = [
        f"Original\n'{source_prompt}'",
        "Reconstructed",
        f"Edited\n'{target_prompt}'",
    ]

    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(tensor_to_np(img), cmap="gray")
        axes[i].set_title(title)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()
