"""
Hybrid DDPM Inversion Generator combining ControlNet (graph) + Text conditioning.

Uses both:
1. ControlNet with graph embeddings for structural control
2. BioViL-T text embeddings for semantic control

Both graph and text are edited in parallel for the target pathology.

Based on: "An Edit Friendly DDPM Noise Space: Inversion and Manipulations" (CVPR 2024)
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pickle

import torch
import numpy as np
import networkx as nx
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, UNet2DConditionModel
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .base import BaseCounterfactualGenerator
from .text_inversion import PATHOLOGY_TEMPLATES, get_template_set, AVAILABLE_TEMPLATE_SETS
from graph_embedding import SimpleGraph2VecEmbedder

logger = logging.getLogger(__name__)


class HybridDDPMGenerator(BaseCounterfactualGenerator):
    """
    Hybrid generator combining ControlNet (graph) + text conditioning.
    
    Pipeline:
    1. Edit graph (add/remove pathology node) → encode with graph embedder
    2. Construct target text prompt → encode with BioViL-T
    3. DDPM forward diffusion to create latent trajectory
    4. Denoise with BOTH:
       - ControlNet gets edited graph embedding (structural control)
       - UNet gets text embedding (semantic control)
    
    Args:
        controlnet_path: Path to trained ControlNet checkpoint
        embedder_path: Path to trained SimpleGraph2VecEmbedder pickle
        device: Device for computation
        embedding_strategy: Strategy used during ControlNet training
        pretrained_model: Base RadEdit model name
        num_inference_steps: Number of diffusion steps (T)
        skip: Number of steps to skip from end (start from T-skip)
        guidance_scale: CFG scale for reconstruction
    """
    
    PATHOLOGY_MAPPING = {
        'Atelectasis': ['atelectasis'],
        'Consolidation': ['consolidation'],
        'Infiltration': ['infiltration'],
        'Pneumothorax': ['pneumothorax'],
        'Edema': ['edema', 'pulmonary edema'],
        'Emphysema': ['emphysema'],
        'Fibrosis': ['fibrosis'],
        'Effusion': ['effusion', 'pleural effusion'],
        'Pneumonia': ['pneumonia'],
        'Pleural_Thickening': ['pleural thickening', 'pleural_thickening'],
        'Cardiomegaly': ['cardiomegaly', 'enlarged heart'],
        'Nodule': ['nodule', 'lung nodule'],
        'Mass': ['mass', 'lung mass'],
        'Hernia': ['hernia', 'hiatal hernia'],
        'Lung Lesion': ['lung lesion', 'lesion'],
        'Fracture': ['fracture', 'rib fracture'],
        'Lung Opacity': ['lung opacity', 'opacity'],
        'Enlarged Cardiomediastinum': ['enlarged cardiomediastinum', 'cardiomediastinum'],
    }
    
    PATHOLOGY_KEYWORDS = {
        'Atelectasis': ['atelectasis'],
        'Consolidation': ['consolidation'],
        'Infiltration': ['infiltration'],
        'Pneumothorax': ['pneumothorax'],
        'Edema': ['edema', 'pulmonary edema'],
        'Emphysema': ['emphysema'],
        'Fibrosis': ['fibrosis'],
        'Effusion': ['effusion', 'pleural effusion'],
        'Pneumonia': ['pneumonia'],
        'Pleural_Thickening': ['pleural thickening'],
        'Cardiomegaly': ['cardiomegaly', 'enlarged heart', 'cardiac enlargement'],
        'Nodule': ['nodule'],
        'Mass': ['mass'],
        'Hernia': ['hernia', 'hiatal hernia'],
        'Lung Lesion': ['lesion', 'lung lesion'],
        'Fracture': ['fracture'],
        'Lung Opacity': ['opacity', 'lung opacity'],
        'Enlarged Cardiomediastinum': ['cardiomediastinum', 'enlarged cardiomediastinum'],
    }
    
    def __init__(
        self,
        controlnet_path: str,
        embedder_path: str,
        device: str = 'cuda',
        embedding_strategy: str = 'linear',
        pretrained_model: str = 'microsoft/radedit',
        num_inference_steps: int = 50,
        skip: int = 36,
        guidance_scale: float = 15.0,
        controlnet_scale: float = 1.0,
        use_independent_noise: bool = False,
        template_set: str = 'default',
    ):
        self.device = device
        self.embedding_strategy = embedding_strategy
        self.num_inference_steps = num_inference_steps
        self.skip = skip
        self.guidance_scale = guidance_scale
        self.controlnet_scale = controlnet_scale
        self.use_independent_noise = use_independent_noise
        self.template_set = template_set
        self.templates = get_template_set(template_set)
        
        logger.info("Initializing HybridDDPMGenerator...")
        logger.info(f"  skip={skip}, guidance_scale={guidance_scale}, controlnet_scale={controlnet_scale}, use_independent_noise={use_independent_noise}, template_set={template_set}")
        
        # Load models (including text encoder)
        self._load_models(controlnet_path, pretrained_model)
        
        # Load graph embedder
        self._load_embedder(embedder_path)
        
        logger.info("HybridDDPMGenerator initialized successfully")
    
    def _load_models(self, controlnet_path: str, pretrained_model: str):
        """Load UNet, VAE, ControlNet, BioViL-T text encoder, and schedulers."""
        from controlnet import GraphControlNet
        
        logger.info(f"Loading models from {pretrained_model}...")
        
        # Load UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model,
            subfolder="unet",
        ).to(self.device)
        self.unet.eval()
        
        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sdxl-vae",
        ).to(self.device)
        self.vae.eval()
        
        # Load BioViL-T text encoder (KEEP IT for runtime text encoding)
        logger.info("Loading BioViL-T text encoder...")
        self.text_encoder = AutoModel.from_pretrained(
            "microsoft/BiomedVLP-BioViL-T",
            trust_remote_code=True,
        ).to(self.device)
        self.text_encoder.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/BiomedVLP-BioViL-T",
            model_max_length=128,
            trust_remote_code=True,
        )
        
        # Precompute empty prompt embeddings for CFG
        logger.info("Precomputing empty prompt embeddings...")
        with torch.no_grad():
            self.empty_prompt_embeds = self._encode_text([""])
        
        # Load ControlNet
        logger.info(f"Loading ControlNet from {controlnet_path}...")
        self.controlnet = GraphControlNet(self.unet, embedding_strategy=self.embedding_strategy)
        checkpoint = torch.load(controlnet_path, map_location=self.device)
        self.controlnet.load_state_dict(checkpoint)
        self.controlnet = self.controlnet.to(self.device)
        self.controlnet.eval()
        
        # Setup DDPM scheduler for forward process
        self.ddpm_scheduler = DDPMScheduler(
            beta_schedule="linear",
            clip_sample=False,
            prediction_type="epsilon",
            timestep_spacing="trailing",
            steps_offset=1,
        )
        
        # Setup DDIM scheduler for denoising
        self.ddim_scheduler = DDIMScheduler(
            beta_schedule="linear",
            clip_sample=False,
            prediction_type="epsilon",
            timestep_spacing="trailing",
            steps_offset=1,
        )
        
        logger.info("Models loaded successfully")
    
    def _load_embedder(self, embedder_path: str):
        """Load trained graph embedder."""
        logger.info(f"Loading graph embedder from {embedder_path}...")
        with open(embedder_path, 'rb') as f:
            self.embedder = pickle.load(f)
        logger.info("Graph embedder loaded successfully")
    
    @property
    def supported_pathologies(self) -> List[str]:
        """Return list of supported pathologies."""
        return list(self.PATHOLOGY_MAPPING.keys())
    
    def _encode_text(self, prompts: List[str]) -> torch.Tensor:
        """Encode text prompts to embeddings using BioViL-T."""
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        with torch.no_grad():
            embeddings = self.text_encoder(
                input_ids=text_inputs.input_ids.to(self.device),
                attention_mask=text_inputs.attention_mask.to(self.device),
            ).last_hidden_state
        
        return embeddings
    
    def _find_pathology_in_graph(self, graph: nx.DiGraph, pathology: str) -> Optional[str]:
        """Find if pathology exists in graph (case-insensitive)."""
        possible_names = self.PATHOLOGY_MAPPING.get(pathology, [pathology.lower()])
        graph_nodes_lower = {str(node).lower(): node for node in graph.nodes()}
        
        for name in possible_names:
            if name.lower() in graph_nodes_lower:
                return graph_nodes_lower[name.lower()]
        
        return None
    
    def _edit_graph_for_pathology(
        self,
        graph: nx.DiGraph,
        pathology: str,
    ) -> Tuple[nx.DiGraph, str]:
        """Edit graph by adding or removing pathology node."""
        existing_node = self._find_pathology_in_graph(graph, pathology)
        edited_graph = graph.copy()
        
        if existing_node is not None:
            edited_graph.remove_node(existing_node)
            direction = "decrease"
        else:
            node_name = pathology.lower().replace('_', ' ')
            edited_graph.add_node(node_name, type='pathology')
            direction = "increase"
        
        return edited_graph, direction
    
    def _construct_target_prompt(self, pathology: str, direction: str) -> str:
        """Construct target prompt using templates."""
        templates = self.templates[direction]
        if pathology in templates:
            return templates[pathology]
        else:
            return templates['default'].format(pathology=pathology.lower().replace('_', ' '))
    
    def _encode_graphs(self, graphs: List[nx.DiGraph]) -> np.ndarray:
        """Encode graphs to embeddings using the trained embedder."""
        return self.embedder.encode(graphs)
    
    def _encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image tensor to latent space using VAE."""
        with torch.no_grad():
            image = image.to(self.device, dtype=torch.float32)
            latents = self.vae.encode(image).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        return latents
    
    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to image space."""
        latents = latents / self.vae.config.scaling_factor
        with torch.no_grad():
            image = self.vae.decode(latents, return_dict=False)[0]
        return image
    
    def _get_variance(self, timestep: int) -> torch.Tensor:
        """Compute variance for a given timestep (matches reference implementation)."""
        prev_timestep = timestep - self.ddpm_scheduler.config.num_train_timesteps // self.ddpm_scheduler.num_inference_steps
        alpha_prod_t = self.ddpm_scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.ddpm_scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0, device=self.device)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance
    
    @torch.no_grad()
    def ddpm_forward_diffusion(
        self,
        image: torch.Tensor,
        graph_embedding: torch.Tensor,
        num_inference_steps: Optional[int] = None,
        eta: float = 1.0,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        DDPM forward diffusion with error correction (matches reference implementation).
        
        Args:
            image: [B, 3, H, W] image tensor in [-1, 1] range
            graph_embedding: [B, 768] graph embedding for conditioning during inversion
            num_inference_steps: Number of diffusion steps
            eta: Stochasticity parameter (1.0 = full stochastic, 0.0 = deterministic DDIM)
            
        Returns:
            Tuple of (z_T, latent_trajectory, zs)
            - z_T: [B, 4, H/8, W/8] fully noised latents
            - latent_trajectory: [z_0, z_1, ..., z_T] latents at each timestep
            - zs: [T, B, 4, H/8, W/8] normalized noise vectors for error correction
            
        Note:
            If use_independent_noise=True, uses UNet-based inversion with error correction
            as per "An Edit Friendly DDPM Noise Space" (CVPR 2024).
        """
        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps
        
        # Encode to latents
        z_0 = self._encode_image(image)
        batch_size = z_0.shape[0]
        
        # Setup scheduler
        self.ddpm_scheduler.set_timesteps(num_inference_steps)
        timesteps = self.ddpm_scheduler.timesteps.to(self.device)
        
        # Get alphas_cumprod for the forward process
        alphas_cumprod = self.ddpm_scheduler.alphas_cumprod.to(self.device)
        sqrt_one_minus_alpha_bar = (1 - alphas_cumprod) ** 0.5
        
        # Create timestep to index mapping
        t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
        
        # Initialize trajectories
        xts = torch.zeros((num_inference_steps + 1, *z_0.shape), device=self.device)
        xts[0] = z_0
        
        # Sample xts from x0 with independent noise (closed-form)
        for t in reversed(timesteps):
            idx = num_inference_steps - t_to_idx[int(t)]
            xts[idx] = z_0 * (alphas_cumprod[t] ** 0.5) + torch.randn_like(z_0) * sqrt_one_minus_alpha_bar[t]
        
        # Initialize zs (normalized noise vectors)
        zs = torch.zeros((num_inference_steps, *z_0.shape), device=self.device)
        
        if not self.use_independent_noise or eta == 0:
            # Original mode: use closed-form, no error correction
            latent_trajectory = [xts[i].clone() for i in range(num_inference_steps + 1)]
            z_T = latent_trajectory[-1]
            return z_T, latent_trajectory, zs
        
        # Edit-friendly inversion with error correction (matching reference)
        # Prepare conditioning
        empty_embeddings = self.empty_prompt_embeds.repeat(batch_size, 1, 1)
        graph_embedding = graph_embedding.to(self.device)
        
        for t in timesteps:
            idx = num_inference_steps - t_to_idx[int(t)] - 1
            
            # Get x_{t+1} from pre-computed trajectory
            xt = xts[idx + 1].clone()
            
            # Predict noise using ControlNet + UNet
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                noisy_latents=xt,
                timestep=t,
                graph_embeddings=graph_embedding,
                conditioning_scale=self.controlnet_scale,
                return_dict=False,
            )
            
            noise_pred = self.unet(
                xt,
                t,
                encoder_hidden_states=empty_embeddings,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                return_dict=False,
            )[0]
            
            # Compute predicted x0
            alpha_bar_t = alphas_cumprod[t]
            pred_x0 = (xt - (1 - alpha_bar_t) ** 0.5 * noise_pred) / alpha_bar_t ** 0.5
            
            # Get previous timestep info
            prev_timestep = t - self.ddpm_scheduler.config.num_train_timesteps // self.ddpm_scheduler.num_inference_steps
            alpha_prod_t_prev = alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0, device=self.device)
            
            # Compute variance
            variance = self._get_variance(t)
            
            # Compute predicted sample direction
            pred_sample_direction = (1 - alpha_prod_t_prev - eta * variance) ** 0.5 * noise_pred
            
            # Compute mu_xt (expected x_{t-1})
            mu_xt = alpha_prod_t_prev ** 0.5 * pred_x0 + pred_sample_direction
            
            # Get actual x_{t-1} from trajectory
            xtm1 = xts[idx].clone()
            
            # Compute z: the normalized noise that gives us the actual x_{t-1}
            z = (xtm1 - mu_xt) / (eta * variance ** 0.5 + 1e-8)
            zs[idx] = z
            
            # Error correction: update trajectory to avoid error accumulation
            xtm1_corrected = mu_xt + eta * variance ** 0.5 * z
            xts[idx] = xtm1_corrected
        
        # Set zs[0] to zero (no noise at t=0)
        zs[0] = torch.zeros_like(zs[0])
        
        # Build latent trajectory list
        latent_trajectory = [xts[i].clone() for i in range(num_inference_steps + 1)]
        z_T = latent_trajectory[-1]
        
        return z_T, latent_trajectory, zs
    
    @torch.no_grad()
    def edit_friendly_denoise(
        self,
        latent_trajectory: List[torch.Tensor],
        original_graph_embedding: torch.Tensor,
        edited_graph_embedding: torch.Tensor,
        text_embedding: torch.Tensor,
        zs: Optional[torch.Tensor] = None,
        skip: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        eta: float = 1.0,
    ) -> torch.Tensor:
        """
        Denoise with BOTH ControlNet (graph) AND text conditioning.
        
        This is the key hybrid method:
        - ControlNet receives graph embeddings (structural control)
        - UNet receives text embeddings (semantic control)
        
        CFG uses:
        - Uncond: original graph + empty text
        - Cond: edited graph + target text
        
        Args:
            latent_trajectory: [z_0, z_1, ..., z_T] from forward diffusion
            original_graph_embedding: [B, 768] original (unedited) graph embedding
            edited_graph_embedding: [B, 768] edited graph embedding
            text_embedding: [B, 128, 768] target text embedding
            zs: [T, B, 4, H/8, W/8] normalized noise vectors from forward pass (for error correction)
            skip: Number of timesteps to skip
            num_inference_steps: Total number of timesteps
            guidance_scale: CFG scale - amplifies both graph and text edit effect
            eta: Stochasticity parameter (1.0 = use stored zs, 0.0 = deterministic)
            
        Returns:
            [B, 3, H, W] edited image tensor
        """
        if skip is None:
            skip = self.skip
        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
        
        # Clamp skip to valid range
        skip = min(skip, num_inference_steps - 1)
        skip = max(skip, 0)
        
        # Start from z_{T-skip}
        start_idx = len(latent_trajectory) - 1 - skip
        latents = latent_trajectory[start_idx].clone()
        
        batch_size = latents.shape[0]
        
        # Setup scheduler
        self.ddim_scheduler.set_timesteps(num_inference_steps)
        
        # Get timesteps we're actually using
        timesteps = self.ddim_scheduler.timesteps.to(self.device)
        active_timesteps = timesteps[skip:]
        
        # Create timestep to index mapping
        t_to_idx = {int(v): k for k, v in enumerate(timesteps[skip:])}
        
        # Get alphas_cumprod
        alphas_cumprod = self.ddpm_scheduler.alphas_cumprod.to(self.device)
        
        # Prepare embeddings
        empty_embeddings = self.empty_prompt_embeds.repeat(batch_size, 1, 1)
        original_graph_embedding = original_graph_embedding.to(self.device)
        edited_graph_embedding = edited_graph_embedding.to(self.device)
        text_embedding = text_embedding.to(self.device)
        
        # Denoising loop with DUAL conditioning and error correction
        for t in active_timesteps:
            idx = num_inference_steps - t_to_idx[int(t)] - (num_inference_steps - len(active_timesteps) + 1)
            
            if guidance_scale > 1.0:
                # Classifier-free guidance with DIFFERENT graph AND text conditions
                # Unconditioned: original graph + empty text
                down_uncond, mid_uncond = self.controlnet(
                    noisy_latents=latents,
                    timestep=t,
                    graph_embeddings=original_graph_embedding,
                    conditioning_scale=self.controlnet_scale,
                    return_dict=False,
                )
                
                # Conditioned: edited graph + target text
                down_cond, mid_cond = self.controlnet(
                    noisy_latents=latents,
                    timestep=t,
                    graph_embeddings=edited_graph_embedding,
                    conditioning_scale=self.controlnet_scale,
                    return_dict=False,
                )
                
                latent_input = torch.cat([latents] * 2)
                
                # Text: empty for uncond, target for cond
                encoder_hidden_states = torch.cat([empty_embeddings, text_embedding])
                
                # Combine ControlNet outputs: [uncond, cond]
                down_block_res_samples = [torch.cat([u, c]) for u, c in zip(down_uncond, down_cond)]
                mid_block_res_sample = torch.cat([mid_uncond, mid_cond])
                
                noise_pred = self.unet(
                    latent_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]
                
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
            else:
                # No CFG - use edited graph and target text directly
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    noisy_latents=latents,
                    timestep=t,
                    graph_embeddings=edited_graph_embedding,
                    conditioning_scale=self.controlnet_scale,
                    return_dict=False,
                )
                
                noise_pred = self.unet(
                    latents,
                    t,
                    encoder_hidden_states=text_embedding,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]
            
            # Reverse step with stored zs (error correction)
            latents = self._reverse_step(
                noise_pred, t, latents,
                eta=eta if self.use_independent_noise else 0,
                variance_noise=zs[idx] if (zs is not None and self.use_independent_noise and idx < zs.shape[0]) else None
            )
        
        # Decode to image
        image = self._decode_latents(latents)
        return image
    
    def _reverse_step(
        self,
        noise_pred: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0,
        variance_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Single reverse diffusion step (matches reference implementation).
        
        Args:
            noise_pred: Predicted noise from UNet
            timestep: Current timestep
            sample: Current latent sample
            eta: Stochasticity parameter
            variance_noise: Pre-computed noise (z from forward pass) for error correction
            
        Returns:
            Previous sample (x_{t-1})
        """
        # 1. Get previous timestep
        prev_timestep = timestep - self.ddpm_scheduler.config.num_train_timesteps // self.ddpm_scheduler.num_inference_steps
        
        # 2. Compute alphas, betas
        alpha_prod_t = self.ddpm_scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.ddpm_scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0, device=self.device)
        beta_prod_t = 1 - alpha_prod_t
        
        # 3. Compute predicted original sample (x_0)
        pred_original_sample = (sample - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
        
        # 4. Compute variance
        variance = self._get_variance(timestep)
        
        # 5. Compute "direction pointing to x_t"
        pred_sample_direction = (1 - alpha_prod_t_prev - eta * variance) ** 0.5 * noise_pred
        
        # 6. Compute x_{t-1} without noise
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        
        # 7. Add noise if eta > 0
        if eta > 0:
            if variance_noise is None:
                variance_noise = torch.randn_like(noise_pred, device=self.device)
            sigma_z = eta * variance ** 0.5 * variance_noise
            prev_sample = prev_sample + sigma_z
        
        return prev_sample
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> List[Image.Image]:
        """Convert tensor [B, 3, H, W] in [-1, 1] to list of PIL images."""
        images = (tensor / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        
        pil_images = []
        for img in images:
            img = (img * 255).round().astype("uint8")
            pil_images.append(Image.fromarray(img))
        
        return pil_images
    
    def generate_batch(
        self,
        batch: Dict[str, Any],
        target_pathology: str,
    ) -> Dict[str, Any]:
        """
        Generate counterfactual images using BOTH graph and text conditioning.
        
        Args:
            batch: Dictionary from dataloader containing:
                - 'image': torch.Tensor [B, 3, H, W] in [-1, 1]
                - 'graph': List[nx.DiGraph]
                - 'text_prompt': Optional[List[str]] - source text prompts
            target_pathology: Target pathology to modify
            
        Returns:
            Dictionary containing counterfactual images and metadata
        """
        self.validate_batch(batch)
        self.validate_pathology(target_pathology)
        
        if 'graph' not in batch:
            raise ValueError(
                "Batch must contain 'graph' key. "
                "Use dataloader with load_graphs=True."
            )
        
        original_images_tensor = batch['image'].to(self.device)
        graphs = batch['graph']
        
        batch_size = original_images_tensor.shape[0]
        
        # Edit graphs and determine directions
        edited_graphs = []
        directions = []
        target_prompts = []
        edit_stats = {'added': 0, 'removed': 0}
        
        for graph in graphs:
            edited_graph, direction = self._edit_graph_for_pathology(graph, target_pathology)
            edited_graphs.append(edited_graph)
            directions.append(direction)
            
            if direction == 'increase':
                edit_stats['added'] += 1
            else:
                edit_stats['removed'] += 1
            
            # Construct corresponding text prompt
            target_prompt = self._construct_target_prompt(target_pathology, direction)
            target_prompts.append(target_prompt)
        
        # Encode graphs
        original_embeddings = self._encode_graphs(list(graphs))
        edited_embeddings = self._encode_graphs(edited_graphs)
        
        original_embeddings_tensor = torch.from_numpy(original_embeddings).float().to(self.device)
        edited_embeddings_tensor = torch.from_numpy(edited_embeddings).float().to(self.device)
        
        # Encode text prompts
        target_text_embeddings = self._encode_text(target_prompts)
        
        # Compute embedding similarities
        embedding_similarities = []
        for orig, edit in zip(original_embeddings, edited_embeddings):
            sim = float(np.dot(orig, edit) / (np.linalg.norm(orig) * np.linalg.norm(edit)))
            embedding_similarities.append(sim)
        
        # Step 1: DDPM forward diffusion with error correction
        # Use original graph embedding for inversion (to compute zs for reconstruction)
        logger.debug(f"DDPM forward diffusion for {batch_size} images...")
        z_T, latent_trajectory, zs = self.ddpm_forward_diffusion(
            original_images_tensor,
            original_embeddings_tensor,  # Use original embedding for inversion
        )
        
        # Step 2a: Reconstruct with ORIGINAL graph + neutral text (baseline)
        # Both branches use original graph + empty text - no edit, just reconstruction
        # Use stored zs for perfect reconstruction
        logger.debug(f"Reconstructing with original embedding (skip={self.skip})...")
        empty_text = self.empty_prompt_embeds.repeat(batch_size, 1, 1)
        reconstructed_tensor = self.edit_friendly_denoise(
            latent_trajectory,
            original_embeddings_tensor,  # uncond: original
            original_embeddings_tensor,  # cond: original (no edit)
            empty_text,
            zs=zs,  # Use stored noise for reconstruction
        )
        
        # Step 2b: Edit with BOTH edited graph AND target text
        # CFG amplifies: (original graph, empty text) → (edited graph, target text)
        # Use stored zs to maintain structure while changing semantics
        logger.debug(f"Hybrid denoising with edited graph + target text (skip={self.skip})...")
        counterfactual_tensor = self.edit_friendly_denoise(
            latent_trajectory,
            original_embeddings_tensor,   # uncond: original
            edited_embeddings_tensor,     # cond: edited
            target_text_embeddings,
            zs=zs,  # Use stored noise to maintain structure
        )
        
        # Convert to PIL
        original_images = self._tensor_to_pil(original_images_tensor)
        reconstructed_images = self._tensor_to_pil(reconstructed_tensor)
        counterfactual_images = self._tensor_to_pil(counterfactual_tensor)
        
        return {
            'original_images': original_images,
            'reconstructed_images': reconstructed_images,
            'counterfactual_images': counterfactual_images,
            'directions': directions,
            'metadata': {
                'method': 'hybrid_fixed' if self.use_independent_noise else 'hybrid_ddpm',
                'edit_stats': edit_stats,
                'embedding_similarities': embedding_similarities,
                'mean_similarity': float(np.mean(embedding_similarities)),
                'std_similarity': float(np.std(embedding_similarities)),
                'target_prompts': target_prompts,
                'num_inference_steps': self.num_inference_steps,
                'skip': self.skip,
                'guidance_scale': self.guidance_scale,
                'use_independent_noise': self.use_independent_noise,
            }
        }

