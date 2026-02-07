"""
DDIM Inversion-based counterfactual generator.

Uses deterministic DDIM inversion to map original image to noise space,
then reconstructs with edited graph embedding via ControlNet.

This preserves the original image structure while applying graph-conditioned edits.

Based on: "Denoising Diffusion Implicit Models" (Song et al., 2020)
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
from diffusers import AutoencoderKL, DDIMScheduler, DDIMInverseScheduler, UNet2DConditionModel
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .base import BaseCounterfactualGenerator
from graph_embedding import SimpleGraph2VecEmbedder

logger = logging.getLogger(__name__)


class DDIMInversionGenerator(BaseCounterfactualGenerator):
    """
    Counterfactual generator using DDIM inversion.
    
    Pipeline:
    1. DDIM invert original image to noise (with original graph embedding via ControlNet)
    2. Edit graph (add/remove pathology node)
    3. Re-encode edited graph
    4. Denoise from inverted noise (with edited graph embedding via ControlNet)
    
    This method PRESERVES the original image structure while applying edits.
    
    Args:
        controlnet_path: Path to trained ControlNet checkpoint
        embedder_path: Path to trained SimpleGraph2VecEmbedder pickle
        device: Device for computation
        embedding_strategy: Strategy used during ControlNet training
        pretrained_model: Base RadEdit model name
        num_inference_steps: Number of denoising steps
        guidance_scale: Classifier-free guidance for reconstruction
        inversion_guidance_scale: Guidance scale during inversion (1.0 recommended)
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
    
    def __init__(
        self,
        controlnet_path: str,
        embedder_path: str,
        device: str = 'cuda',
        embedding_strategy: str = 'linear',
        pretrained_model: str = 'microsoft/radedit',
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,  # Lower for better reconstruction
        inversion_guidance_scale: float = 1.0,  # No CFG during inversion for accuracy
    ):
        self.device = device
        self.embedding_strategy = embedding_strategy
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.inversion_guidance_scale = inversion_guidance_scale
        
        logger.info("Initializing DDIMInversionGenerator...")
        
        # Load models
        self._load_models(controlnet_path, pretrained_model)
        
        # Load graph embedder
        self._load_embedder(embedder_path)
        
        logger.info("DDIMInversionGenerator initialized successfully")
    
    def _load_models(self, controlnet_path: str, pretrained_model: str):
        """Load UNet, VAE, ControlNet, schedulers, and precompute empty prompt embeddings."""
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
        
        # Load text encoder for empty prompt embeddings
        logger.info("Loading BioViL-T text encoder...")
        text_encoder = AutoModel.from_pretrained(
            "microsoft/BiomedVLP-BioViL-T",
            trust_remote_code=True,
        ).to(self.device)
        text_encoder.eval()
        
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/BiomedVLP-BioViL-T",
            model_max_length=128,
            trust_remote_code=True,
        )
        
        # Precompute empty prompt embeddings
        logger.info("Precomputing empty prompt embeddings...")
        with torch.no_grad():
            text_inputs = tokenizer(
                [""],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            self.empty_prompt_embeds = text_encoder(
                input_ids=text_inputs.input_ids.to(self.device),
                attention_mask=text_inputs.attention_mask.to(self.device),
            ).last_hidden_state  # [1, 128, 768]
        
        # Free text encoder
        del text_encoder
        torch.cuda.empty_cache()
        
        # Load ControlNet
        logger.info(f"Loading ControlNet from {controlnet_path}...")
        self.controlnet = GraphControlNet(self.unet, embedding_strategy=self.embedding_strategy)
        checkpoint = torch.load(controlnet_path, map_location=self.device)
        self.controlnet.load_state_dict(checkpoint)
        self.controlnet = self.controlnet.to(self.device)
        self.controlnet.eval()
        
        # Setup forward scheduler (for denoising)
        self.scheduler = DDIMScheduler(
            beta_schedule="linear",
            clip_sample=False,
            prediction_type="epsilon",
            timestep_spacing="trailing",
            steps_offset=1,
        )
        
        # Setup inverse scheduler (for inversion)
        self.inv_scheduler = DDIMInverseScheduler(
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
    
    def _encode_graphs(self, graphs: List[nx.DiGraph]) -> np.ndarray:
        """Encode graphs to embeddings using the trained embedder."""
        return self.embedder.encode(graphs)
    
    def _encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image tensor to latent space using VAE."""
        with torch.no_grad():
            # Ensure image is on correct device
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
    
    @torch.no_grad()
    def ddim_inversion(
        self,
        image: torch.Tensor,
        graph_embedding: torch.Tensor,
        num_inference_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Perform DDIM inversion to get noise trajectory.
        
        Args:
            image: [B, 3, H, W] image tensor in [-1, 1] range
            graph_embedding: [B, 768] graph embedding for ControlNet
            num_inference_steps: Number of inversion steps
            
        Returns:
            Tuple of (final_noise, trajectory)
            - final_noise: [B, 4, H/8, W/8] inverted noise
            - trajectory: List of latents at each timestep [z_0, z_1, ..., z_T]
        """
        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps
        
        # Encode image to latent space
        latents = self._encode_image(image)
        batch_size = latents.shape[0]
        
        # Setup inverse scheduler
        self.inv_scheduler.set_timesteps(num_inference_steps)
        timesteps = self.inv_scheduler.timesteps
        
        # Prepare embeddings
        empty_embeddings = self.empty_prompt_embeds.repeat(batch_size, 1, 1)
        graph_embedding = graph_embedding.to(self.device)
        
        trajectory = [latents.clone()]
        
        # DDIM inversion: go from clean image to noise
        for t in timesteps:
            # Get ControlNet outputs with original graph embedding
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                noisy_latents=latents,
                timestep=t,
                graph_embeddings=graph_embedding,
                conditioning_scale=1.0,
                return_dict=False,
            )
            
            # Predict noise (no CFG for accurate inversion)
            noise_pred = self.unet(
                latents,
                t,
                encoder_hidden_states=empty_embeddings,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                return_dict=False,
            )[0]
            
            # DDIM inverse step
            latents = self.inv_scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            trajectory.append(latents.clone())
        
        return latents, trajectory
    
    @torch.no_grad()
    def reconstruct_with_controlnet(
        self,
        noise: torch.Tensor,
        graph_embedding: torch.Tensor,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Reconstruct image from inverted noise using ControlNet with (edited) graph embedding.
        
        Args:
            noise: [B, 4, H/8, W/8] inverted noise
            graph_embedding: [B, 768] edited graph embedding for ControlNet
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            
        Returns:
            [B, 3, H, W] reconstructed image tensor in [-1, 1] range
        """
        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
        
        batch_size = noise.shape[0]
        latents = noise.clone()
        
        # Setup forward scheduler
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        
        # Prepare embeddings
        empty_embeddings = self.empty_prompt_embeds.repeat(batch_size, 1, 1)
        graph_embedding = graph_embedding.to(self.device)
        
        # Denoising loop
        for t in timesteps:
            # Get ControlNet outputs with edited graph embedding
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                noisy_latents=latents,
                timestep=t,
                graph_embeddings=graph_embedding,
                conditioning_scale=1.0,
                return_dict=False,
            )
            
            if guidance_scale > 1.0:
                # Classifier-free guidance
                latent_input = torch.cat([latents] * 2)
                encoder_hidden_states = torch.cat([empty_embeddings, empty_embeddings])
                
                down_block_res_samples_dup = [torch.cat([s] * 2) for s in down_block_res_samples]
                mid_block_res_sample_dup = torch.cat([mid_block_res_sample] * 2)
                
                noise_pred = self.unet(
                    latent_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples_dup,
                    mid_block_additional_residual=mid_block_res_sample_dup,
                    return_dict=False,
                )[0]
                
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
            else:
                noise_pred = self.unet(
                    latents,
                    t,
                    encoder_hidden_states=empty_embeddings,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]
            
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        # Decode to image
        image = self._decode_latents(latents)
        return image
    
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
        Generate counterfactual images for a batch using DDIM inversion.
        
        Args:
            batch: Dictionary from dataloader containing:
                - 'image': torch.Tensor [B, 3, H, W] in [-1, 1]
                - 'graph': List[nx.DiGraph]
            target_pathology: Target pathology to modify
            
        Returns:
            Dictionary containing:
                - 'original_images': List[PIL.Image]
                - 'counterfactual_images': List[PIL.Image]
                - 'directions': List[str]
                - 'metadata': Dict
        """
        self.validate_batch(batch)
        self.validate_pathology(target_pathology)
        
        if 'graph' not in batch:
            raise ValueError(
                "Batch must contain 'graph' key. "
                "Use dataloader with load_graphs=True."
            )
        
        original_images_tensor = batch['image'].to(self.device)  # [B, 3, H, W]
        graphs = batch['graph']  # List of nx.DiGraph
        
        batch_size = original_images_tensor.shape[0]
        
        # Edit graphs and collect
        edited_graphs = []
        directions = []
        edit_stats = {'added': 0, 'removed': 0}
        
        for graph in graphs:
            edited_graph, direction = self._edit_graph_for_pathology(graph, target_pathology)
            edited_graphs.append(edited_graph)
            directions.append(direction)
            
            if direction == 'increase':
                edit_stats['added'] += 1
            else:
                edit_stats['removed'] += 1
        
        # Encode original and edited graphs
        original_embeddings = self._encode_graphs(list(graphs))
        edited_embeddings = self._encode_graphs(edited_graphs)
        
        original_embeddings_tensor = torch.from_numpy(original_embeddings).float().to(self.device)
        edited_embeddings_tensor = torch.from_numpy(edited_embeddings).float().to(self.device)
        
        # Compute embedding similarities
        embedding_similarities = []
        for orig, edit in zip(original_embeddings, edited_embeddings):
            sim = float(np.dot(orig, edit) / (np.linalg.norm(orig) * np.linalg.norm(edit)))
            embedding_similarities.append(sim)
        
        # Step 1: DDIM inversion with original graph embedding
        logger.debug(f"DDIM inverting {batch_size} images...")
        inverted_noise, trajectory = self.ddim_inversion(
            original_images_tensor,
            original_embeddings_tensor,
        )
        
        # Step 2a: Reconstruct with ORIGINAL embedding (baseline for measuring inversion quality)
        logger.debug(f"Reconstructing with original graph embeddings...")
        reconstructed_tensor = self.reconstruct_with_controlnet(
            inverted_noise,
            original_embeddings_tensor,
        )
        
        # Step 2b: Reconstruct with EDITED graph embedding (counterfactual)
        logger.debug(f"Reconstructing with edited graph embeddings...")
        counterfactual_tensor = self.reconstruct_with_controlnet(
            inverted_noise,
            edited_embeddings_tensor,
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
                'method': 'ddim_inversion',
                'edit_stats': edit_stats,
                'embedding_similarities': embedding_similarities,
                'mean_similarity': float(np.mean(embedding_similarities)),
                'std_similarity': float(np.std(embedding_similarities)),
                'num_inference_steps': self.num_inference_steps,
                'guidance_scale': self.guidance_scale,
                'inversion_guidance_scale': self.inversion_guidance_scale,
            }
        }

