"""
From-scratch counterfactual generator.

Generates counterfactual images by editing medical graphs (add/remove pathology nodes),
re-encoding the edited graphs, and generating NEW images from random noise with ControlNet.

This is the original method - no inversion, no structure preservation from original image.
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
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import AutoModel, AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .base import BaseCounterfactualGenerator
# Import SimpleGraph2VecEmbedder so pickle can find it when loading
from graph_embedding import SimpleGraph2VecEmbedder

logger = logging.getLogger(__name__)


class FromScratchGenerator(BaseCounterfactualGenerator):
    """
    Counterfactual generator using from-scratch generation.
    
    Pipeline:
    1. Extract original graphs from batch
    2. Edit graphs (add/remove pathology node)
    3. Re-encode edited graphs with SimpleGraph2VecEmbedder
    4. Generate NEW counterfactual images from random noise with ControlNet
    
    Note: This method does NOT preserve structure from the original image.
    For structure-preserving counterfactuals, use DDIMInversionGenerator or DDPMInversionGenerator.
    
    Args:
        controlnet_path: Path to trained ControlNet checkpoint
        embedder_path: Path to trained SimpleGraph2VecEmbedder pickle
        device: Device for computation
        embedding_strategy: Strategy used during ControlNet training
        pretrained_model: Base RadEdit model name
        num_inference_steps: Number of denoising steps
        guidance_scale: Classifier-free guidance scale
    """
    
    # Pathologies from TorchXRayVision (with lowercase variants for graph matching)
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
        guidance_scale: float = 7.5,
    ):
        self.device = device
        self.embedding_strategy = embedding_strategy
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        
        logger.info("Initializing FromScratchGenerator...")
        
        # Load models
        self._load_models(controlnet_path, pretrained_model)
        
        # Load graph embedder
        self._load_embedder(embedder_path)
        
        logger.info("FromScratchGenerator initialized successfully")
    
    def _load_models(self, controlnet_path: str, pretrained_model: str):
        """Load UNet, VAE, ControlNet, and precompute empty prompt embeddings."""
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
        
        # Setup scheduler
        self.scheduler = DDIMScheduler(
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
    
    @torch.no_grad()
    def _generate_images(
        self,
        graph_embeddings: torch.Tensor,
        height: int = 512,
        width: int = 512,
    ) -> List[Image.Image]:
        """Generate images from graph embeddings (from random noise)."""
        batch_size = graph_embeddings.shape[0]
        device = self.device
        
        # Setup scheduler
        self.scheduler.set_timesteps(self.num_inference_steps)
        
        # Prepare latents (random noise)
        latent_channels = self.unet.config.in_channels
        latent_height = height // 8
        latent_width = width // 8
        
        latents = torch.randn(
            (batch_size, latent_channels, latent_height, latent_width),
            device=device,
            dtype=torch.float32,
        )
        latents = latents * self.scheduler.init_noise_sigma
        
        # Expand empty prompt embeddings to batch size
        empty_embeddings = self.empty_prompt_embeds.repeat(batch_size, 1, 1)
        
        # Ensure graph embeddings are on device
        graph_embeddings = graph_embeddings.to(device)
        
        # Denoising loop
        for t in self.scheduler.timesteps:
            # Get ControlNet outputs
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                noisy_latents=latents,
                timestep=t,
                graph_embeddings=graph_embeddings,
                conditioning_scale=1.0,
                return_dict=False,
            )
            
            if self.guidance_scale > 1.0:
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
                noise_pred = noise_pred_uncond + self.guidance_scale * (
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
        
        # Decode latents to images
        latents = latents / self.vae.config.scaling_factor
        images = self.vae.decode(latents, return_dict=False)[0]
        
        # Convert to PIL Images
        images = (images / 2 + 0.5).clamp(0, 1)
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
        """Generate counterfactual images for a batch."""
        self.validate_batch(batch)
        self.validate_pathology(target_pathology)
        
        if 'graph' not in batch:
            raise ValueError(
                "Batch must contain 'graph' key. "
                "Use dataloader with load_graphs=True."
            )
        
        original_images_tensor = batch['image']  # [B, 3, H, W]
        graphs = batch['graph']  # List of nx.DiGraph
        
        batch_size = original_images_tensor.shape[0]
        
        # Convert original images to PIL
        original_images = []
        for i in range(batch_size):
            img_tensor = original_images_tensor[i]
            img_np = ((img_tensor.permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
            original_images.append(Image.fromarray(img_np))
        
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
        
        # Compute embedding similarities
        embedding_similarities = []
        for orig, edit in zip(original_embeddings, edited_embeddings):
            sim = float(np.dot(orig, edit) / (np.linalg.norm(orig) * np.linalg.norm(edit)))
            embedding_similarities.append(sim)
        
        # Generate counterfactual images from scratch
        edited_embeddings_tensor = torch.from_numpy(edited_embeddings).float()
        counterfactual_images = self._generate_images(edited_embeddings_tensor)
        
        return {
            'original_images': original_images,
            'counterfactual_images': counterfactual_images,
            'directions': directions,
            'metadata': {
                'method': 'from_scratch',
                'edit_stats': edit_stats,
                'embedding_similarities': embedding_similarities,
                'mean_similarity': float(np.mean(embedding_similarities)),
                'std_similarity': float(np.std(embedding_similarities)),
            }
        }


# Backward compatibility alias
GraphCounterfactualGenerator = FromScratchGenerator


