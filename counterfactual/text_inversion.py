"""
Text-based counterfactual generators using RadEdit's native text conditioning.

Uses BioViL-T text encoder directly (no ControlNet) with DDIM/DDPM inversion.
The text prompts are constructed using templates based on the target pathology.

This serves as a baseline comparison for graph-based counterfactual generation.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler, DDIMInverseScheduler, DDPMScheduler, UNet2DConditionModel
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .base import BaseCounterfactualGenerator

logger = logging.getLogger(__name__)


# Template sets for ablation experiments on prompt style
# Each set contains 'increase' and 'decrease' templates for each pathology
TEMPLATE_SETS = {
    # Default: Structured format "Chest x-ray showing/without X"
    'default': {
        'increase': {
            'Atelectasis': "Chest x-ray showing atelectasis",
            'Consolidation': "Chest x-ray showing consolidation",
            'Infiltration': "Chest x-ray showing infiltration",
            'Pneumothorax': "Chest x-ray showing pneumothorax",
            'Edema': "Chest x-ray showing pulmonary edema",
            'Emphysema': "Chest x-ray showing emphysema",
            'Fibrosis': "Chest x-ray showing fibrosis",
            'Effusion': "Chest x-ray showing pleural effusion",
            'Pneumonia': "Chest x-ray showing pneumonia",
            'Pleural_Thickening': "Chest x-ray showing pleural thickening",
            'Cardiomegaly': "Chest x-ray showing cardiomegaly",
            'Nodule': "Chest x-ray showing lung nodule",
            'Mass': "Chest x-ray showing lung mass",
            'Hernia': "Chest x-ray showing hiatal hernia",
            'Lung Lesion': "Chest x-ray showing lung lesion",
            'Fracture': "Chest x-ray showing rib fracture",
            'Lung Opacity': "Chest x-ray showing lung opacity",
            'Enlarged Cardiomediastinum': "Chest x-ray showing enlarged cardiomediastinum",
            'default': "Chest x-ray showing {pathology}",
        },
        'decrease': {
            'Atelectasis': "Normal chest x-ray without atelectasis",
            'Consolidation': "Normal chest x-ray without consolidation",
            'Infiltration': "Normal chest x-ray without infiltration",
            'Pneumothorax': "Normal chest x-ray without pneumothorax",
            'Edema': "Normal chest x-ray without pulmonary edema",
            'Emphysema': "Normal chest x-ray without emphysema",
            'Fibrosis': "Normal chest x-ray without fibrosis",
            'Effusion': "Normal chest x-ray without pleural effusion",
            'Pneumonia': "Normal chest x-ray without pneumonia",
            'Pleural_Thickening': "Normal chest x-ray without pleural thickening",
            'Cardiomegaly': "Normal chest x-ray without cardiomegaly",
            'Nodule': "Normal chest x-ray without lung nodule",
            'Mass': "Normal chest x-ray without lung mass",
            'Hernia': "Normal chest x-ray without hiatal hernia",
            'Lung Lesion': "Normal chest x-ray without lung lesion",
            'Fracture': "Normal chest x-ray without rib fracture",
            'Lung Opacity': "Normal chest x-ray without lung opacity",
            'Enlarged Cardiomediastinum': "Normal chest x-ray without enlarged cardiomediastinum",
            'default': "Normal chest x-ray without {pathology}",
        }
    },
    # Freeform: Simple, direct statements "X is present" / "No X"
    'freeform': {
        'increase': {
            'Atelectasis': "Atelectasis is present",
            'Consolidation': "Consolidation is present",
            'Infiltration': "Infiltration is present",
            'Pneumothorax': "Pneumothorax is present",
            'Edema': "Pulmonary edema is visible",
            'Emphysema': "Emphysema is present",
            'Fibrosis': "Fibrosis is present",
            'Effusion': "Pleural effusion is present",
            'Pneumonia': "Pneumonia is present",
            'Pleural_Thickening': "Pleural thickening is present",
            'Cardiomegaly': "Cardiomegaly is present",
            'Nodule': "Lung nodule is present",
            'Mass': "Lung mass is present",
            'Hernia': "Hiatal hernia is present",
            'Lung Lesion': "Lung lesion is present",
            'Fracture': "Rib fracture is present",
            'Lung Opacity': "Lung opacity is present",
            'Enlarged Cardiomediastinum': "Enlarged cardiomediastinum is present",
            'default': "{pathology} is present",
        },
        'decrease': {
            'Atelectasis': "No atelectasis",
            'Consolidation': "No consolidation",
            'Infiltration': "No infiltration",
            'Pneumothorax': "No pneumothorax",
            'Edema': "No pulmonary edema",
            'Emphysema': "No emphysema",
            'Fibrosis': "No fibrosis",
            'Effusion': "No pleural effusion",
            'Pneumonia': "No pneumonia",
            'Pleural_Thickening': "No pleural thickening",
            'Cardiomegaly': "No cardiomegaly",
            'Nodule': "No lung nodule",
            'Mass': "No lung mass",
            'Hernia': "No hiatal hernia",
            'Lung Lesion': "No lung lesion",
            'Fracture': "No rib fracture",
            'Lung Opacity': "No lung opacity",
            'Enlarged Cardiomediastinum': "No enlarged cardiomediastinum",
            'default': "No {pathology}",
        }
    },
    # Detailed: More clinical detail with anatomical descriptions
    'detailed': {
        'increase': {
            'Atelectasis': "Chest x-ray demonstrating atelectasis with volume loss and increased opacity",
            'Consolidation': "Chest x-ray demonstrating consolidation with dense airspace opacity",
            'Infiltration': "Chest x-ray demonstrating pulmonary infiltration with diffuse opacities",
            'Pneumothorax': "Chest x-ray demonstrating pneumothorax with visible pleural line and absent lung markings",
            'Edema': "Chest x-ray demonstrating pulmonary edema with bilateral interstitial opacities and vascular congestion",
            'Emphysema': "Chest x-ray demonstrating emphysema with hyperinflation and flattened diaphragms",
            'Fibrosis': "Chest x-ray demonstrating fibrosis with reticular opacities and volume loss",
            'Effusion': "Chest x-ray demonstrating pleural effusion with blunted costophrenic angle and meniscus sign",
            'Pneumonia': "Chest x-ray demonstrating pneumonia with focal consolidation and air bronchograms",
            'Pleural_Thickening': "Chest x-ray demonstrating pleural thickening with irregular pleural margins",
            'Cardiomegaly': "Chest x-ray demonstrating cardiomegaly with enlarged cardiac silhouette exceeding half of thoracic width",
            'Nodule': "Chest x-ray demonstrating lung nodule with well-circumscribed rounded opacity",
            'Mass': "Chest x-ray demonstrating lung mass with large irregular opacity",
            'Hernia': "Chest x-ray demonstrating hiatal hernia with retrocardiac air-fluid level",
            'Lung Lesion': "Chest x-ray demonstrating lung lesion with focal parenchymal abnormality",
            'Fracture': "Chest x-ray demonstrating rib fracture with cortical disruption",
            'Lung Opacity': "Chest x-ray demonstrating lung opacity with increased parenchymal density",
            'Enlarged Cardiomediastinum': "Chest x-ray demonstrating enlarged cardiomediastinum with widened mediastinal contour",
            'default': "Chest x-ray demonstrating {pathology} with characteristic findings",
        },
        'decrease': {
            'Atelectasis': "Chest x-ray with normal lung expansion, no atelectasis or volume loss",
            'Consolidation': "Chest x-ray with clear lung parenchyma, no consolidation",
            'Infiltration': "Chest x-ray with clear lung fields, no infiltration or diffuse opacities",
            'Pneumothorax': "Chest x-ray with normal pleural apposition, no pneumothorax",
            'Edema': "Chest x-ray with clear lung fields and normal pulmonary vasculature, no edema",
            'Emphysema': "Chest x-ray with normal lung volumes and diaphragm position, no emphysema",
            'Fibrosis': "Chest x-ray with normal lung parenchyma, no fibrosis or reticular changes",
            'Effusion': "Chest x-ray with sharp costophrenic angles, no pleural effusion",
            'Pneumonia': "Chest x-ray with clear lung fields, no pneumonia or focal consolidation",
            'Pleural_Thickening': "Chest x-ray with normal smooth pleural margins, no thickening",
            'Cardiomegaly': "Chest x-ray with normal cardiac silhouette and cardiothoracic ratio, no cardiomegaly",
            'Nodule': "Chest x-ray with clear lung fields, no nodules identified",
            'Mass': "Chest x-ray with clear lung fields, no mass lesion",
            'Hernia': "Chest x-ray with normal retrocardiac region, no hiatal hernia",
            'Lung Lesion': "Chest x-ray with normal lung parenchyma, no focal lesions",
            'Fracture': "Chest x-ray with intact ribs and normal cortical margins, no fracture",
            'Lung Opacity': "Chest x-ray with clear lung fields, no abnormal opacity",
            'Enlarged Cardiomediastinum': "Chest x-ray with normal mediastinal contour, no enlargement",
            'default': "Chest x-ray with normal findings, no {pathology}",
        }
    },
}

# Available template set names for validation
AVAILABLE_TEMPLATE_SETS = list(TEMPLATE_SETS.keys())

# Default templates (backwards compatibility alias)
PATHOLOGY_TEMPLATES = TEMPLATE_SETS['default']


def get_template_set(template_set: str = 'default') -> Dict:
    """
    Get a template set by name.
    
    Args:
        template_set: Name of the template set ('default', 'freeform', 'detailed')
        
    Returns:
        Dictionary with 'increase' and 'decrease' template mappings
        
    Raises:
        ValueError: If template_set is not recognized
    """
    if template_set not in TEMPLATE_SETS:
        raise ValueError(
            f"Unknown template_set '{template_set}'. "
            f"Available: {AVAILABLE_TEMPLATE_SETS}"
        )
    return TEMPLATE_SETS[template_set]


class TextDDIMInversionGenerator(BaseCounterfactualGenerator):
    """
    Text-based counterfactual generator using DDIM inversion.
    
    Pipeline:
    1. DDIM invert original image to noise (with source text prompt)
    2. Determine direction based on whether pathology is mentioned in source text
    3. Construct target prompt using templates
    4. Denoise from inverted noise (with target text prompt)
    
    This method uses RadEdit's native text conditioning via BioViL-T,
    without any ControlNet or graph embeddings.
    
    Args:
        device: Device for computation
        pretrained_model: Base RadEdit model name
        num_inference_steps: Number of denoising steps
        guidance_scale: Classifier-free guidance for reconstruction
        inversion_guidance_scale: Guidance scale during inversion (1.0 recommended)
        template_set: Name of the template set to use ('default', 'freeform', 'detailed')
    """
    
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
        device: str = 'cuda',
        pretrained_model: str = 'microsoft/radedit',
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        inversion_guidance_scale: float = 1.0,
        template_set: str = 'default',
    ):
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.inversion_guidance_scale = inversion_guidance_scale
        self.template_set = template_set
        self.templates = get_template_set(template_set)
        
        logger.info("Initializing TextDDIMInversionGenerator...")
        logger.info(f"  template_set={template_set}")
        
        # Load models
        self._load_models(pretrained_model)
        
        logger.info("TextDDIMInversionGenerator initialized successfully")
    
    def _load_models(self, pretrained_model: str):
        """Load UNet, VAE, text encoder, and schedulers."""
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
        
        # Load BioViL-T text encoder (keep it loaded for encoding prompts)
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
        with torch.no_grad():
            self.empty_prompt_embeds = self._encode_text([""])
        
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
    
    @property
    def supported_pathologies(self) -> List[str]:
        """Return list of supported pathologies."""
        return list(self.PATHOLOGY_KEYWORDS.keys())
    
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
    
    def _pathology_in_text(self, text: str, pathology: str) -> bool:
        """Check if pathology is mentioned in text (case-insensitive)."""
        text_lower = text.lower()
        keywords = self.PATHOLOGY_KEYWORDS.get(pathology, [pathology.lower()])
        return any(kw.lower() in text_lower for kw in keywords)
    
    def _determine_direction(self, pathology: str, labels: dict = None, source_text: str = None) -> str:
        """
        Determine direction (increase/decrease) for counterfactual generation.
        
        Uses ground truth labels if available, falls back to keyword-based detection.
        
        Args:
            pathology: Target pathology name
            labels: Optional dict of pathology -> label value (1.0, 0.0, -1.0, None)
            source_text: Source text prompt (used if labels not available)
            
        Returns:
            'decrease' if pathology is present (label=1.0), 'increase' otherwise
        """
        if labels is not None:
            label_value = labels.get(pathology)
            if label_value == 1.0:  # Positive label means pathology is present
                return "decrease"
            else:  # null, 0.0, -1.0 → try to add
                return "increase"
        elif source_text is not None:
            # Fallback to keyword-based detection
            if self._pathology_in_text(source_text, pathology):
                return "decrease"
            else:
                return "increase"
        else:
            # Default to increase if no information available
            return "increase"
    
    def _construct_target_prompt(self, source_text: str, pathology: str, direction: str) -> str:
        """Construct target prompt using the selected template set."""
        templates = self.templates[direction]
        if pathology in templates:
            return templates[pathology]
        else:
            return templates['default'].format(pathology=pathology.lower().replace('_', ' '))
    
    @torch.no_grad()
    def ddim_inversion(
        self,
        image: torch.Tensor,
        source_embeds: torch.Tensor,
        num_inference_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Perform DDIM inversion to get noise trajectory.
        
        Args:
            image: [B, 3, H, W] image tensor in [-1, 1] range
            source_embeds: [B, 128, 768] source text embeddings
            num_inference_steps: Number of inversion steps
            
        Returns:
            Tuple of (final_noise, trajectory)
        """
        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps
        
        # Encode image to latent space
        latents = self._encode_image(image)
        batch_size = latents.shape[0]
        
        # Setup inverse scheduler
        self.inv_scheduler.set_timesteps(num_inference_steps)
        timesteps = self.inv_scheduler.timesteps
        
        trajectory = [latents.clone()]
        
        # DDIM inversion: go from clean image to noise
        for t in timesteps:
            # Predict noise using source text embeddings (no CFG for accurate inversion)
            noise_pred = self.unet(
                latents,
                t,
                encoder_hidden_states=source_embeds,
                return_dict=False,
            )[0]
            
            # DDIM inverse step
            latents = self.inv_scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            trajectory.append(latents.clone())
        
        return latents, trajectory
    
    @torch.no_grad()
    def reconstruct_with_text(
        self,
        noise: torch.Tensor,
        target_embeds: torch.Tensor,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Reconstruct image from inverted noise using target text embeddings.
        
        Args:
            noise: [B, 4, H/8, W/8] inverted noise
            target_embeds: [B, 128, 768] target text embeddings
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            
        Returns:
            [B, 3, H, W] reconstructed image tensor
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
        
        # Prepare embeddings for CFG
        empty_embeddings = self.empty_prompt_embeds.repeat(batch_size, 1, 1)
        
        # Denoising loop
        for t in timesteps:
            if guidance_scale > 1.0:
                # Classifier-free guidance
                latent_input = torch.cat([latents] * 2)
                text_input = torch.cat([empty_embeddings, target_embeds])
                
                noise_pred = self.unet(
                    latent_input,
                    t,
                    encoder_hidden_states=text_input,
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
                    encoder_hidden_states=target_embeds,
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
        Generate counterfactual images for a batch using DDIM inversion with text.
        
        Args:
            batch: Dictionary from dataloader containing:
                - 'image': torch.Tensor [B, 3, H, W] in [-1, 1]
                - 'text_prompt': List[str] - source text prompts
                - 'labels': Optional[List[dict]] - ground truth pathology labels
            target_pathology: Target pathology to modify
            
        Returns:
            Dictionary containing counterfactual images and metadata
        """
        self.validate_batch(batch)
        self.validate_pathology(target_pathology)
        
        if 'text_prompt' not in batch:
            raise ValueError(
                "Batch must contain 'text_prompt' key for text-based generation."
            )
        
        original_images_tensor = batch['image'].to(self.device)
        source_prompts = batch['text_prompt']
        labels_list = batch.get('labels', None)  # Optional ground truth labels
        
        batch_size = original_images_tensor.shape[0]
        
        # Determine direction for each sample and construct target prompts
        directions = []
        target_prompts = []
        edit_stats = {'added': 0, 'removed': 0}
        
        for i, source_text in enumerate(source_prompts):
            # Use labels if available, otherwise fall back to keyword detection
            labels = labels_list[i] if labels_list else None
            direction = self._determine_direction(target_pathology, labels=labels, source_text=source_text)
            
            if direction == "decrease":
                edit_stats['removed'] += 1
            else:
                edit_stats['added'] += 1
            
            directions.append(direction)
            target_prompt = self._construct_target_prompt(source_text, target_pathology, direction)
            target_prompts.append(target_prompt)
        
        # Encode source and target prompts
        source_embeds = self._encode_text(source_prompts)
        target_embeds = self._encode_text(target_prompts)
        
        # Step 1: DDIM inversion with source text
        logger.debug(f"DDIM inverting {batch_size} images with source text...")
        inverted_noise, trajectory = self.ddim_inversion(
            original_images_tensor,
            source_embeds,
        )
        
        # Step 2: Reconstruct with target text
        logger.debug(f"Reconstructing with target text prompts...")
        counterfactual_tensor = self.reconstruct_with_text(
            inverted_noise,
            target_embeds,
        )
        
        # Convert to PIL
        original_images = self._tensor_to_pil(original_images_tensor)
        counterfactual_images = self._tensor_to_pil(counterfactual_tensor)
        
        return {
            'original_images': original_images,
            'counterfactual_images': counterfactual_images,
            'directions': directions,
            'metadata': {
                'method': 'text_ddim_inversion',
                'template_set': self.template_set,
                'edit_stats': edit_stats,
                'source_prompts': source_prompts,
                'target_prompts': target_prompts,
                'num_inference_steps': self.num_inference_steps,
                'guidance_scale': self.guidance_scale,
                'inversion_guidance_scale': self.inversion_guidance_scale,
            }
        }


class TextDDPMInversionGenerator(BaseCounterfactualGenerator):
    """
    Text-based counterfactual generator using DDPM "Edit Friendly" inversion.
    
    Pipeline:
    1. DDPM forward diffusion: progressively add noise to image
    2. Determine direction based on whether pathology is mentioned in source text
    3. Construct target prompt using templates
    4. Denoise from timestep T-skip (with target text prompt)
    
    This method uses RadEdit's native text conditioning via BioViL-T,
    without any ControlNet or graph embeddings.
    
    Args:
        device: Device for computation
        pretrained_model: Base RadEdit model name
        num_inference_steps: Number of diffusion steps (T)
        skip: Number of steps to skip from end (start from T-skip)
        guidance_scale: CFG scale for reconstruction
        template_set: Name of the template set to use ('default', 'freeform', 'detailed')
    """
    
    PATHOLOGY_KEYWORDS = TextDDIMInversionGenerator.PATHOLOGY_KEYWORDS
    
    def __init__(
        self,
        device: str = 'cuda',
        pretrained_model: str = 'microsoft/radedit',
        num_inference_steps: int = 50,
        skip: int = 36,
        guidance_scale: float = 15.0,
        template_set: str = 'default',
        use_independent_noise: bool = False,
    ):
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.skip = skip
        self.guidance_scale = guidance_scale
        self.template_set = template_set
        self.templates = get_template_set(template_set)
        self.use_independent_noise = use_independent_noise
        
        logger.info("Initializing TextDDPMInversionGenerator...")
        logger.info(f"  skip={skip}, guidance_scale={guidance_scale}, template_set={template_set}, use_independent_noise={use_independent_noise}")
        
        # Load models
        self._load_models(pretrained_model)
        
        logger.info("TextDDPMInversionGenerator initialized successfully")
    
    def _load_models(self, pretrained_model: str):
        """Load UNet, VAE, text encoder, and schedulers."""
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
        
        # Load BioViL-T text encoder
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
        with torch.no_grad():
            self.empty_prompt_embeds = self._encode_text([""])
        
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
    
    @property
    def supported_pathologies(self) -> List[str]:
        """Return list of supported pathologies."""
        return list(self.PATHOLOGY_KEYWORDS.keys())
    
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
    
    def _pathology_in_text(self, text: str, pathology: str) -> bool:
        """Check if pathology is mentioned in text (case-insensitive)."""
        text_lower = text.lower()
        keywords = self.PATHOLOGY_KEYWORDS.get(pathology, [pathology.lower()])
        return any(kw.lower() in text_lower for kw in keywords)
    
    def _determine_direction(self, pathology: str, labels: dict = None, source_text: str = None) -> str:
        """
        Determine direction (increase/decrease) for counterfactual generation.
        
        Uses ground truth labels if available, falls back to keyword-based detection.
        
        Args:
            pathology: Target pathology name
            labels: Optional dict of pathology -> label value (1.0, 0.0, -1.0, None)
            source_text: Source text prompt (used if labels not available)
            
        Returns:
            'decrease' if pathology is present (label=1.0), 'increase' otherwise
        """
        if labels is not None:
            label_value = labels.get(pathology)
            if label_value == 1.0:  # Positive label means pathology is present
                return "decrease"
            else:  # null, 0.0, -1.0 → try to add
                return "increase"
        elif source_text is not None:
            # Fallback to keyword-based detection
            if self._pathology_in_text(source_text, pathology):
                return "decrease"
            else:
                return "increase"
        else:
            # Default to increase if no information available
            return "increase"
    
    def _construct_target_prompt(self, source_text: str, pathology: str, direction: str) -> str:
        """Construct target prompt using the selected template set."""
        templates = self.templates[direction]
        if pathology in templates:
            return templates[pathology]
        else:
            return templates['default'].format(pathology=pathology.lower().replace('_', ' '))
    
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
        source_embeds: Optional[torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
        eta: float = 1.0,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        DDPM forward diffusion with error correction (matches reference implementation).
        
        Args:
            image: [B, 3, H, W] image tensor in [-1, 1] range
            source_embeds: [B, 128, 768] source text embeddings for conditioning during inversion
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
        
        if not self.use_independent_noise or eta == 0 or source_embeds is None:
            # Original mode: use closed-form, no error correction
            latent_trajectory = [xts[i].clone() for i in range(num_inference_steps + 1)]
            z_T = latent_trajectory[-1]
            return z_T, latent_trajectory, zs
        
        # Edit-friendly inversion with error correction (matching reference)
        # Use empty prompt for unconditional predictions during forward pass
        uncond_embeds = self.empty_prompt_embeds.repeat(batch_size, 1, 1)
        
        for t in timesteps:
            idx = num_inference_steps - t_to_idx[int(t)] - 1
            
            # Get x_{t+1} from pre-computed trajectory
            xt = xts[idx + 1].clone()
            
            # Predict noise using UNet with source text conditioning
            noise_pred = self.unet(
                xt,
                t,
                encoder_hidden_states=uncond_embeds,  # Use empty for inversion
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
        target_embeds: torch.Tensor,
        zs: Optional[torch.Tensor] = None,
        skip: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        eta: float = 1.0,
    ) -> torch.Tensor:
        """
        Denoise starting from timestep T-skip using target text embeddings.
        
        Args:
            latent_trajectory: [z_0, z_1, ..., z_T] from forward diffusion
            target_embeds: [B, 128, 768] target text embeddings
            zs: [T, B, 4, H/8, W/8] normalized noise vectors from forward pass (for error correction)
            skip: Number of timesteps to skip
            num_inference_steps: Total number of timesteps
            guidance_scale: CFG scale
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
        
        # Prepare embeddings for CFG
        empty_embeddings = self.empty_prompt_embeds.repeat(batch_size, 1, 1)
        
        # Denoising loop with error correction
        for t in active_timesteps:
            idx = num_inference_steps - t_to_idx[int(t)] - (num_inference_steps - len(active_timesteps) + 1)
            
            if guidance_scale > 1.0:
                # Classifier-free guidance
                latent_input = torch.cat([latents] * 2)
                text_input = torch.cat([empty_embeddings, target_embeds])
                
                noise_pred = self.unet(
                    latent_input,
                    t,
                    encoder_hidden_states=text_input,
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
                    encoder_hidden_states=target_embeds,
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
        Generate counterfactual images for a batch using DDPM Edit Friendly inversion with text.
        
        Args:
            batch: Dictionary from dataloader containing:
                - 'image': torch.Tensor [B, 3, H, W] in [-1, 1]
                - 'text_prompt': List[str] - source text prompts
                - 'labels': Optional[List[dict]] - ground truth pathology labels
            target_pathology: Target pathology to modify
            
        Returns:
            Dictionary containing counterfactual images and metadata
        """
        self.validate_batch(batch)
        self.validate_pathology(target_pathology)
        
        if 'text_prompt' not in batch:
            raise ValueError(
                "Batch must contain 'text_prompt' key for text-based generation."
            )
        
        original_images_tensor = batch['image'].to(self.device)
        source_prompts = batch['text_prompt']
        labels_list = batch.get('labels', None)  # Optional ground truth labels
        
        batch_size = original_images_tensor.shape[0]
        
        # Determine direction for each sample and construct target prompts
        directions = []
        target_prompts = []
        edit_stats = {'added': 0, 'removed': 0}
        
        for i, source_text in enumerate(source_prompts):
            # Use labels if available, otherwise fall back to keyword detection
            labels = labels_list[i] if labels_list else None
            direction = self._determine_direction(target_pathology, labels=labels, source_text=source_text)
            
            if direction == "decrease":
                edit_stats['removed'] += 1
            else:
                edit_stats['added'] += 1
            
            directions.append(direction)
            target_prompt = self._construct_target_prompt(source_text, target_pathology, direction)
            target_prompts.append(target_prompt)
        
        # Encode source and target prompts
        source_embeds = self._encode_text(source_prompts)
        target_embeds = self._encode_text(target_prompts)
        
        # Step 1: DDPM forward diffusion with error correction
        # Use source text embeddings for inversion (to compute zs for reconstruction)
        logger.debug(f"DDPM forward diffusion for {batch_size} images...")
        z_T, latent_trajectory, zs = self.ddpm_forward_diffusion(
            original_images_tensor,
            source_embeds,  # Use source embedding for inversion
        )
        
        # Step 2a: Reconstruct with source text (for inversion fidelity measurement)
        # Use stored zs for perfect reconstruction
        logger.debug(f"Reconstructing with source text (skip={self.skip})...")
        reconstructed_tensor = self.edit_friendly_denoise(
            latent_trajectory,
            source_embeds,
            zs=zs,  # Use stored noise for reconstruction
        )
        
        # Step 2b: Edit-friendly denoising with target text
        # Use stored zs to maintain structure while changing semantics
        logger.debug(f"Edit-friendly denoising with target text (skip={self.skip})...")
        counterfactual_tensor = self.edit_friendly_denoise(
            latent_trajectory,
            target_embeds,
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
                'method': 'text_ddpm_fixed' if self.use_independent_noise else 'text_ddpm_edit_friendly',
                'template_set': self.template_set,
                'edit_stats': edit_stats,
                'source_prompts': source_prompts,
                'target_prompts': target_prompts,
                'num_inference_steps': self.num_inference_steps,
                'skip': self.skip,
                'guidance_scale': self.guidance_scale,
                'use_independent_noise': self.use_independent_noise,
            }
        }

