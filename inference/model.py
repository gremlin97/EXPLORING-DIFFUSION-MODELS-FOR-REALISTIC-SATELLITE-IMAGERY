"""Inference module for Remote Sensing Image Generation"""
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from pathlib import Path
from utils.constants import (
    DIFFUSION_MODEL_PATH,
    INFERENCE_CONFIG,
    NEGATIVE_PROMPT,
    QUALITY_PROMPT_SUFFIX
)
from utils.logger import setup_logger
    
logger = setup_logger(
    "inference",
    Path("logs/inference.log")
)

class RemoteInferenceModel:
    def __init__(self):
        self.pipe = None
        
    def load_model(self):
        """Load the diffusion model."""
        try:
            logger.info(f"Loading model from {DIFFUSION_MODEL_PATH}")
            self.pipe = StableDiffusionPipeline.from_pretrained(DIFFUSION_MODEL_PATH)
            self.pipe = self.pipe.to('cuda')
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
    def generate_image(self, prompt: str) -> Image.Image:
        """Generate image from prompt."""
        try:
            if not self.pipe:
                self.load_model()
                
            # Enhance prompt
            enhanced_prompt = prompt + QUALITY_PROMPT_SUFFIX
            logger.info(f"Generating image for prompt: {enhanced_prompt}")
            
            # Generate image
            image = self.pipe(
                prompt=enhanced_prompt,
                negative_prompt=NEGATIVE_PROMPT,
                **INFERENCE_CONFIG
            ).images[0]
            
            logger.info("Image generated successfully")
            return image
            
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            raise 