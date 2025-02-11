"""FID Score calculation for Remote Sensing Image Generation"""
import random
import torch
import torchmetrics
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import functional as F
from diffusers import DiffusionPipeline
import os
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple
from utils.constants import MODEL_PATH
from utils.logger import setup_logger

logger = setup_logger(
    "metrics",
    Path("logs/metrics.log")
)

class FIDCalculator:
    def __init__(
        self, 
        captions_file: Path,
        dataset_path: Path,
        sample_size: int = 100,
        image_size: Tuple[int, int] = (512, 512)
    ):
        self.captions_file = captions_file
        self.dataset_path = dataset_path
        self.sample_size = sample_size
        self.image_size = image_size
        self.pipeline = None
        self.cap_dict = {}
        
    def load_model(self):
        """Load the diffusion model."""
        logger.info("Loading diffusion model...")
        self.pipeline = DiffusionPipeline.from_pretrained(MODEL_PATH)
        self.pipeline.to('cuda')
        logger.info("Model loaded successfully")
        
    def load_captions(self) -> Dict[str, str]:
        """Load and sample image captions."""
        logger.info(f"Loading captions from {self.captions_file}")
        try:
            with open(self.captions_file, 'r') as file:
                data = file.readlines()
            
            # Extract image paths and captions
            image_captions = [line.strip().split(': ') for line in data]
            
            # Sample data
            sampled_data = random.sample(image_captions, self.sample_size)
            
            # Create caption dictionary
            for image_path, captions in sampled_data:
                self.cap_dict[image_path] = eval(captions)[0]
                
            logger.info(f"Loaded {len(self.cap_dict)} image-caption pairs")
            return self.cap_dict
            
        except Exception as e:
            logger.error(f"Error loading captions: {e}")
            raise
            
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for FID calculation."""
        image = torch.tensor(image).unsqueeze(0)
        image = image.permute(0, 3, 1, 2) / 255.0
        return F.center_crop(image, self.image_size)
    
    def load_real_images(self) -> torch.Tensor:
        """Load and preprocess real images."""
        logger.info("Loading real images...")
        try:
            image_paths = sorted([
                os.path.join(self.dataset_path, x) 
                for x in self.cap_dict.keys()
            ])
            real_images = [
                np.array(Image.open(path).convert("RGB")) 
                for path in image_paths
            ]
            processed_images = torch.cat([
                self.preprocess_image(image) 
                for image in real_images
            ])
            logger.info(f"Loaded {len(real_images)} real images")
            return processed_images
        except Exception as e:
            logger.error(f"Error loading real images: {e}")
            raise
            
    def generate_fake_images(self) -> torch.Tensor:
        """Generate and preprocess fake images."""
        logger.info("Generating fake images...")
        try:
            fake_images = [
                np.array(self.pipeline(x).images[0]) 
                for x in self.cap_dict.values()
            ]
            processed_images = torch.cat([
                self.preprocess_image(image) 
                for image in fake_images
            ])
            logger.info(f"Generated {len(fake_images)} fake images")
            return processed_images
        except Exception as e:
            logger.error(f"Error generating fake images: {e}")
            raise
            
    def calculate_fid(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
        """Calculate FID score."""
        logger.info("Calculating FID score...")
        try:
            fid = FrechetInceptionDistance(normalize=True)
            fid.update(real_images, real=True)
            fid.update(fake_images, real=False)
            score = float(fid.compute())
            logger.info(f"FID Score: {score}")
            return score
        except Exception as e:
            logger.error(f"Error calculating FID score: {e}")
            raise

def main():
    # Initialize calculator
    calculator = FIDCalculator(
        captions_file=Path('data/captions.txt'),
        dataset_path=Path('data/images'),
        sample_size=100
    )
    
    try:
        # Run FID calculation pipeline
        calculator.load_model()
        calculator.load_captions()
        real_images = calculator.load_real_images()
        fake_images = calculator.generate_fake_images()
        fid_score = calculator.calculate_fid(real_images, fake_images)
        
        logger.info("FID calculation completed successfully")
        return fid_score
        
    except Exception as e:
        logger.error(f"FID calculation failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
