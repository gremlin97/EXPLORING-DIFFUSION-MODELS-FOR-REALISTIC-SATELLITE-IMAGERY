"""Module for generating LULC (Land Use Land Cover) images using RemoteGPT and RemoteDiff."""

import logging
from pathlib import Path
from typing import List

import torch
from diffusers import DiffusionPipeline
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.constants import (
    BASE_MODEL_PATH,
    DIFFUSION_MODEL_PATH,
    HUB_MODEL_NAME,
    LULC_CLASSES,
    GENERATION_CONFIG
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_gpt_model() -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Set up the GPT model and tokenizer.
    
    Returns:
        Tuple of model and tokenizer
    """
    # config = PeftConfig.from_pretrained(HUB_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH)
    model = PeftModel.from_pretrained(model, HUB_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True
    )
    
    return model, tokenizer


def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str
) -> str:
    """Generate text using the GPT model.
    
    Args:
        model: The GPT model
        tokenizer: The tokenizer
        prompt: Input prompt
        
    Returns:
        Generated text
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    tokens = model.generate(
        **inputs,
        max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
        temperature=GENERATION_CONFIG["temperature"],
        eos_token_id=tokenizer.eos_token_id,
        early_stopping=True,
        do_sample=True,
        repetition_penalty=GENERATION_CONFIG["repetition_penalty"]
    )
    
    return tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]


def setup_diffusion_pipeline() -> DiffusionPipeline:
    """Set up the diffusion pipeline.
    
    Returns:
        Configured diffusion pipeline
    """
    pipeline = DiffusionPipeline.from_pretrained(DIFFUSION_MODEL_PATH)
    if torch.cuda.is_available():
        pipeline.to('cuda')
    return pipeline


def generate_images(
    pipeline: DiffusionPipeline,
    prompts: List[str],
    output_dir: Path
) -> None:
    """Generate images from prompts using the diffusion pipeline.
    
    Args:
        pipeline: The diffusion pipeline
        prompts: List of prompts to generate images from
        output_dir: Directory to save generated images
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, prompt in enumerate(prompts, 1):
        try:
            image = pipeline(prompt).images[0]
            image_path = output_dir / f'image_{idx:04d}.png'
            image.save(image_path)
            logger.info(f"Generated image {idx}/{len(prompts)}: {image_path}")
            
        except Exception as e:
            logger.error(f"Error generating image for prompt {idx}: {str(e)}")


def main():
    """Main function to run the generation pipeline."""
    try:
        logger.info("Starting image generation pipeline")
        
        # Setup models
        gpt_model, tokenizer = setup_gpt_model()
        diffusion_pipeline = setup_diffusion_pipeline()
        
        # Set up output directory
        output_dir = Path("generated_images")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate images for each class
        for lulc_class in LULC_CLASSES:
            logger.info(f"Generating images for class: {lulc_class}")
            
            class_dir = output_dir / lulc_class.lower().replace(" ", "_")
            class_dir.mkdir(exist_ok=True)
            
            # Generate prompts using GPT
            prompt_template = f"Create a prompt for generating satellite images for the {lulc_class} class."
            generated_text = generate_text(gpt_model, tokenizer, prompt_template)
            
            # Extract prompts and generate images
            prompts = [generated_text]  # You might want to extract multiple prompts
            generate_images(diffusion_pipeline, prompts, class_dir)
            
        logger.info("Image generation pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

