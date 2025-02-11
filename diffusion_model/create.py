"""Module for processing Parquet files and extracting images with captions."""

import io
import logging
from pathlib import Path
from typing import Tuple

import pyarrow.parquet as pq
from PIL import Image

from utils.constants import (
    RSICD_IMAGES_DIR,
    CAPTIONS_FILE,
    LogMessages,
    PARQUET_FILE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_output_directory(path: str) -> Path:
    """Create and return output directory if it doesn't exist.
    
    Args:
        path: Directory path to create
        
    Returns:
        Path object for the created directory
    """
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory at: {output_dir}")
    return output_dir


def process_image_data(image_data: bytes) -> Image.Image:
    """Convert binary image data to PIL Image.
    
    Args:
        image_data: Raw bytes of the image
        
    Returns:
        PIL Image object
    """
    return Image.open(io.BytesIO(image_data))


def extract_parquet_data(file_path: str) -> Tuple[pq.Table, pq.Table, pq.Table]:
    """Extract relevant columns from Parquet file.
    
    Args:
        file_path: Path to the Parquet file
        
    Returns:
        Tuple of tables containing filenames, captions and images
    """
    table = pq.read_table(file_path)
    return (
        table.column('filename'),
        table.column('captions'),
        table.column('image')
    )


def save_images_and_captions(
    images_dir: Path,
    captions_file: Path,
    filename_column: pq.Table,
    caption_column: pq.Table,
    image_column: pq.Table
) -> None:
    """Save images and create caption file from Parquet data.
    
    Args:
        images_dir: Directory to save images
        captions_file: File to save captions
        filename_column: Table column with filenames
        caption_column: Table column with captions
        image_column: Table column with image data
    """
    total_items = len(filename_column)
    
    logger.info(f"Starting to process {total_items} items")
    
    with open(captions_file, 'w', encoding='utf-8') as text_file:
        for idx, (filename, caption, image_data_struct) in enumerate(zip(
            filename_column,
            caption_column,
            image_column
        ), 1):
            filename = filename.as_py()
            caption = caption.as_py()
            
            # Extract and process image
            image_data = image_data_struct['bytes'].as_py()
            image = process_image_data(image_data)
            
            # Save image
            image_path = images_dir / filename
            image.save(image_path)
            
            # Write caption
            text_file.write(f"{filename}: {caption}\n")
            
            if idx % 100 == 0:  # Log progress every 100 items
                logger.info(f"Processed {idx}/{total_items} items")

    logger.info("Completed processing all items")


def main():
    """Main function to process Parquet file and save outputs."""
    try:
        logger.info(LogMessages.STARTING_PIPELINE)
        
        # Create directories
        images_dir = create_output_directory(RSICD_IMAGES_DIR)
        captions_file = Path(CAPTIONS_FILE)
        captions_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Reading parquet file: {PARQUET_FILE}")
        filename_col, caption_col, image_col = extract_parquet_data(PARQUET_FILE)
        
        save_images_and_captions(
            images_dir,
            captions_file,
            filename_col,
            caption_col,
            image_col
        )
        
        logger.info(LogMessages.SAVE_SUCCESS.format(CAPTIONS_FILE))
        logger.info(f"Images saved to: {images_dir}")
        logger.info(LogMessages.PIPELINE_COMPLETE)
        
    except Exception as e:
        logger.error(LogMessages.PROCESSING_ERROR.format("parquet processing", str(e)))
        raise


if __name__ == '__main__':
    main()
