"""Module for converting caption text files to JSONL format."""

import json
import logging
from pathlib import Path
from typing import Dict, Any

from utils.constants import CAPTIONS_FILE, METADATA_FILE, LogMessages

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_line(line: str) -> Dict[str, Any]:
    """Parse a single line from the input file.
    
    Args:
        line: Input line containing filename and captions
        
    Returns:
        Dictionary with file_name and text fields
    """
    parts = line.strip().split(': ')
    file_name = Path(parts[0]).name  # Extract filename from path
    captions = eval(parts[1])  # Convert string to list
    
    return {
        "file_name": file_name,
        "text": " ".join(captions)
    }


def convert_text_to_jsonl(input_file: str, output_file: str) -> None:
    """Convert text file with captions to JSONL format.
    
    Args:
        input_file: Path to input text file
        output_file: Path to output JSONL file
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    logger.info(f"Starting conversion from {input_path} to {output_path}")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as infile:
            total_lines = sum(1 for _ in infile)
        
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for idx, line in enumerate(infile, 1):
                try:
                    json_data = parse_line(line)
                    json_line = json.dumps(json_data, ensure_ascii=False)
                    outfile.write(json_line + '\n')
                    
                    if idx % 1000 == 0:  # Log progress every 1000 items
                        logger.info(f"Processed {idx}/{total_lines} lines")
                        
                except Exception as e:
                    logger.error(LogMessages.PROCESSING_ERROR.format(idx, str(e)))
                    continue
            
        logger.info(f"Successfully converted {total_lines} lines to JSONL format")
        
    except Exception as e:
        logger.error(f"An error occurred during conversion: {str(e)}", exc_info=True)
        raise


def main():
    """Main function to handle the conversion process."""
    try:
        logger.info(LogMessages.STARTING_PIPELINE)
        convert_text_to_jsonl(CAPTIONS_FILE, METADATA_FILE)
        logger.info(LogMessages.PIPELINE_COMPLETE)
        
    except Exception as e:
        logger.error(LogMessages.PROCESSING_ERROR.format("conversion", str(e)))
        raise


if __name__ == "__main__":
    main()
