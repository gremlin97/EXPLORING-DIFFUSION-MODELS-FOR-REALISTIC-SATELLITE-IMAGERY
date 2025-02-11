# Model paths
MODEL_PATH = ""
BASE_MODEL_PATH = "microsoft/phi-1_5"
DIFFUSION_MODEL_PATH = "[Insert Path]"

# File paths
DEFAULT_CSV_PATH = "papers_list.csv"
DEFAULT_PDF_DIR = "598_papers"
DEFAULT_OUTPUT_FILENAME = "remote_sensing_data.txt"
CAPTIONS_FILE = "data/captions.txt"
METADATA_FILE = "data/metadata.jsonl"
MODEL_OUTPUT_DIR = "model_outputs"
HUB_MODEL_NAME = "remote_sensing_gpt"
IMAGES_DIR = "data/images"
RSICD_IMAGES_DIR = "data/rsicd_images"

# URLs
PAPERS_DOWNLOAD_URL = ""

# Metric parameters
FID_SAMPLE_SIZE = 100
IMAGE_SIZE = (512, 512)

# Model parameters
BLOCK_SIZE = 256
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_TRAIN_EPOCHS = 5
TEST_SIZE = 0.1

# Text Processing
CHUNK_SIZE = 512
TEST_SPLIT_SIZE = 0.11
CONTEXT_LENGTH = 512

# Downstream Evaluation Configuration
DOWNSTREAM_CONFIG = {
    "batch_size": 8,
    "num_workers": 2,
    "max_epochs": 35,
    "learning_rate": 0.001,
    "num_classes": 7,
    "image_size": 512,
    "random_seed": 42,
    "patience": 5,
    "normalize_mean": (0.485, 0.456, 0.406),
    "normalize_std": (0.229, 0.224, 0.225)
}

# Training Configuration
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 8,
    "target_modules": ["Wqkv", "out_proj", "fc1", "fc2"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

# Inference Configuration
INFERENCE_CONFIG = {
    "num_inference_steps": 100,
    "guidance_scale": 7.5
}

# Prompts
NEGATIVE_PROMPT = "weird colors, low quality, jpeg artifacts, lowres, grainy, deformed structures, blurry, opaque, low contrast, distorted details, details are low"
QUALITY_PROMPT_SUFFIX = " , 8k, best quality, high-resolution"

# UI Configuration
UI_CONFIG = {
    "title": "RemoteDifusion",
    "description": "Stable Diffusion for Remote Sensing!"
}

# Add these to the existing constants
PARQUET_FILE = '/data/train.parquet'

# LULC Classes
LULC_CLASSES = [
    'Water body',
    'Bare ground',
    'Woody Vegetation',
    'Cultivated Vegetation',
    'Crop Land',
    'Natural Vegetation',
    'Snow/Ice'
]

# Generation Configuration
GENERATION_CONFIG = {
    "max_new_tokens": 256,
    "temperature": 0.2,
    "repetition_penalty": 1.1,
    "num_images_per_class": 50
}

# Natural Vegetation Prompts (example prompts - you can add more)
NATURAL_VEGETATION_PROMPTS = [
    "A satellite image showing a dense tropical rainforest canopy with diverse tree species.",
    "An aerial view of a temperate deciduous forest ablaze with autumn colors.",
    # ... add other prompts as needed
]

# Diffusion Model Training Configuration
DIFFUSION_TRAINING = {
    "model_name": "runwayml/stable-diffusion-v1-5",
    "data_dir": "/data/rsicd_images",
    "output_dir": "/data/output_directory/RemoteDiff"
}

# Log messages
class LogMessages:
    STARTING_PIPELINE = "Starting paper processing pipeline..."
    DOWNLOADING_PAPERS = "Downloading paper list..."
    EXTRACTING_PAPERS = "Extracting papers..."
    DOWNLOAD_SUCCESS = "Successfully downloaded and extracted papers"
    DOWNLOAD_ERROR = "Error during download or extraction: {}"
    FILE_NOT_FOUND = "File not found at {}"
    PROCESSING_PDFS = "Processing {} PDF files..."
    PROCESSED_FILE = "Processed {}"
    PROCESSING_ERROR = "Error processing {}: {}"
    SAVE_SUCCESS = "Text has been saved to {}"
    PIPELINE_COMPLETE = "Pipeline completed successfully"
    MODEL_TRAINING_START = "Starting model training..."
    MODEL_TRAINING_COMPLETE = "Model training completed"
    EVAL_RESULTS = "Evaluation Results - Perplexity: {:.2f}"
    CONVERSION_START = "Starting text to JSONL conversion..."
    CONVERSION_COMPLETE = "JSONL conversion completed successfully"
    PROCESSING_LINE_ERROR = "Error processing line {}: {}"
    DOWNSTREAM_START = "Starting downstream evaluation..."
    DOWNSTREAM_COMPLETE = "Downstream evaluation completed"
