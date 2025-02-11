# Remote Sensing Diffusion Model training

This repository contains the code for training a Stable Diffusion model on remote sensing imagery with text captions. The pipeline processes a Parquet dataset containing remote sensing images and their captions, prepares it for training, and fine-tunes a pre-trained Stable Diffusion model.

## Installation

1. Clone the Diffusers repository in this directory and install dependencies:
```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
cd examples/text_to_image
pip install -r requirements.txt
```

2. Install additional required packages:
```bash
pip install accelerate diffusers transformers datasets wandb
```

3. Configure Accelerate:
```bash
# Option 1: Interactive configuration
accelerate config

# Option 2: Default configuration
accelerate config default

# Option 3: For notebook environments
from accelerate.utils import write_basic_config
write_basic_config()
```

4. Login to Hugging Face and Weights & Biases:
```bash
huggingface-cli login
wandb login
```

## Project Structure

```
.
├── diffusion_model/
│   ├── create.py        # Processes Parquet files and extracts images with captions
│   └── format.py        # Converts caption text files to JSONL format
├── utils/
│   ├── constants.py     # Contains project constants and configurations
│   └── logger.py        # Logging utilities
├── train_text_to_image.py  # Main training script
├── train.sh            # Training pipeline shell script
└── data/               # Directory for storing datasets
```

## Training Configuration

### Environment Setup

Set the following environment variables:
```bash
export MODEL_NAME="[path to pre-trained model]"
export DATA_DIR="[path to rsicd_images]"
export OUTPUT_DIR="[path to RemoteDiff]"
```

### Training Parameters

The training script (`train.sh`) includes the following optimized parameters:

#### Basic Parameters:
- `--resolution=512`: Input image resolution
- `--train_batch_size=1`: Number of images per batch
- `--validation_prompt="A satellite image of a crop field"`: Prompt for validation

#### Optimization Parameters:
- `--learning_rate=1e-06`: Learning rate for training
- `--gradient_accumulation_steps=4`: Accumulates gradients over 4 steps
- `--max_train_steps=3500`: Total number of training steps
- `--max_grad_norm=1`: Maximum gradient norm for clipping
- `--lr_scheduler="constant"`: Constant learning rate schedule
- `--lr_warmup_steps=0`: No warmup steps

#### Memory Optimization:
- `--mixed_precision="fp16"`: Uses 16-bit floating point precision
- `--gradient_checkpointing`: Enables gradient checkpointing
- `--use_ema`: Uses EMA model averaging

#### Monitoring and Saving:
- `--checkpointing_steps=500`: Save checkpoint every 500 steps
- `--validation_epochs=10`: Run validation every 10 epochs
- `--push_to_hub`: Push model to Hugging Face Hub
- `--report_to=wandb`: Log metrics to Weights & Biases

## Inference Configuration

The trained model uses the following default inference settings:
- Number of inference steps: 100
- Guidance scale: 7.5

## Running Training

1. Make the training script executable:
```bash
chmod +x train.sh
```

2. Start training:
```bash
./train.sh
```

## Model Outputs

The training process produces:
1. Regular model checkpoints (every 500 steps)
2. Validation images with the specified prompt
3. Training metrics in Weights & Biases
4. Final model pushed to Hugging Face Hub

## Additional Resources

- [Diffusers Documentation](https://huggingface.co/docs/diffusers/)
- [Stable Diffusion Training Guide](https://huggingface.co/docs/diffusers/training/text2image)
- [Dataset Creation Guide](https://huggingface.co/docs/datasets/image_dataset)

## Data Preparation Process

### 1. Image and Caption Extraction (create.py)

The first step uses `create.py` to process the training Parquet file and extract images and captions:

```bash
python diffusion_model/create.py \
  --parquet_path /path/to/train.parquet \
  --output_dir /path/to/output_directory \
  --caption_file captions.txt
```

This script will:
- Read the Parquet file containing image data and captions
- Extract and save images to `output_directory/rsicd_images/`
- Create `captions.txt` mapping image filenames to their captions

The `captions.txt` format will be:
```
image_000001.jpg\tThis is a caption for image 1
image_000002.jpg\tThis is a caption for image 2
...
```

### 2. Caption Formatting (format.py)

Next, use `format.py` to convert the caption text file to the required JSONL format:

```bash
python diffusion_model/format.py \
  --caption_file /path/to/captions.txt \
  --output_file metadata.jsonl
```

This script will:
- Read the tab-separated captions file
- Convert each entry to a JSON object
- Create `metadata.jsonl` with the following format:
```json
{"file_name": "image_000001.jpg", "text": "This is a caption for image 1"}
{"file_name": "image_000002.jpg", "text": "This is a caption for image 2"}
```

### Directory Structure After Preparation

After running both scripts, your output directory should look like this:

```
output_directory/
├── rsicd_images/          # Contains 512x512 RGB satellite images
│   ├── image_000001.jpg   # Each image is resized and saved as JPEG
│   ├── image_000002.jpg
│   └── ...
├── captions.txt           # Tab-separated caption file
└── metadata.jsonl         # JSONL format required for training
```

The images in `rsicd_images/` are:
- Resized to 512x512 pixels (matching training resolution)
- Converted to RGB format
- Saved as JPEG files for efficient storage
- Named sequentially (image_000001.jpg, image_000002.jpg, etc.)

This prepared data structure is what the training script expects to find at the path specified in `DATA_DIR`.
