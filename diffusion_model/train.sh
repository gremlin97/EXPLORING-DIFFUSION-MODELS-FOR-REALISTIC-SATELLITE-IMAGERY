#!/bin/bash

# Set environment variables
MODEL_NAME="runwayml/stable-diffusion-v1-5"
DATA_DIR="/output_directory/rsicd_images"
OUTPUT_DIR="/output_directory/RemoteDiff"

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Run training script with accelerate
accelerate launch --mixed_precision="fp16" train_text_to_image.py \
  --pretrained_model_name_or_path=${MODEL_NAME} \
  --train_data_dir=${DATA_DIR} \
  --use_ema \
  --resolution=224 \
  --center_crop \
  --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=3500 \
  --learning_rate=1e-06 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --push_to_hub \
  --checkpointing_steps=500 \
  --validation_prompt="A satellite image of a crop field" \
  --validation_epochs=10 \
  --report_to=wandb 