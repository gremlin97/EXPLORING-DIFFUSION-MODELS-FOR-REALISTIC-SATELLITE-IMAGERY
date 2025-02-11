# EXPLORING DIFFUSION MODELS FOR REALISTIC SATELLITE IMAGERY

This repository contains the implementation of a diffusion model-based approach for generating realistic satellite imagery, along with tools for evaluation and downstream applications.

## 🌟 Project Overview

This project explores the application of diffusion models to generate high-quality, realistic satellite imagery. It includes components for:
- Text-to-image generation using diffusion models
- Causal language model training for remote sensing text generation
- Downstream evaluation and data generation
- Metrics calculation and model evaluation
- Interactive inference UI



## 📁 Repository Structure

```
.
├── causal_llm/                  # Causal language model implementation
├── diffusion_model/             # Core diffusion model implementation
├── downstream_data_generation/  # Tools for generating downstream task data
├── downstream_evaluation/       # Evaluation scripts for downstream tasks
├── inference/                   # Model inference and deployment
├── metrics/                     # Evaluation metrics implementation
├── utils/                      # Shared utility functions
├── gradio_ui/                  # Interactive web interface
└── data/                       # Data storage and management
```

## 🛠️ Setup and Installation

### Prerequisites
- Python 3.9 or higher
- Poetry for dependency management
- Replace the constants.py file with your own values
- Download the RSICD dataset from [here](https://rsicd.github.io/rsicd-dataset/) and save the parquet file in the data/RSICD/ directory

### Installation Steps

1. Clone the repository:
```bash
git clone [REPOSITORY_URL]
cd exploring-diffusion-models
```

2. Install dependencies using Poetry:
```bash
poetry install
```

## 📊 Main Components

### 1. Diffusion Model
- Training pipeline for satellite imagery generation
- Custom data loading and preprocessing as per the diffusers documentation
- Image size: 512x512

### 2. Causal Language Model
- Text corpus creation for remote sensing
- Training scripts for language model fine-tuning
- Text generation utilities
- Model parameters:
  - Block size: 256
  - Context length: 512
  - Learning rate: 2e-5
  - Weight decay: 0.01

### 3. Inference
- Model serving infrastructure

### 4. Evaluation
- FID score calculation (Sample size: 100)

### 5. Downstream Evaluation Configuration
- Batch size: 8
- Learning rate: 0.001
- Maximum epochs: 35
- Image size: 512x512
- Early stopping patience: 5
- Normalization parameters:
  - Mean: (0.485, 0.456, 0.406)
  - Std: (0.229, 0.224, 0.225)

## 🔧 Technical Details

### Dependencies
- PyTorch (>=2.0.0)
- Diffusers (^0.25.0)
- Transformers (^4.0.0)
- TorchGeo (0.5.0)
- Gradio (^4.0.0)
- Additional dependencies listed in pyproject.toml