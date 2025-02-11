"""Module for downstream evaluation of RemoteDiff model using LULC dataset."""

import logging
from pathlib import Path
from typing import Optional

import albumentations
import numpy as np
import pytorch_lightning as pl
import torch
from datasets import load_dataset
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader
from torchgeo.models import ResNet18_Weights
from torchgeo.trainers import ClassificationTask

from utils.constants import DOWNSTREAM_CONFIG, LogMessages

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CustomDataModule(pl.LightningDataModule):
    """Custom DataModule for handling LULC dataset."""
    
    def __init__(
        self,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader
    ):
        """Initialize the data module.
        
        Args:
            train_dataloader: DataLoader for training data
            test_dataloader: DataLoader for testing data
        """
        super().__init__()
        self.prepare_data_per_node = False
        self.train_dataloader_obj = train_dataloader
        self.test_dataloader_obj = test_dataloader

    def train_dataloader(self) -> DataLoader:
        """Get the training data loader."""
        return self.train_dataloader_obj

    def val_dataloader(self) -> DataLoader:
        """Get the validation data loader."""
        return self.test_dataloader_obj

    def test_dataloader(self) -> DataLoader:
        """Get the test data loader."""
        return self.test_dataloader_obj


def get_transform() -> albumentations.Compose:
    """Create image transformation pipeline.
    
    Returns:
        Composed transformation pipeline
    """
    return albumentations.Compose([
        albumentations.RandomCrop(
            width=DOWNSTREAM_CONFIG["image_size"],
            height=DOWNSTREAM_CONFIG["image_size"]
        ),
        albumentations.Normalize(
            mean=DOWNSTREAM_CONFIG["normalize_mean"],
            std=DOWNSTREAM_CONFIG["normalize_std"]
        ),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightnessContrast(p=0.2),
    ])


def transform_examples(examples: dict) -> dict:
    """Apply transformations to dataset examples.
    
    Args:
        examples: Dictionary containing dataset examples
        
    Returns:
        Transformed examples
    """
    transform = get_transform()
    examples["image"] = [
        transform(image=np.array(image))["image"]
        for image in examples["image"]
    ]
    examples['image'] = torch.tensor(examples['image']).permute(0, 3, 1, 2)
    return examples


def prepare_dataset():
    """Prepare and split the LULC dataset.
    
    Returns:
        Tuple of train and test dataloaders
    """
    # Load and split dataset
    dataset = load_dataset("[Insert Hugging Face DatasetPath]")
    dataset = dataset['train'].train_test_split(
        test_size=0.1,
        seed=DOWNSTREAM_CONFIG["random_seed"]
    )
    
    # Apply transformations
    dataset.set_transform(transform_examples)
    
    # # Convert to PyTorch format
    # dataset_train = dataset['train'].with_format("torch")
    # dataset_test = dataset['test'].with_format("torch")
    
    # Create dataloaders
    train_dataloader = DataLoader(
        dataset['train'],
        batch_size=DOWNSTREAM_CONFIG["batch_size"],
        num_workers=DOWNSTREAM_CONFIG["num_workers"],
        shuffle=True
    )
    test_dataloader = DataLoader(
        dataset['test'],
        batch_size=DOWNSTREAM_CONFIG["batch_size"],
        num_workers=DOWNSTREAM_CONFIG["num_workers"]
    )
    
    return train_dataloader, test_dataloader


def setup_trainer(
    fast_dev_run: bool = False,
    output_dir: Optional[Path] = None
) -> Trainer:
    """Set up the PyTorch Lightning trainer.
    
    Args:
        fast_dev_run: Whether to run a debug test run
        output_dir: Directory to save outputs
        
    Returns:
        Configured trainer
    """
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    if output_dir is None:
        output_dir = Path.cwd() / "experiments"
    
    return Trainer(
        accelerator=accelerator,
        fast_dev_run=fast_dev_run,
        log_every_n_steps=1,
        min_epochs=1,
        max_epochs=DOWNSTREAM_CONFIG["max_epochs"],
        default_root_dir=output_dir
    )


def main():
    """Main function to run the training pipeline."""
    try:
        logger.info(LogMessages.DOWNSTREAM_START)
        
        # Prepare dataset
        train_dataloader, test_dataloader = prepare_dataset()
        data_module = CustomDataModule(train_dataloader, test_dataloader)
        
        # Setup model
        weights = ResNet18_Weights.SENTINEL2_RGB_MOCO
        task = ClassificationTask(
            model="resnet18",
            loss="ce",
            weights=weights,
            in_channels=3,
            num_classes=DOWNSTREAM_CONFIG["num_classes"],
            lr=DOWNSTREAM_CONFIG["learning_rate"],
            patience=DOWNSTREAM_CONFIG["patience"],
        )
        
        # Train model
        trainer = setup_trainer()
        trainer.fit(
            model=task,
            train_dataloaders=data_module.train_dataloader(),
            val_dataloaders=data_module.test_dataloader()
        )
        
        # Test model
        test_results = trainer.test(
            model=task,
            dataloaders=data_module.test_dataloader()
        )
        logger.info(f"Test results: {test_results}")
        logger.info(LogMessages.DOWNSTREAM_COMPLETE)
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

