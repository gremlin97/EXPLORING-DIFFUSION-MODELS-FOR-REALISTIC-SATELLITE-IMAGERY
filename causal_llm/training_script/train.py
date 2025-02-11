"""Training script for Remote Sensing Language Model"""
from transformers import AutoTokenizer
import math
from datasets import Dataset, load_dataset
from transformers import DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, LoftQConfig
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from pathlib import Path
from utils.constants import (
    BASE_MODEL_PATH,
    MODEL_OUTPUT_DIR,
    HUB_MODEL_NAME,
    CHUNK_SIZE,
    TEST_SPLIT_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    LORA_CONFIG
)
from utils.logger import setup_logger

logger = setup_logger(
    "training",
    Path("logs/training.log")
)

def break_string_into_lists(text: str, chunk_size: int = CHUNK_SIZE) -> list:
    """Break text into chunks at sentence boundaries."""
    chunks = []
    start_index = 0
    end_index = chunk_size
    
    while start_index < len(text):
        if end_index >= len(text):
            chunks.append(text[start_index:])
            break
        
        nearest_fullstop = text.rfind('.', start_index, end_index)
        if nearest_fullstop != -1:
            chunks.append(text[start_index:nearest_fullstop+1])
            start_index = nearest_fullstop + 1
        else:
            chunks.append(text[start_index:end_index])
            start_index = end_index
        end_index = start_index + chunk_size
    return chunks

def main():
    try:
        # Load dataset
        logger.info("Loading RemoteSensingCorpus dataset")
        dataset = load_dataset("[Insert Hugging Face DatasetPath]")
        concatenated_strings = ' '.join(
            text for data_split in dataset.values() 
            for text in data_split['text']
        )

        # Initialize tokenizer
        logger.info("Initializing tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
        
        # Break text into chunks
        logger.info("Breaking text into chunks")
        broken_lists = break_string_into_lists(concatenated_strings)
        broken_lists = [sublist + tokenizer.eos_token for sublist in broken_lists]

        # Create dataset
        logger.info("Creating dataset")
        dataset_eos = Dataset.from_dict({'text': broken_lists})
        dataset_eos = dataset_eos.train_test_split(
            test_size=TEST_SPLIT_SIZE,
            shuffle=False
        )

        # Tokenize dataset
        logger.info("Tokenizing dataset")
        def tokenize(element):
            return tokenizer(
                element["text"],
                truncation=True,
                max_length=CHUNK_SIZE,
                return_length=True,
            )

        tokenized_datasets = dataset_eos.map(
            tokenize, 
            remove_columns=dataset_eos["train"].column_names
        )

        # Prepare for training
        logger.info("Preparing for training")
        tokenizer.pad_token = tokenizer.eos_token
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=False
        )

        # Initialize model with QLoRA
        logger.info("Initializing model with QLoRA")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH, 
            trust_remote_code=True,
            device_map="auto"
        )

        loftq_config = LoftQConfig(loftq_bits=4)
        lora_config = LoraConfig(
            init_lora_weights="loftq",
            loftq_config=loftq_config,
            **LORA_CONFIG
        )

        model = get_peft_model(model, lora_config)

        # Set up training arguments
        logger.info("Setting up training arguments")
        training_args = TrainingArguments(
            output_dir=MODEL_OUTPUT_DIR,
            evaluation_strategy="epoch",
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            push_to_hub=True,
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['test'],
            data_collator=data_collator,
        )

        # Train model
        logger.info("Starting training")
        trainer.train()

        # Evaluate
        logger.info("Evaluating model")
        eval_results = trainer.evaluate()
        perplexity = math.exp(eval_results['eval_loss'])
        logger.info(f"Perplexity: {perplexity:.2f}")

        # Save model and tokenizer
        logger.info("Saving model and tokenizer")
        trainer.push_to_hub()
        tokenizer.push_to_hub(HUB_MODEL_NAME)

        logger.info("Training completed successfully")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 