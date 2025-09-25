# -*- coding: utf-8 -*-
import argparse
import os
import torch
from datasets import Dataset
from src.train.config import TrainingConfig, QWEN_MODELS, DATASET_BASES
from src.train.dataset import load_datasets, prepare_datasets, combine_and_shuffle_datasets
from src.train.model import load_model_and_tokenizer, apply_lora, setup_chat_template, create_trainer, save_model
from src.train.prompts import format_functions
from src.train.utils import extract_response

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune model for chemical reaction tasks")
    parser.add_argument('--prompt_style', type=str, default='with_plan', choices=['with_plan', 'without_plan'],
                       help="System prompt style: 'with_plan' or 'without_plan'")
    parser.add_argument('--tasks', type=str, default='retrosynthesis,retrosynthesis_class,forward_prediction',
                       help="Comma-separated list of tasks to train on")
    parser.add_argument('--dataset_variants', type=str, default='unmapped',
                       help="Comma-separated dataset variants")
    parser.add_argument('--retro_dataset', type=str, default='_50K',
                       help="Retrosynthesis dataset suffix")
    parser.add_argument('--forward_dataset', type=str, default='_480K',
                       help="Forward prediction dataset suffix")
    parser.add_argument('--cuda_device', type=str, default="1",
                       help="CUDA device number(s) to use")
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="Model name or path")
    args = parser.parse_args()
    
    # Process selected tasks
    args.selected_tasks = [task.strip() for task in args.tasks.split(',') if
                         task.strip() in ['retrosynthesis', 'retrosynthesis_class', 'forward_prediction']]
    
    # Process dataset variants
    args.dataset_variants = [v.strip() for v in args.dataset_variants.split(',') if 
                           v.strip() in ['mapped', 'unmapped0', 'unmappedsmile', 'unmappedraw']]
    
    if not args.dataset_variants:
        raise ValueError("At least one dataset variant must be specified.")
    
    return args

def main():
    args = parse_args()
    config = TrainingConfig()
    
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    
    # Load datasets
    train_datasets, val_datasets = load_datasets(args)
    
    # Prepare datasets with appropriate formatting
    prepared_train, prepared_val = prepare_datasets(train_datasets, val_datasets, args.selected_tasks)
    
    # Combine and shuffle datasets for multi-task training
    combined_train, combined_val = combine_and_shuffle_datasets(prepared_train, prepared_val)
    
    # Load model and tokenizer
    model, tokenizer, _ = load_model_and_tokenizer(args.model_name, config.max_seq_length)
    
    # Apply LoRA
    model = apply_lora(model, config)
    
    # Setup chat template
    tokenizer = setup_chat_template(tokenizer)
    
    # Apply chat template to datasets
    def apply_template(example):
        return {
            "text": tokenizer.apply_chat_template(example["conversations"], tokenize=False, add_generation_prompt=False),
            "response": extract_response(example["conversations"][2]["content"])
        }
    
    combined_train = combined_train.map(apply_template, batched=False)
    combined_val = combined_val.map(apply_template, batched=False)
    
    # Create and run trainer
    trainer = create_trainer(model, tokenizer, combined_train, combined_val, args, config)
    trainer_stats = trainer.train()
    
    # Save model
    saved_path = save_model(model, tokenizer, args, config)
    
    print(f"Training completed. LoRA saved to loras/{saved_path}")
    print(f"Merged 16-bit model saved to merged_models/{saved_path}")

if __name__ == "__main__":
    main()