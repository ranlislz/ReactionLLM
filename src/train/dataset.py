# -*- coding: utf-8 -*-
from datasets import load_dataset, Dataset, concatenate_datasets
import json
import re
from typing import Dict, List
from src.train.prompts import format_functions

def load_datasets(args):
    """Load and combine datasets for selected tasks and variants."""
    train_datasets = {}
    val_datasets = {}
    
    for task in args.selected_tasks:
        train_list = []
        val_list = []
        for variant in args.dataset_variants:
            suffix = get_dataset_suffix(task, variant, args)
            
            ds = load_dataset("json",
                data_files={
                    "train": f"data/train{suffix}.jsonl", 
                    "test": f"data/validation{suffix}.jsonl"
                }
            )

            def add_variant(example, var=variant):
                example['variant'] = var
                return example

            train_list.append(ds["train"].map(add_variant))
            val_list.append(ds["test"].map(add_variant))

        train_datasets[task] = concatenate_datasets(train_list) if len(train_list) > 1 else train_list[0]
        val_datasets[task] = concatenate_datasets(val_list) if len(val_list) > 1 else val_list[0]
    
    return train_datasets, val_datasets

def get_dataset_suffix(task, variant, args):
    """Get the appropriate dataset suffix based on task and variant."""
    if variant == 'mapped':
        return args.retro_dataset if task == 'retrosynthesis' else args.forward_dataset
    elif variant == 'unmapped0':
        return f"{args.retro_dataset if task == 'retrosynthesis' else args.forward_dataset}_unmapped0"
    elif variant == 'unmappedsmile':
        return f"{args.retro_dataset if task == 'retrosynthesis' else args.forward_dataset}_unmappedsmile"
    elif variant == 'unmappedraw':
        return f"{args.retro_dataset if task == 'retrosynthesis' else args.forward_dataset}_unmappedraw"
    return ""

def prepare_datasets(train_datasets, val_datasets, selected_tasks):
    """Prepare datasets by applying the appropriate formatting functions."""
    prepared_train = {}
    prepared_val = {}
    
    for task in selected_tasks:
        prepared_train[task] = train_datasets[task].map(format_functions[task], batched=False)
        prepared_val[task] = val_datasets[task].map(format_functions[task], batched=False)
    
    return prepared_train, prepared_val

def combine_and_shuffle_datasets(prepared_train, prepared_val, seed=3407):
    """Combine and shuffle datasets for multi-task training."""
    combined_train = concatenate_datasets(list(prepared_train.values())).shuffle(seed=seed)
    combined_val = concatenate_datasets(list(prepared_val.values())).shuffle(seed=seed)
    return combined_train, combined_val

def extract_response(text: str) -> str:
    """Extract the response part from the assistant's answer."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else text