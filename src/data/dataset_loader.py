# -*- coding: utf-8 -*-
from datasets import load_dataset
from typing import Dict, List
from ..config.settings import EvaluationConfig

def get_dataset_files(config: EvaluationConfig) -> Dict[str, str]:
    return {
        'retrosynthesis': config.retrosynthesis_dataset,
        'retrosynthesis_class': config.retrosynthesis_class_dataset,
        'forward_prediction': config.forward_prediction_dataset,
    }

def load_dataset_for_task(task: str, dataset_file: str):
    if not dataset_file.endswith('.jsonl'):
        raise ValueError(f"Dataset file for {task} must end with '.jsonl'")
    
    dataset_path = f"data/{dataset_file}"
    return load_dataset("json", data_files=dataset_path)["train"]