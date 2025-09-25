# -*- coding: utf-8 -*-
import argparse
import sys
import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config.settings import EvaluationConfig
from src.data.dataset_loader import get_dataset_files, load_dataset_for_task
from src.models.model_loader import load_model_and_tokenizer
from src.evaluation.prompt_formatters import get_prompt_formatter
from src.evaluation.evaluator import Evaluator
from src.utils.logging import setup_logging

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5 models with vLLM on chemical reaction tasks")
    parser.add_argument('--cuda_device', type=str, default="1", help="CUDA device number(s) to use (e.g., '6' or '0,1').")
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="Model name or path")
    parser.add_argument('--tasks', type=str, default="retrosynthesis",
                       help="Comma-separated list of tasks to evaluate")
    parser.add_argument('--retrosynthesis_dataset', type=str, default="test_50K.jsonl",
                       help="Dataset file path for retrosynthesis")
    parser.add_argument('--retrosynthesis_class_dataset', type=str, default="test_typed.jsonl",
                       help="Dataset file path for retrosynthesis_class")
    parser.add_argument('--forward_prediction_dataset', type=str, default="test_480K.jsonl",
                       help="Dataset file path for forward_prediction")
    parser.add_argument('--prompt_type', type=str, default='1-plan',
                       help="System prompt type from available prompt types")
    parser.add_argument('--n', type=int, default=10,
                       help="Number of top sequences to return after deduplication and reranking")
    parser.add_argument('--generate_n', type=int, default=50,
                       help="Number of initial sequences to generate before deduplication and reranking")
    parser.add_argument('--top_k_metrics', type=str, default='1,3,5,10',
                       help="Comma-separated list of top-k accuracies to report")
    parser.add_argument('--batch_size', type=int, default=100000,
                       help="Batch size for vLLM inference")
    parser.add_argument('--temperature', type=float, default=1.3,
                       help="temperature")
    parser.add_argument('--top_p', type=float, default=0.98,
                       help="Top-p sampling parameter")
    parser.add_argument('--top_k', type=int, default=100,
                       help="Top-k sampling parameter")
    parser.add_argument('--disable_logging_csv', action='store_true',
                       help='Disable logging to file and CSV output.')

    args = parser.parse_args()
    
    try:
        config = EvaluationConfig(
            cuda_device=args.cuda_device,
            model_name=args.model_name,
            tasks=args.tasks,
            retrosynthesis_dataset=args.retrosynthesis_dataset,
            retrosynthesis_class_dataset=args.retrosynthesis_class_dataset,
            forward_prediction_dataset=args.forward_prediction_dataset,
            prompt_type=args.prompt_type,
            n=args.n,
            generate_n=args.generate_n,
            top_k_metrics=args.top_k_metrics,
            batch_size=args.batch_size,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            disable_logging_csv=args.disable_logging_csv
        )
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Get dataset files mapping
    dataset_files = get_dataset_files(config)
    
    # Initialize evaluator
    evaluator = Evaluator(model, tokenizer, config)
    
    # Run evaluation for each selected task
    for task in config.selected_tasks:
        dataset_file = dataset_files[task]
        dataset_name = dataset_file.replace('.jsonl', '')
        
        # Setup logging
        logger, log_file = setup_logging(task, config.model_name, dataset_name, config, config.disable_logging_csv)
        
        try:
            # Load dataset
            dataset = load_dataset_for_task(task, dataset_file)
            
            # Prepare dataset with task-specific prompt
            format_func = get_prompt_formatter(task)
            if task in ['retrosynthesis', 'retrosynthesis_class']:
                eval_dataset = dataset.map(lambda ex: format_func(ex, task=task, prompt_type=config.prompt_type, dataset_file=dataset_file), batched=False)
            else:
                eval_dataset = dataset.map(lambda ex: format_func(ex, prompt_type=config.prompt_type, dataset_file=dataset_file), batched=False)

            # Run evaluation
            results = evaluator.evaluate_dataset(
                task=task,
                dataset=eval_dataset,
                dataset_name=dataset_name,
                prompt_type=config.prompt_type,
                batch_size=config.batch_size,
                logger=logger
            )
            
            if not config.disable_logging_csv:
                print(f"Full log for {task} saved to {log_file}")
                if logger:
                    logger.info(f"Full log for {task} saved to {log_file}")
            else:
                print("Logging and CSV saving are disabled.")
                
        except Exception as e:
            print(f"Error evaluating task {task}: {e}")
            if logger:
                logger.error(f"Error evaluating task {task}: {e}")
            continue

if __name__ == "__main__":
    main()