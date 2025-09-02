import os
import json
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# LLaMA Factory imports
from llamafactory.api import ChatModel
from llamafactory.chat import ChatModel as LlamaChatModel
from llamafactory.data import get_dataset
from llamafactory.extras.constants import TRAINING_STAGES
from llamafactory.hparams import get_train_args, get_infer_args
from llamafactory.model import load_model_and_tokenizer
from llamafactory.train.sft import run_sft
from llamafactory.train.tuner import export_model

# Your existing evaluation imports
from sentence_transformers import SentenceTransformer
from evaluate import load as hf_load
import numpy as np

# Import your evaluation functions (assuming they're in a separate file)
# from your_evaluation_module import answer_evaluation_batch, gpt_evaluation

@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning pipeline"""
    # Model settings
    model_name: str = "llama2_7b"  # or your specific model
    template: str = "llama2"

    # Data settings
    train_dataset: str = "train_data"
    valid_dataset: Optional[str] = "valid_data"
    test_dataset: Optional[str] = "test_data"
    dataset_dir: str = "./data"
    
    # Training hyperparameters
    learning_rate: float = 1e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    
    # LoRA settings (for efficient fine-tuning)
    finetuning_type: str = "lora"  # "full" for full fine-tuning
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Output settings
    output_dir: str = "./fine_tuned_models"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    
    # Evaluation settings
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    
    # Export settings
    export_dir: Optional[str] = None
    export_size: int = 2
    export_device: str = "cpu"

class LlamaFactoryFineTuner:
    """Main fine-tuning pipeline class"""
    
    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.setup_logging()
        self.setup_directories()
        
        # Initialize evaluation model
        self.eval_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        self.rouge_metric = hf_load("rouge")
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'fine_tuning_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        """Create necessary directories"""
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.dataset_dir).mkdir(parents=True, exist_ok=True)
        if self.config.export_dir:
            Path(self.config.export_dir).mkdir(parents=True, exist_ok=True)
        
    def prepare_dataset(self, data: List[Dict], dataset_name: str) -> str:
        """
        Prepare dataset in LLaMA Factory format
        Expected input format: [{"instruction": "...", "input": "...", "output": "..."}]
        """
        dataset_path = Path(self.config.dataset_dir) / f"{dataset_name}.json"
        
        # Convert to LLaMA Factory format if needed
        formatted_data = []
        for item in data:
            if "instruction" in item and "output" in item:
                formatted_item = {
                    "instruction": item["instruction"],
                    "input": item.get("input", ""),
                    "output": item["output"]
                }
                formatted_data.append(formatted_item)
            else:
                self.logger.warning(f"Skipping invalid data item: {item}")
                
        with open(dataset_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"Prepared {len(formatted_data)} samples for {dataset_name}")
        return str(dataset_path)
        
    def create_dataset_info(self):
        """Create dataset_info.json for LLaMA Factory"""
        dataset_info = {
            self.config.train_dataset: {
                "file_name": f"{self.config.train_dataset}.json",
                "formatting": "alpaca",
                "columns": {
                    "prompt": "instruction",
                    "query": "input", 
                    "response": "output"
                }
            }
        }
        
        if self.config.valid_dataset:
            dataset_info[self.config.valid_dataset] = {
                "file_name": f"{self.config.valid_dataset}.json",
                "formatting": "alpaca",
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output"
                }
            }
            
        dataset_info_path = Path(self.config.dataset_dir) / "dataset_info.json"
        with open(dataset_info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
            
        return str(dataset_info_path)
        
    def get_training_args(self) -> Dict:
        """Get training arguments for LLaMA Factory"""
        args = {
            "stage": "sft",
            "model_name": self.config.model_name,
            "template": self.config.template,
            "dataset": self.config.train_dataset,
            "dataset_dir": self.config.dataset_dir,
            "finetuning_type": self.config.finetuning_type,
            "output_dir": self.config.output_dir,
            
            # Training hyperparameters
            "learning_rate": self.config.learning_rate,
            "num_train_epochs": self.config.num_train_epochs,
            "per_device_train_batch_size": self.config.per_device_train_batch_size,
            "per_device_eval_batch_size": self.config.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "max_seq_length": self.config.max_seq_length,
            
            # LoRA settings
            "lora_rank": self.config.lora_rank,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            
            # Evaluation and saving
            "evaluation_strategy": self.config.evaluation_strategy,
            "save_strategy": self.config.save_strategy,
            "save_steps": self.config.save_steps,
            "eval_steps": self.config.eval_steps,
            "logging_steps": self.config.logging_steps,
            "load_best_model_at_end": self.config.load_best_model_at_end,
            "metric_for_best_model": self.config.metric_for_best_model,
            
            # Other settings
            "fp16": torch.cuda.is_available(),
            "remove_unused_columns": False,
            "report_to": "none",  # Disable wandb/tensorboard for now
        }
        
        if self.config.valid_dataset:
            args["val_dataset"] = self.config.valid_dataset
            
        return args
        
    def fine_tune(self, train_data: List[Dict], valid_data: Optional[List[Dict]] = None):
        """Main fine-tuning method"""
        self.logger.info("Starting fine-tuning pipeline...")
        
        # Prepare datasets
        self.prepare_dataset(train_data, self.config.train_dataset)
        if valid_data:
            self.prepare_dataset(valid_data, self.config.valid_dataset)
            
        # Create dataset info
        self.create_dataset_info()
        
        # Get training arguments
        train_args = self.get_training_args()
        
        try:
            # Parse arguments using LLaMA Factory's argument parser
            model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(train_args)
            
            # Run fine-tuning
            self.logger.info("Starting SFT training...")
            run_sft(model_args, data_args, training_args, finetuning_args, generating_args)
            
            self.logger.info("Fine-tuning completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Error during fine-tuning: {str(e)}")
            raise
            
    def export_model(self, export_dir: Optional[str] = None):
        """Export fine-tuned model"""
        if not export_dir:
            export_dir = self.config.export_dir or f"{self.config.output_dir}_exported"
            
        self.logger.info(f"Exporting model to {export_dir}")
        
        export_args = {
            "model_name": self.config.model_name,
            "template": self.config.template,
            "finetuning_type": self.config.finetuning_type,
            "checkpoint_dir": self.config.output_dir,
            "export_dir": export_dir,
            "export_size": self.config.export_size,
            "export_device": self.config.export_device,
        }
        
        try:
            model_args, _, _, finetuning_args, _ = get_train_args(export_args)
            export_model(model_args, finetuning_args)
            self.logger.info(f"Model exported successfully to {export_dir}")
            return export_dir
        except Exception as e:
            self.logger.error(f"Error during model export: {str(e)}")
            raise
            
    def load_fine_tuned_model(self, checkpoint_dir: Optional[str] = None):
        """Load fine-tuned model for inference"""
        if not checkpoint_dir:
            checkpoint_dir = self.config.output_dir
            
        infer_args = {
            "model_name": self.config.model_name,
            "template": self.config.template,
            "finetuning_type": self.config.finetuning_type,
            "checkpoint_dir": checkpoint_dir,
        }
        
        try:
            model_args, data_args, _, finetuning_args, generating_args = get_infer_args(infer_args)
            chat_model = LlamaChatModel(model_args, finetuning_args, generating_args)
            return chat_model
        except Exception as e:
            self.logger.error(f"Error loading fine-tuned model: {str(e)}")
            raise
    
    def generate_answers(self, chat_model, test_data: List[Dict]) -> List[str]:
        """Generate answers using fine-tuned model"""
        generated_answers = []
        
        for item in test_data:
            query = item.get("instruction", "")
            if "input" in item and item["input"]:
                query += f"\n{item['input']}"

            try:
                response = chat_model.chat(
                    query=query,
                    history=[],
                    system=""
                )[0].response_text
                generated_answers.append(response)
            except Exception as e:
                self.logger.error(f"Error generating answer for query: {query[:50]}... Error: {str(e)}")
                generated_answers.append("")
                
        return generated_answers
        
    def evaluate_model(self, test_data: List[Dict], checkpoint_dir: Optional[str] = None) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        self.logger.info("Starting model evaluation...")
        
        # Load fine-tuned model
        chat_model = self.load_fine_tuned_model(checkpoint_dir)
        
        # Generate answers
        generated_answers = self.generate_answers(chat_model, test_data)
        original_answers = [item["output"] for item in test_data]
        
        # Calculate similarity scores using your existing evaluation functions
        evaluation_results = self.answer_evaluation_batch(
            generated_answers, 
            original_answers, 
            self.eval_model, 
            self.rouge_metric
        )
        
        # Calculate averages
        avg_scores = {
            "rouge_score": np.mean([r["rouge_score"] for r in evaluation_results]),
            "sbert_score": np.mean([r["sbert_score"] for r in evaluation_results]),
            "bleu_score": np.mean([r["bleu_score"] for r in evaluation_results])
        }
        
        # Log results
        self.logger.info(f"Evaluation Results - ROUGE: {avg_scores['rouge_score']:.4f}, "
                        f"SBERT: {avg_scores['sbert_score']:.4f}, BLEU: {avg_scores['bleu_score']:.4f}")
        
        return {
            "per_item_scores": evaluation_results,
            "average_scores": avg_scores,
            "generated_answers": generated_answers,
            "original_answers": original_answers
        }
        
    def answer_evaluation_batch(self, generated_answers: List[str], original_answers: List[str], 
                               model, rouge_metric) -> List[Dict]:
        """Use your existing batch evaluation function"""
        # Import and use your existing evaluation functions
        from your_evaluation_module import answer_evaluation_batch
        return answer_evaluation_batch(generated_answers, original_answers, model, rouge_metric)
        
    def run_full_pipeline(self, train_data: List[Dict], valid_data: Optional[List[Dict]] = None,
                         test_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Run the complete fine-tuning and evaluation pipeline"""
        results = {}
        
        try:
            # Fine-tune the model
            self.fine_tune(train_data, valid_data)
            results["fine_tuning_status"] = "success"
            
            # Export the model
            export_dir = self.export_model()
            results["export_dir"] = export_dir
            
            # Evaluate if test data provided
            if test_data:
                eval_results = self.evaluate_model(test_data)
                results["evaluation"] = eval_results
                
                # Save evaluation results
                eval_file = Path(self.config.output_dir) / "evaluation_results.json"
                with open(eval_file, 'w') as f:
                    json.dump(eval_results, f, indent=2, default=str)
                    
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            results["error"] = str(e)
            return results


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning pipeline with LLaMA Factory")

    # Model and template
    parser.add_argument("--model_name", type=str, default="llama2_7b", help="Base model name")
    parser.add_argument("--template", type=str, default="llama2", help="Model template")

    # Dataset configs
    parser.add_argument("--train_dataset", type=str, default="my_train_data")
    parser.add_argument("--valid_dataset", type=str, default="my_valid_data")
    parser.add_argument("--test_dataset", type=str, default="my_test_data")
    parser.add_argument("--dataset_dir", type=str, default="./data")

    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=2048)

    # LoRA settings
    parser.add_argument("--finetuning_type", type=str, default="lora", choices=["lora", "full"])
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    # Output settings
    parser.add_argument("--output_dir", type=str, default="./fine_tuned_llama")
    parser.add_argument("--export_dir", type=str, default="./exported_llama")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=100)

    return parser.parse_args()


# Example usage
def main():
    args = parse_args()

    # Map args into config dataclass
    config = FineTuningConfig(
        model_name=args.model_name,
        template=args.template,
        train_dataset=args.train_dataset,
        valid_dataset=args.valid_dataset,
        test_dataset=args.test_dataset,
        dataset_dir=args.dataset_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_length=args.max_seq_length,
        finetuning_type=args.finetuning_type,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        output_dir=args.output_dir,
        export_dir=args.export_dir,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
    )

    # Load some toy example data (replace with your own loader)
    train_data = [
        {"instruction": "Answer the following question:", "input": "What is the capital of France?", "output": "Paris."}
    ]
    valid_data = [
        {"instruction": "Answer the following question:", "input": "What is the capital of Germany?", "output": "Berlin."}
    ]
    test_data = [
        {"instruction": "Answer the following question:", "input": "What is the capital of Italy?", "output": "Rome."}
    ]

    fine_tuner = LlamaFactoryFineTuner(config)
    results = fine_tuner.run_full_pipeline(train_data, valid_data, test_data)

    print("Fine-tuning pipeline completed!")
    print(json.dumps(results, indent=2, default=str))

if __name__ == "__main__":
    main()