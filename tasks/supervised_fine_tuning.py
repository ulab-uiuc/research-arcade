import os
import json
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import sys

# LLaMA Factory imports
from llamafactory.chat import ChatModel as LlamaChatModel
from llamafactory.data import get_dataset
from llamafactory.extras.constants import TRAINING_STAGES
from llamafactory.hparams import get_train_args, get_infer_args
from llamafactory.train.sft import run_sft
from llamafactory.train.tuner import export_model as lf_export_model
import argparse

# Your existing evaluation imports
from sentence_transformers import SentenceTransformer
from evaluate import load as hf_load
import numpy as np

# Import your evaluation functions (assuming they're in a separate file)
# from your_evaluation_module import answer_evaluation_batch, gpt_evaluation

@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning pipeline"""
    # Model settings - FIXED: use model_name_or_path instead of model_name
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"  # Updated to a more common model
    template: str = "qwen"  # Updated template to match Qwen model

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
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
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
            "model_name_or_path": self.config.model_name_or_path,
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
            "cutoff_len": self.config.max_seq_length,

            # LoRA settings
            "lora_rank": self.config.lora_rank,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,

            # Evaluation and saving
            "eval_strategy": self.config.evaluation_strategy,
            "save_strategy": self.config.save_strategy,
            "save_steps": self.config.save_steps,
            "eval_steps": self.config.eval_steps,
            "logging_steps": self.config.logging_steps,
            "load_best_model_at_end": self.config.load_best_model_at_end,
            "metric_for_best_model": self.config.metric_for_best_model,

            # Other settings
            "fp16": torch.cuda.is_available(),
            "remove_unused_columns": False,
            "report_to": "none",
            "do_train": True,
        }

        if self.config.valid_dataset:
            args["eval_dataset"] = self.config.valid_dataset  # FIXED: use eval_dataset
            args["do_eval"] = True

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
        
        # FIXED: Convert dict to list format for argument parsing
        args_list = []
        for key, value in train_args.items():
            args_list.extend([f"--{key}", str(value)])
        
        try:
            # Parse arguments using LLaMA Factory's argument parser
            model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args_list)
            
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

        finetune_type = (self.config.finetuning_type or "").lower()
        args = {
            "template": self.config.template,
            "finetuning_type": finetune_type,
            "export_dir": export_dir,
            "export_size": self.config.export_size,
            "export_device": self.config.export_device,
            "trust_remote_code": True,
        }

        if finetune_type == "lora":
            # LoRA merge: base model + adapter path (your training output_dir)
            args["model_name_or_path"] = self.config.model_name_or_path
            args["adapter_name_or_path"] = self.config.output_dir
        else:
            # Full fine-tune: the checkpoint dir *is* the model
            args["model_name_or_path"] = self.config.output_dir

        try:
            lf_export_model(args)  # <- single dict argument
            self.logger.info(f"Model exported successfully to {export_dir}")
            return export_dir
        except Exception as e:
            self.logger.error(f"Error during model export: {str(e)}")
            raise
    

    def load_fine_tuned_model(self, checkpoint_dir: Optional[str] = None):
        """Load fine-tuned model for inference"""
        ckpt_dir = checkpoint_dir or self.config.output_dir
        ft = (self.config.finetuning_type or "").lower()

        # Build a single dict for ChatModel
        args = {
            "template": self.config.template,
            "finetuning_type": ft,             # "lora" or "full"
            "infer_backend": "huggingface",    # or "vllm" if you use vLLM
            "trust_remote_code": True,
        }

        if ft == "lora":
            # Base model + adapter dir (training output_dir)
            args["model_name_or_path"] = self.config.model_name_or_path
            args["adapter_name_or_path"] = ckpt_dir
        else:
            # Full fine-tune OR merged export: directory *is* the model
            args["model_name_or_path"] = ckpt_dir

        try:
            # Pass a single dict into ChatModel
            chat_model = LlamaChatModel(args)
            return chat_model
        except Exception as e:
            self.logger.error(f"Error loading fine-tuned model: {str(e)}")
            raise
    
    def generate_answers(self, chat_model, test_data: List[Dict]) -> List[str]:
        """Generate answers using fine-tuned model"""
        generated_answers = []
        
        # Here, instead of loading the "instruction", we load the "prompt" as input.
        for item in test_data:
            query = item.get("instruction", "")  # FIXED: use "instruction" as primary field
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
        """Placeholder for your existing batch evaluation function"""
        # Since we can't import your actual evaluation module, here's a simplified version
        # Replace this with your actual evaluation function
        results = []
        for gen_ans, orig_ans in zip(generated_answers, original_answers):
            # Simple similarity calculation as placeholder
            rouge_score = self.rouge_metric.compute(predictions=[gen_ans], references=[orig_ans])['rouge1']
            
            # Simple SBERT similarity
            embeddings = model.encode([gen_ans, orig_ans])
            sbert_score = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
            
            # Simple BLEU score placeholder
            bleu_score = 0.5  # Replace with actual BLEU calculation
            
            results.append({
                "rouge_score": rouge_score,
                "sbert_score": float(sbert_score),
                "bleu_score": bleu_score
            })
        
        return results

    def save_config(self):
        """Save complete training configuration"""
        config_file = Path(self.config.output_dir) / "training_config.json"
        config_dict = {
            "model_name_or_path": self.config.model_name_or_path,  # FIXED: corrected key name
            "template": self.config.template,
            "train_dataset": self.config.train_dataset,
            "valid_dataset": self.config.valid_dataset,
            "test_dataset": self.config.test_dataset,
            "dataset_dir": self.config.dataset_dir,
            "learning_rate": self.config.learning_rate,
            "num_train_epochs": self.config.num_train_epochs,
            "per_device_train_batch_size": self.config.per_device_train_batch_size,
            "per_device_eval_batch_size": self.config.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "max_seq_length": self.config.max_seq_length,
            "finetuning_type": self.config.finetuning_type,
            "lora_rank": self.config.lora_rank,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "output_dir": self.config.output_dir,
            "export_dir": self.config.export_dir,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        self.logger.info(f"Training configuration saved to {config_file}")

    def save_pipeline_summary(self, results: Dict[str, Any]):
        """Save complete pipeline summary"""
        summary_file = Path(self.config.output_dir) / "pipeline_summary.json"
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config.__dict__,
            "results": results,
            "paths": {
                "model_checkpoint": self.config.output_dir,
                "exported_model": results.get("export_dir"),
                "training_log": f'fine_tuning_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
                "evaluation_results": str(Path(self.config.output_dir) / "evaluation_results.json") if "evaluation" in results else None
            }
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Pipeline summary saved to {summary_file}")
    
    def run_full_pipeline(self, train_data: List[Dict], valid_data: Optional[List[Dict]] = None,
                        test_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Run the complete fine-tuning and evaluation pipeline"""
        results = {}
        
        # Save configuration first
        self.save_config()
        
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
            
            # Save complete pipeline summary
            self.save_pipeline_summary(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            results["error"] = str(e)
            # Still save what we can
            self.save_pipeline_summary(results)
            return results

def _to_cli_args(args: Dict[str, Any]) -> List[str]:
    lst = []
    for k, v in args.items():
        if v is None:
            continue
        lst.extend([f"--{k.replace('_', '-')}", str(v)])
    return lst


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning pipeline with LLaMA Factory")

    # Model and template - FIXED: use model_name_or_path
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B-Instruct", 
                       help="Base model name or path")
    # BACKWARD COMPATIBILITY: Also accept --model_name for old scripts
    parser.add_argument("--model_name", type=str, help="Base model name (deprecated, use --model_name_or_path)")
    parser.add_argument("--template", type=str, default="qwen", help="Model template")

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
    parser.add_argument("--evaluation_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["steps", "epoch"])
    parser.add_argument("--load_best_model_at_end", action="store_true")
    parser.add_argument("--metric_for_best_model", type=str, default="eval_loss")

    return parser.parse_args()


def _parse_infer_args(args_list: List[str]):
    parsed = get_infer_args(args_list)
    # Support both signatures:
    #  (model_args, data_args, finetuning_args, generating_args)
    #  (model_args, data_args, training_args, finetuning_args, generating_args)
    if len(parsed) == 4:
        model_args, data_args, finetuning_args, generating_args = parsed
    elif len(parsed) == 5:
        model_args, data_args, _training_args, finetuning_args, generating_args = parsed
    else:
        raise RuntimeError(f"Unexpected get_infer_args return of length {len(parsed)}")
    return model_args, finetuning_args, generating_args



# Example usage
def main():
    args = parse_args()


    # file_exists = os.path.exists(training_data_path)
    # print(file_exists)
    training_data_path = f"{args.dataset_dir}/{args.train_dataset}.json"
    valid_data_path = f"{args.dataset_dir}/{args.valid_dataset}.json"
    test_data_path = f"{args.dataset_dir}/{args.test_dataset}.json"


    train_data = []
    with open(training_data_path, 'r') as json_file:
        raw = json.load(json_file)
        for json_line in raw:
            train_data.append(json_line)
    valid_data = []
    with open(valid_data_path, 'r') as json_file:
        raw = json.load(json_file)
        for json_line in raw:
            valid_data.append(json_line)
    test_data = []
    with open(test_data_path, 'r') as json_file:
        raw = json.load(json_file)
        for json_line in raw:
            test_data.append(json_line)
    
    # print(train_data)
    print(len(train_data))
    print(len(valid_data))
    print(len(test_data))
    # sys.exit()
    # Handle backward compatibility for model_name
    model_name_or_path = args.model_name_or_path
    if args.model_name:
        print("Warning: --model_name is deprecated. Use --model_name_or_path instead.")
        model_name_or_path = args.model_name

    # Map args into config dataclass - FIXED: use model_name_or_path
    config = FineTuningConfig(
        model_name_or_path=model_name_or_path,
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
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
    )

    fine_tuner = LlamaFactoryFineTuner(config)
    
    # Try to load data from files first, fall back to dummy data
        
    # Load some toy example data (fallback)
    # training data 

    training_data_path = f"{args.dataset_dir}/{args.train_dataset}.json"
    valid_data_path = f"{args.dataset_dir}/{args.valid_dataset}.json"
    test_data_path = f"{args.dataset_dir}/{args.test_dataset}.json"




    results = fine_tuner.run_full_pipeline(train_data, valid_data, test_data)

    print("Fine-tuning pipeline completed!")
    print(json.dumps(results, indent=2, default=str))

if __name__ == "__main__":
    main()