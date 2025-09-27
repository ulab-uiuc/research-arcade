import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
import json
import numpy as np
from typing import Dict, List, Optional
import random
from tqdm import tqdm
import wandb
from sklearn.cluster import KMeans
import sys

# Set device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiModalQwenGenerator(nn.Module):
    def __init__(self, base_model_name: str, embedding_dim: int, use_lora: bool = True, lora_config: Optional[LoraConfig] = None):
        super().__init__()
        self.device = device
        self.use_lora = use_lora
        
        # Load tokenizer and model for generation
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            device_map="auto"
        )
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.hidden_size = self.base_model.config.hidden_size
        
        # Projector for external embeddings (figures/tables/context)
        self.embedding_projector = nn.Linear(embedding_dim, self.hidden_size)
        
        # Special token for context
        self.context_token = "<|context|>"
        special_tokens = {"additional_special_tokens": [self.context_token]}
        self.tokenizer.add_special_tokens(special_tokens)
        self.base_model.resize_token_embeddings(len(self.tokenizer))
        
        # Apply LoRA
        if use_lora:
            if lora_config is None:
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,  # Changed for generation
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.1,
                    target_modules=[
                        "q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"
                    ],
                    bias="none",
                )
            
            print("Applying LoRA to base model...")
            self.base_model = get_peft_model(self.base_model, lora_config)
            self.base_model.print_trainable_parameters()

    def forward(self, input_ids, attention_mask=None, external_embeddings=None, labels=None):
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
            
        # Process external embeddings if provided
        if external_embeddings is not None:
            external_embeddings = external_embeddings.to(self.device)
            B, N, D = external_embeddings.shape
            
            # Project external embeddings to model hidden size
            context_embeds = self.embedding_projector(
                external_embeddings.view(B * N, D)
            ).view(B, N, -1)
            
            # Get text embeddings
            text_embeds = self.base_model.get_input_embeddings()(input_ids)
            
            # Concatenate context + text embeddings
            input_embeds = torch.cat([context_embeds, text_embeds], dim=1)
            
            # Update attention mask
            if attention_mask is not None:
                prefix_mask = torch.ones((B, N), dtype=attention_mask.dtype, device=self.device)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

            # Update labels if provided
            if labels is not None:
                prefix_labels = torch.full((B, N), -100, dtype=labels.dtype, device=self.device)
                labels = torch.cat([prefix_labels, labels], dim=1)
        else:
            input_embeds = None

        # Forward pass
        outputs = self.base_model(
            input_ids=input_ids if external_embeddings is None else None,
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

        return outputs

    def generate(self, input_ids, attention_mask=None, external_embeddings=None, **kwargs):
        """Generation method for inference"""
        # Handle external embeddings for generation
        if external_embeddings is not None:
            external_embeddings = external_embeddings.to(self.device)
            B, N, D = external_embeddings.shape

            # Project external embeddings
            context_embeds = self.embedding_projector(
                external_embeddings.view(B * N, D)
            ).view(B, N, -1)

            # Get text embeddings
            text_embeds = self.base_model.get_input_embeddings()(input_ids)

            # Concatenate
            input_embeds = torch.cat([context_embeds, text_embeds], dim=1)

            # Update attention mask
            if attention_mask is not None:
                prefix_mask = torch.ones((B, N), dtype=attention_mask.dtype, device=self.device)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

            # Generate with inputs_embeds
            return self.base_model.generate(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                **kwargs
            )
        else:
            return self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )


# It seems that I need to first convert all these annoying stuuf
def get_kmeans_cluster_centers(tem_all_embedding, n_clusters = 20, random_state=42):
    embeddings = np.array(tem_all_embedding, dtype=np.float32)

    # 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(embeddings)
    labels = kmeans.labels_

    # 计算每个 cluster 的均值向量
    cluster_centers = []
    for k in range(n_clusters):
        cluster_embeddings = embeddings[labels == k]
        cluster_mean = cluster_embeddings.mean(axis=0)
        cluster_centers.append(cluster_mean.tolist())

    return cluster_centers


class ParagraphGenerationDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data (JSON)
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)   # ✅ loads the entire list at once

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx, n_clusters = 20):
        item = self.data[idx]


        # --- Extract fields from JSON ---
        prompt = item.get("prompt", "") 
        context_parts = prompt

        adjacent_paragraphs = item.get("adjacent_paragraphs", "")
        context_parts += f"\n {adjacent_paragraphs}"

        image_descriptions = item.get("image_descriptions", "")
        context_parts += f"\n {image_descriptions}"

        bib_keys = item.get("bib_keys", "")
        context_parts += f"\n {bib_keys}"

        table_contents = item.get("table_contents", "")
        context_parts += f"\n {table_contents}"

        target_paragraph = item.get("target_paragraph", "")
        context_parts += f"\n {target_paragraph}"


        # Final context string
        context_text = f"<|context|> " + " ".join(context_parts) + " Generate the missing paragraph:"
        
        # --- Full training text (context + gold target) ---
        full_text = context_text + " " + target_paragraph + self.tokenizer.eos_token

        # --- Tokenize ---
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        labels = encoding["input_ids"].clone()

        # Mask context part in labels
        context_encoding = self.tokenizer(
            context_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        context_length = context_encoding["input_ids"].size(1)
        labels[:, :context_length] = -100

        # TODO here we have to be careful about the data structure of returned data.
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
            # If you later want embeddings, add them here
            "external_embeddings": torch.zeros(10, 1024, dtype=torch.float32)  
        }

    

class ParagraphGenerationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            external_embeddings=inputs.get('external_embeddings'),
            labels=inputs['labels']
        )
        
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


class ParagraphGenerationEvaluator:
    def __init__(self, model, tokenizer, test_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.test_dataset = test_dataset
    
    def evaluate_generation(self, batch_size: int = 4, save_path: Optional[str] = None) -> dict:
        self.model.eval()
        
        generated_texts = []
        reference_texts = []
        
        test_dataloader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, 
            collate_fn=self._collate_fn
        )
        
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Generating paragraphs"):
                # Extract context for generation
                context_encodings = []
                references = []
                
                for i in range(len(batch['input_ids'])):
                    # Find the generation start point
                    input_ids = batch['input_ids'][i]
                    labels = batch['labels'][i]
                    
                    # Find where labels stop being -100 (start of target)
                    target_start = (labels != -100).nonzero(as_tuple=True)[0]
                    if len(target_start) > 0:
                        context_end = target_start[0].item()
                        context_ids = input_ids[:context_end]
                        target_ids = input_ids[context_end:]
                        target_ids = target_ids[target_ids != self.tokenizer.pad_token_id]
                        
                        context_encodings.append(context_ids)
                        references.append(self.tokenizer.decode(target_ids, skip_special_tokens=True))
                    else:
                        # Fallback: use first half as context
                        mid = len(input_ids) // 2
                        context_encodings.append(input_ids[:mid])
                        target_ids = input_ids[mid:]
                        target_ids = target_ids[target_ids != self.tokenizer.pad_token_id]
                        references.append(self.tokenizer.decode(target_ids, skip_special_tokens=True))
                
                # Pad context encodings
                max_context_len = max(len(ctx) for ctx in context_encodings)
                padded_contexts = []
                context_masks = []
                
                for ctx in context_encodings:
                    padded = torch.cat([
                        ctx, 
                        torch.full((max_context_len - len(ctx),), self.tokenizer.pad_token_id, dtype=ctx.dtype)
                    ])
                    mask = torch.cat([
                        torch.ones(len(ctx), dtype=torch.bool),
                        torch.zeros(max_context_len - len(ctx), dtype=torch.bool)
                    ])
                    padded_contexts.append(padded)
                    context_masks.append(mask)
                
                context_input_ids = torch.stack(padded_contexts).to(device)
                context_attention_mask = torch.stack(context_masks).to(device)
                
                # Generate
                generated_ids = self.model.generate(
                    input_ids=context_input_ids,
                    attention_mask=context_attention_mask,
                    external_embeddings=batch['external_embeddings'].to(device),
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                # Extract only the generated part
                for i, gen_ids in enumerate(generated_ids):
                    context_len = len(context_encodings[i])
                    generated_part = gen_ids[context_len:]
                    generated_text = self.tokenizer.decode(generated_part, skip_special_tokens=True)
                    generated_texts.append(generated_text)
                
                reference_texts.extend(references)
        
        # Simple evaluation metrics
        avg_gen_length = sum(len(text.split()) for text in generated_texts) / len(generated_texts)
        avg_ref_length = sum(len(text.split()) for text in reference_texts) / len(reference_texts)
        results = {
            'avg_generated_length': avg_gen_length,
            'avg_reference_length': avg_ref_length,
            'num_samples': len(generated_texts),
            'generated_samples': list(zip(reference_texts[:3], generated_texts[:3]))
        }
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        return results

    
    def _collate_fn(self, batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        external_embeddings = torch.stack([item['external_embeddings'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'external_embeddings': external_embeddings
        }



class BestGenerationCallback(TrainerCallback):
    def __init__(self, generation_evaluator, save_dir: str = "./best_generation_model", 
                 eval_steps: int = 100):
        self.generation_evaluator = generation_evaluator
        self.save_dir = save_dir
        self.eval_steps = eval_steps
        self.last_eval_step = 0
        os.makedirs(save_dir, exist_ok=True)
    
    def on_log(self, args, state, control, model, logs=None, **kwargs):
        if (state.global_step > 0 and state.global_step - self.last_eval_step >= self.eval_steps):
            print(f"\n🔍 Evaluating paragraph generation at step {state.global_step}...")
            
            eval_results = self.generation_evaluator.evaluate_generation()
            
            print(f"📊 Generated {eval_results['num_samples']} samples")
            print(f"📊 Avg generated length: {eval_results['avg_generated_length']:.1f} words")
            print(f"📊 Avg reference length: {eval_results['avg_reference_length']:.1f} words")
            
            # Save model periodically
            print(f"💾 Saving model at step {state.global_step}...")
            step_save_dir = os.path.join(self.save_dir, f"step_{state.global_step}")
            model.save_pretrained(step_save_dir)
            
            # Print sample generations
            print("\n📝 Sample generations:")
            for i, (ref, gen) in enumerate(eval_results['generated_samples']):
                print(f"\nSample {i+1}:")
                print(f"Reference: {ref[:200]}...")
                print(f"Generated: {gen[:200]}...")
                print("-" * 80)
            
            model.train()
            self.last_eval_step = state.global_step


def create_sample_data(output_path: str, num_samples: int = 100):
    """Create sample data for testing the pipeline"""
    sample_data = []
    
    for i in range(num_samples):
        # Generate synthetic data
        adjacent_paragraphs = f"This is the context paragraph {i}. It discusses various topics related to the research domain."
        target_paragraph = f"This is the target paragraph {i} that should be generated based on the context. It continues the discussion and provides relevant information."
        figures_tables_embeddings = np.random.randn(5, 1024).tolist()  # Random embeddings
        
        sample_data.append({
            'adjacent_paragraphs': adjacent_paragraphs,
            'target_paragraph': target_paragraph,
            'figures_tables_embeddings': figures_tables_embeddings
        })
    
    # Save as JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in sample_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Created {num_samples} sample data points at {output_path}")


def main():
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available! Please check GPU and CUDA installation.")
    
    print(f"Using device: {device}")
    
    # Configuration
    base_model_name = "Qwen/Qwen3-0.6B"  # Use a smaller model for testing
    # base_model_name = "Qwen/Qwen2.5-0.5B"  # Alternative: use actual Qwen model
    embedding_dim = 1024
    max_length = 4096 
    
    # Data paths
    # train_data_path = "./train_paragraphs.jsonl"
    # test_data_path = "./test_paragraphs.jsonl"

    train_data_path = "./data/paragraph_generation/tasks/paragraph_generation_training_exp.json"
    test_data_path = "./data/paragraph_generation/tasks/paragraph_generation_testing_exp.json"

    experiment_result_dir = "./data/paragraph_generation/experiment_result"


    # Create sample data if files don't exist
    if not os.path.exists(train_data_path):
        # Report error
        raise error(f"train_data_path {train_data_path} does not exists")
        create_sample_data(train_data_path, num_samples=200)
    if not os.path.exists(test_data_path):
        raise error(f"test_data_path {test_data_path} does not exists")


    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # For Qwen
        bias="none",
    )
    
    # Initialize model
    print("🔧 Initializing model...")
    model = MultiModalQwenGenerator(
        base_model_name=base_model_name,
        embedding_dim=embedding_dim,
        use_lora=True,
        lora_config=lora_config
    )
    
    # Create datasets
    print("📊 Loading datasets...")
    train_dataset = ParagraphGenerationDataset(train_data_path, model.tokenizer, max_length)
    test_dataset = ParagraphGenerationDataset(test_data_path, model.tokenizer, max_length)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create evaluator and callback
    evaluator = ParagraphGenerationEvaluator(model, model.tokenizer, test_dataset)
    callback = BestGenerationCallback(
        evaluator,
        save_dir="./paragraph_generation_models",
        eval_steps=50
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./paragraph_generation_output",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        learning_rate=5e-5,
        logging_steps=10,
        save_strategy="no",  # We handle saving in callback
        eval_strategy="no",
        warmup_steps=100,
        max_grad_norm=1.0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        fp16=True,
        dataloader_num_workers=0,  # Set to 0 to avoid multiprocessing issues
        gradient_accumulation_steps=4,
    )
    
    # Data collator for language modeling
    def collate_fn(batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        external_embeddings = torch.stack([item['external_embeddings'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'external_embeddings': external_embeddings
        }

    # Create trainer
    trainer = ParagraphGenerationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        callbacks=[callback],
    )
    
    print("🚀 Starting paragraph generation training...")
    
    # Initial evaluation
    # We also need to add save path
    print("📊 Initial evaluation...")
    initial_results = evaluator.evaluate_generation(batch_size=2, save_path = f"{experiment_result_dir}/initial_results.json")
    print(f"Initial results: {initial_results}")
    
    # Train
    trainer.train()
    
    print("✅ Training completed!")
    
    # Final evaluation
    print("📊 Final evaluation...")
    final_results = evaluator.evaluate_generation(batch_size=2, save_path = f"{experiment_result_dir}/final_results.json")
    print(f"Final results: {final_results}")
    
    # Save final model
    final_model_path = "./final_paragraph_generation_model"
    model.save_pretrained(final_model_path)
    print(f"💾 Final model saved to {final_model_path}")


if __name__ == "__main__":
    main()