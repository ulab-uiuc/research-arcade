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
    DataCollatorForLanguageModeling,
    
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
from sentence_transformers import SentenceTransformer

import json
import numpy as np
from typing import Dict, List, Optional
import random
from tqdm import tqdm
import wandb
from sklearn.cluster import KMeans
import sys
import argparse
import numpy as np
from typing import List, Tuple
from collections import deque
import time
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tasks.generated_paragraph_evaluation import answer_evaluation_batch


wandb_api_key = os.getenv('WANDB_TOKEN')
# Enable wandb online logging
os.environ.pop("WANDB_MODE", None)  # Remove offline mode if set
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


wandb.login(key=wandb_api_key)
# wandb.init(project="paragraph-generation")

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

            if external_embeddings.dim() == 3:
                # [B, N, D]
                B, N, D = external_embeddings.shape
                external_embeddings = external_embeddings.view(B, N, D)
            elif external_embeddings.dim() == 4:
                # [B, K, N, D]
                B, K, N, D = external_embeddings.shape
                external_embeddings = external_embeddings.view(B, K * N, D)
            else:
                raise ValueError(f"Unexpected external_embeddings shape: {external_embeddings.shape}")

            # Project external embeddings to model hidden size
            context_embeds = self.embedding_projector(
                external_embeddings.view(-1, D)        # flatten batch
            ).view(B, -1, self.hidden_size)           # restore batch

            # Get text embeddings
            text_embeds = self.base_model.get_input_embeddings()(input_ids)

            # Concatenate context + text embeddings
            input_embeds = torch.cat([context_embeds, text_embeds], dim=1)

            # Update attention mask
            if attention_mask is not None:
                prefix_mask = torch.ones((B, context_embeds.size(1)), dtype=torch.long, device=self.device)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

            # Update labels if provided
            if labels is not None:
                prefix_labels = torch.full((B, context_embeds.size(1)), -100, dtype=labels.dtype, device=self.device)
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
        """Generation method for inference with [B, K, N, D] external embeddings"""
        if external_embeddings is not None:
            external_embeddings = external_embeddings.to(self.device)
            B, K, N, D = external_embeddings.shape  # batch, steps, graph_size, dim

            # Flatten steps + nodes → [B, K*N, D]
            external_embeddings = external_embeddings.view(B, K * N, D)

            # Project external embeddings to hidden size
            context_embeds = self.embedding_projector(
                external_embeddings.view(B * K * N, D)
            ).view(B, K * N, -1)   # [B, K*N, hidden_size]

            # Get text embeddings
            text_embeds = self.base_model.get_input_embeddings()(input_ids)

            # Concatenate context + text embeddings
            input_embeds = torch.cat([context_embeds, text_embeds], dim=1)

            # Update attention mask
            if attention_mask is not None:
                prefix_mask = torch.ones((B, K * N), dtype=torch.long, device=self.device)
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
            
    def save_pretrained(self, save_directory):
        """保存LoRA模型和projector"""
        os.makedirs(save_directory, exist_ok=True)

        # 保存tokenizer
        self.tokenizer.save_pretrained(save_directory)

        # 如果使用LoRA，保存LoRA权重
        if self.use_lora:
            self.base_model.save_pretrained(save_directory)
        else:
            # 如果没有使用LoRA，保存整个模型
            torch.save(self.base_model.state_dict(), os.path.join(save_directory, "base_model.bin"))

        # 只保存embedding_projector权重，移除output_projector
        projector_state = {
            'embedding_projector': self.embedding_projector.state_dict(),
        }
        torch.save(projector_state, os.path.join(save_directory, "projector_weights.bin"))

        # 保存模型配置
        config = {
            "hidden_size": self.hidden_size,
            "embedding_dim": self.embedding_projector.in_features,
            "use_lora": self.use_lora,
        }

        with open(os.path.join(save_directory, "model_config.json"), 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Model saved to {save_directory}")


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


def get_mean_pooling(all_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Perform mean pooling over a batch of external embeddings.
    
    Args:
        all_embeddings (torch.Tensor): [B, N, D] where
            B = batch size
            N = number of embeddings per sample
            D = embedding dimension
    
    Returns:
        torch.Tensor: [B, 1, D] mean-pooled embedding for each sample
    """
    if all_embeddings.dim() != 3:
        raise ValueError(f"Expected input shape [B, N, D], got {all_embeddings.shape}")
    
    # Average over the N dimension
    pooled = all_embeddings.mean(dim=1, keepdim=True)  # [B, 1, D]
    return pooled
    


class ParagraphGenerationDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, embedder_name: str = "all-MiniLM-L6-v2", max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data (JSON)
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)   # ✅ loads the entire list at once

        self.embedder = SentenceTransformer(embedder_name)

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        item = self.data[idx]

        # --- Extract fields ---
        prompt = item.get("prompt", "")
        adjacent_paragraphs = item.get("adjacent_paragraphs", [])
        image_description_list = item.get("image_description_list", [])
        bib_keys = item.get("bib_keys", "")
        table_list = item.get("table_contents", [])
        target_paragraph = item.get("target_paragraph", "")

        # --- Build context string (shortened to avoid OOM) ---
        context_text = f"<|context|> " + prompt + " Generate the missing paragraph:"
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
        
        # Mask context tokens in labels
        context_encoding = self.tokenizer(
            context_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        context_length = context_encoding["input_ids"].size(1)
        labels[:, :context_length] = -100

        # --- External embeddings ---
        external_texts = []

        # Collect adj. paragraphs
        if isinstance(adjacent_paragraphs, list):
            external_texts.extend(adjacent_paragraphs)

        # Collect image descriptions
        if isinstance(image_description_list, list):
            external_texts.extend(image_description_list)

        # Collect tables
        if isinstance(table_list, list):
            external_texts.extend(table_list)

        # Add bib keys (string)
        if isinstance(bib_keys, str) and bib_keys.strip():
            external_texts.append(bib_keys)

        # Encode all external sources (if any)
        if external_texts:
            emb = self.embedder.encode(external_texts, convert_to_tensor=True)  # [N, D]
            emb = emb.unsqueeze(0)  # [1, N, D]
            pooled_ext = get_mean_pooling(emb)  # [1, 1, D]
            external_embeddings = pooled_ext # [1, D]
        else:
            # No external info → zero vector
            external_embeddings = torch.zeros(1, self.embedder.get_sentence_embedding_dimension())

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
            "external_embeddings": external_embeddings.float()  # [D] or [1, D]
        }


class PreEmbeddedParagraphGenerationDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = 2048, k_step_random_walk=2, tokenizer_name="Qwen/Qwen3-0.6B", include_figure=True, include_table=True):
        self.max_length = max_length
        self.k_step_random_walk = k_step_random_walk
        
        with open(data_path, "r", encoding="utf-8") as f:
            first_char = f.read(1)      # Peek at first non-whitespace char
            f.seek(0)                   # Reset file pointer

            if first_char.strip() == "[":  
                # JSON array
                self.data = json.load(f)  
            else:
                # JSONL
                self.data = [json.loads(line) for line in f if line.strip()]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.include_figure = include_figure
        self.include_table = include_table
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]

        # -----------------------------
        # 1. Context (already pre-computed)
        # -----------------------------
        combined_token_ids = torch.tensor(item["combined_token_ids"], dtype=torch.long)
        attention_mask = torch.tensor(item["attention_mask"], dtype=torch.long)

        # -----------------------------
        # 2. Target paragraph (tokenize here)
        # -----------------------------
        target_paragraph = item.get("target_paragraph", "")
        target_ids = self.tokenizer(
            target_paragraph + self.tokenizer.eos_token,
            truncation=True,
            max_length=self.max_length,   # safeguard
            return_tensors="pt"
        )["input_ids"].squeeze(0)

        # -----------------------------
        # 3. Append context + target
        # -----------------------------
        full_ids = torch.cat([combined_token_ids, target_ids], dim=0)
        full_mask = torch.cat([
            attention_mask,
            torch.ones_like(target_ids)
        ], dim=0)

        # -----------------------------
        # 4. Handle padding/truncation
        # -----------------------------
        if full_ids.size(0) > self.max_length:
            full_ids = full_ids[:self.max_length]
            full_mask = full_mask[:self.max_length]
        elif full_ids.size(0) < self.max_length:
            pad_length = self.max_length - full_ids.size(0)
            full_ids = torch.cat([full_ids, torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=torch.long)])
            full_mask = torch.cat([full_mask, torch.zeros(pad_length, dtype=torch.long)])

        # -----------------------------
        # 5. Labels (mask context, train only on target)
        # -----------------------------
        labels = full_ids.clone()
        context_length = combined_token_ids.size(0)
        labels[:context_length] = -100   # ignore context loss

        # -----------------------------
        # 6. External embeddings
        # -----------------------------

        def ensure_2d(arr, D=768):
            arr = np.array(arr)
            
            # Handle completely empty case
            if arr.size == 0:
                return np.zeros((0, D))
            
            if arr.ndim == 1:  
                if len(arr) == D:  # Single embedding vector
                    return arr.reshape(1, -1)
                else:  # Unexpected 1D shape
                    return np.zeros((0, D))
            elif arr.ndim == 2:
                if arr.shape[1] == D:  # Correct 2D shape
                    return arr
                else:  # Wrong second dimension
                    return np.zeros((0, D))
            else:
                return np.zeros((0, D))
        
        if self.include_figure:
            image_description_list_embeddings = ensure_2d(item.get("image_description_list_embeddings", []))
        else:
            image_description_list_embeddings = np.zeros((0, 768))

        if self.include_table:   
            table_list_embeddings = ensure_2d(item.get("table_list_embeddings", []))
        else:
            table_list_embeddings = np.zeros((0, 768))
        
        bib_key_embedding = ensure_2d(item.get("bib_key_embedding", []))
        prev_paras_embeddings = ensure_2d(item["prev_paras_embeddings"])
        next_paras_embeddings = ensure_2d(item["next_paras_embeddings"])
        
        paragraph_external_embeddings = np.zeros((0, 768))
        paragraph_external_embeddings = np.concatenate([image_description_list_embeddings, table_list_embeddings, bib_key_embedding], axis=0)
        

        A, X_list = self.build_adjacence_matrix(
            paragraph_external_embeddings=paragraph_external_embeddings,
            prev_paragraph_embeddings=prev_paras_embeddings,
            next_paragraph_embeddings=next_paras_embeddings,
            k_step_random_walk=self.k_step_random_walk
        )


        if self.include_figure and self.include_table:
            N_max = self.k_step_random_walk * 2 + 6
        elif self.include_figure or self.include_figure:
            N_max = self.k_step_random_walk * 2 + 3
        else:
            N_max = self.k_step_random_walk * 2

        external_embeddings = np.stack(X_list)   # [k+1, N, D]
        N = external_embeddings.shape[1]
        if N < N_max:
            pad = np.zeros((external_embeddings.shape[0], N_max - N, external_embeddings.shape[2]))
            external_embeddings = np.concatenate([external_embeddings, pad], axis=1)
        elif N > N_max:
            external_embeddings = external_embeddings[:, :N_max, :]
        external_embeddings = torch.tensor(external_embeddings, dtype=torch.float)
        return {
            "input_ids": full_ids,
            "attention_mask": full_mask,
            "labels": labels,
            "external_embeddings": external_embeddings.float(),
        }
    
    def build_adjacence_matrix(self, paragraph_external_embeddings, prev_paragraph_embeddings, next_paragraph_embeddings, k_step_random_walk=1):
        """
        Build adjacency matrix A and feature matrix X, then compute
        [X, AX, A^2X, ..., A^kX].

        Args:
            paragraph_external_embeddings (list[np.ndarray])
            prev_paragraph_embeddings (list[np.ndarray])
            next_paragraph_embeddings (list[np.ndarray])
            k_step_random_walk (int): number of random walk steps

        Returns:
            A (np.ndarray): adjacency matrix [N, N]
            X_list (list[np.ndarray]): list of feature matrices
                [AX, A^2X, ..., A^kX], each [N, D]
        """

        # Here, we add the logic that if k = 0, then we only use the paragraph_external_embeddings, not using the prev_paagraph_embeddings and next_paragraph_embeddings

        # if k_step_random_walk == 0:
        #     prev_paragraph_embeddings = []
        #     next_paragraph_embeddings = []
        #     k_step_random_walk = 1

        # Infer embedding dimension D first, before converting to arrays
        prev_paragraph_embeddings = prev_paragraph_embeddings[:min(len(prev_paragraph_embeddings), k_step_random_walk)]
        next_paragraph_embeddings = next_paragraph_embeddings[:min(len(next_paragraph_embeddings), k_step_random_walk)]

        D = None


        # Try to get dimension from any non-empty embedding list
        if paragraph_external_embeddings is not None and len(paragraph_external_embeddings) > 0:
            if isinstance(paragraph_external_embeddings[0], (list, np.ndarray)):
                D = len(paragraph_external_embeddings[0])
        elif prev_paragraph_embeddings is not None and len(prev_paragraph_embeddings) > 0:
            if isinstance(prev_paragraph_embeddings[0], (list, np.ndarray)):
                D = len(prev_paragraph_embeddings[0])
        elif next_paragraph_embeddings is not None and len(next_paragraph_embeddings) > 0:
            if isinstance(next_paragraph_embeddings[0], (list, np.ndarray)):
                D = len(next_paragraph_embeddings[0])
        
        # Default dimension if all lists are empty
        if D is None:
            D = 768  # default embedding dimension
        # Convert to numpy arrays with proper shape handling
        if len(paragraph_external_embeddings) > 0:
            paragraph_external_embeddings = np.array(paragraph_external_embeddings)
        else:
            paragraph_external_embeddings = np.zeros((0, D))
            
        if len(prev_paragraph_embeddings) > 0:
            prev_paragraph_embeddings = np.array(prev_paragraph_embeddings)
        else:
            prev_paragraph_embeddings = np.zeros((0, D))
            
        if len(next_paragraph_embeddings) > 0:
            next_paragraph_embeddings = np.array(next_paragraph_embeddings)
        else:
            next_paragraph_embeddings = np.zeros((0, D))

        def to_2d_array(arr, D):
            arr = np.array(arr)
            if arr.ndim == 1:          # single vector, e.g. shape (D,)
                arr = arr.reshape(1, -1)
            elif arr.ndim == 0:        # completely empty / scalar
                arr = np.zeros((0, D))
            elif arr.shape[1] != D:    # mismatch, pad or truncate
                if arr.shape[1] > D:   # too wide → truncate
                    arr = arr[:, :D]
                else:                  # too narrow → pad
                    pad = D - arr.shape[1]
                    arr = np.pad(arr, ((0, 0), (0, pad)), mode="constant")
            return arr
        
        paragraph_external_embeddings = to_2d_array(paragraph_external_embeddings, D) if len(paragraph_external_embeddings) > 0 else np.zeros((0, D))
        prev_paragraph_embeddings      = to_2d_array(prev_paragraph_embeddings, D) if len(prev_paragraph_embeddings) > 0 else np.zeros((0, D))
        next_paragraph_embeddings      = to_2d_array(next_paragraph_embeddings, D) if len(next_paragraph_embeddings) > 0 else np.zeros((0, D))
        

        # Handle case where all embeddings are empty
        if (paragraph_external_embeddings.size == 0 and 
            prev_paragraph_embeddings.size == 0 and 
            next_paragraph_embeddings.size == 0):
            print("Warning: All embedding lists are empty, using default embedding")
            X = np.zeros((1, D))
            A = np.eye(1, dtype=np.float32)
            
            # Generate random walk sequence
            X_list = [X.copy()]
            X_curr = X.copy()
            for _ in range(k_step_random_walk):
                X_curr = A @ X_curr
                X_list.append(X_curr.copy())
            
            return A, X_list

        # for prev_paragraph_embedding in prev_paragraph_embeddings:
        #     if len(prev_paragraph_embedding) != D:
        #         print("prev_paragraph_embedding")
        #         print(len(prev_paragraph_embedding))
        #         sys.exit()
        # for next_paragraph_embedding in next_paragraph_embeddings:
        #     if len(next_paragraph_embedding) != D:
        #         print("next_paragraph_embedding")
        #         print(len(next_paragraph_embedding))
        #         sys.exit()
        # for paragraph_external_embedding in paragraph_external_embeddings:
        #     if len(paragraph_external_embedding) != D:
        #         print("paragraph_external_embedding")
        #         print(len(paragraph_external_embedding))
        #         sys.exit()
        # Feature matrix X
        self_embedding = np.zeros((1, D))  # empty self
        X = np.vstack([
            self_embedding,
            prev_paragraph_embeddings if prev_paragraph_embeddings.size > 0 else np.zeros((0, D)),
            next_paragraph_embeddings if next_paragraph_embeddings.size > 0 else np.zeros((0, D)),
            paragraph_external_embeddings if paragraph_external_embeddings.size > 0 else np.zeros((0, D))
        ])
        N = X.shape[0]

        # Build adjacency with self loops
        A = np.eye(N, dtype=np.float32)

        offset_prev = 1
        offset_next = offset_prev + len(prev_paragraph_embeddings)
        offset_ext = offset_next + len(next_paragraph_embeddings)

        # Connect self to externals
        for i in range(offset_ext, N):
            A[0, i] = 1
            A[i, 0] = 1

        # Prev chain
        for i in range(len(prev_paragraph_embeddings)):
            idx = offset_prev + i
            if i == 0:
                A[0, idx] = A[idx, 0] = 1
            else:
                A[idx-1, idx] = A[idx, idx-1] = 1

        # Next chain
        for i in range(len(next_paragraph_embeddings)):
            idx = offset_next + i
            if i == 0:
                A[0, idx] = A[idx, 0] = 1
            else:
                A[idx-1, idx] = A[idx, idx-1] = 1

        # Normalize adjacency
        row_sums = A.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        A_norm = A / row_sums

        # Compute random walk sequence
        X_list = [X.copy()]
        X_curr = X.copy()
        for _ in range(k_step_random_walk):
            X_curr = A_norm @ X_curr
            # Only append the very first row, which represents the external embedding of self
            X_list.append(X_curr.copy())

        return A, X_list

class ParagraphGenerationTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.val_dataset = kwargs.pop('val_dataset', None)
        self.val_collate_fn = kwargs.pop('val_collate_fn', None)
        super().__init__(*args, **kwargs)
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs.get('input_ids')
        attention_mask = inputs.get('attention_mask')
        labels = inputs.get('labels')
        external_embeddings = inputs.get('external_embeddings')
        
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            external_embeddings=inputs.get('external_embeddings'),
            labels=inputs['labels']
        )
        
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training step to add validation after each batch"""
        # Normal training step
        model.train()
        train_loss = super().training_step(model, inputs, num_items_in_batch)

        # Validate periodically
        # if self.val_dataset and self.state.global_step % batch_validation_frequency == 0:
        #     val_loss = self.validate_model(model)
            
        #     wandb.log({
        #         "train/batch_loss": train_loss,
        #         "val/batch_loss": val_loss,
        #         "step": self.state.global_step
        #     })

        return train_loss
    
    def validate_model(self, model):
        """Validate model on validation dataset"""
        model.eval()
        val_losses = []
        
        # Create validation dataloader
        val_dataloader = DataLoader(
            self.val_dataset, 
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.val_collate_fn,
            shuffle=False
        )
        
        with torch.no_grad():
            for batch in val_dataloader:
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(model.device)
                
                # Compute validation loss
                outputs = model(**batch)
                val_losses.append(outputs.loss.item())
                
                # Only validate on a subset to save time
                if len(val_losses) >= 10:  # Validate on max 10 batches
                    break
        
        model.train()
        return np.mean(val_losses) if val_losses else float('inf')


class ParagraphGenerationEvaluator:
    def __init__(self, model, tokenizer, test_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.test_dataset = test_dataset
        self.embedder_model = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))

        
    def evaluate_generation(self, batch_size: int = 4, save_path: Optional[str] = None) -> dict:
        self.model.eval()
        
        generated_texts = []
        reference_texts = []
        
        test_dataloader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, 
            collate_fn=self._collate_fn
        )
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Generating paragraphs")):
                # Extract context for generation
                context_encodings = []
                references = []
                
                for i in range(len(batch['input_ids'])):
                    # Find the generation start point
                    input_ids = batch['input_ids'][i]
                    labels = batch['labels'][i]
                    target_start = (labels != -100).nonzero(as_tuple=True)[0]
                    if len(target_start) > 0:
                        context_end = target_start[0].item()
                        context_ids = input_ids[:context_end]
                        target_ids = input_ids[context_end:]
                        target_ids = target_ids[target_ids != self.tokenizer.pad_token_id]

                        if len(target_ids) > 0 and target_ids[-1] == self.tokenizer.eos_token_id:
                            target_ids = target_ids[:-1]  # Remove EOS token
                        reference_text = self.tokenizer.decode(
                            target_ids, 
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True
                        ).strip()

                        context_encodings.append(context_ids)
                        references.append(reference_text)
                    else:
                        # Fallback: use first half as context
                        print("Fallback")
                        mid = len(input_ids) // 2
                        context_encodings.append(input_ids[:mid])
                        target_ids = input_ids[mid:]
                        target_ids = target_ids[target_ids != self.tokenizer.pad_token_id]
                        if len(target_ids) > 0 and target_ids[-1] == self.tokenizer.eos_token_id:
                            target_ids = target_ids[:-1]
                        
                        reference_text = self.tokenizer.decode(
                            target_ids, 
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True
                        ).strip()
                        references.append(reference_text)
                
                # Pad context encodings
                max_context_len = max(len(ctx) for ctx in context_encodings)
                padded_contexts = []
                context_masks = []
                
                for ctx in context_encodings:
                    padded = torch.cat([
                        ctx, 
                        torch.full(
                            (max_context_len - len(ctx),),
                            self.tokenizer.pad_token_id,
                            dtype=ctx.dtype,
                            device=ctx.device   # 🔑 ensure same device
                        )
                    ])
                    mask = torch.cat([
                        torch.ones(len(ctx), dtype=torch.bool, device=ctx.device),
                        torch.zeros(max_context_len - len(ctx), dtype=torch.bool, device=ctx.device)
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
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.0,
                    length_penalty=1.0
                )
                
                # Extract only the generated part
                for i, gen_ids in enumerate(generated_ids):
                    context_len = len(context_encodings[i])
                    generated_part = gen_ids[context_len:]
                    
                    # Clean generated tokens
                    generated_tokens = generated_part.tolist()
                    
                    # Filter out special tokens
                    filtered_tokens = []
                    for token in generated_tokens:
                        if token == self.tokenizer.eos_token_id:
                            break  # Stop at EOS
                        if token not in [self.tokenizer.pad_token_id]:
                            filtered_tokens.append(token)
                    
                    if batch_idx == 0 and i == 0:
                        print(f"Generated token IDs (first 15): {filtered_tokens[:15]}")
                        print("Individual token decoding (first 10):")
                        for j, token_id in enumerate(filtered_tokens[:10]):
                            try:
                                decoded = self.tokenizer.decode([token_id], skip_special_tokens=True)
                                print(f"  Token {j}: ID={token_id}, Text='{decoded}', Repr={repr(decoded)}")
                            except Exception as e:
                                print(f"  Token {j}: ID={token_id}, Error={e}")
                    
                    # Decode the generated text
                    if filtered_tokens:
                        generated_text = self.tokenizer.decode(
                            filtered_tokens, 
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True
                        )
                        
                        # Additional cleanup
                        generated_text = generated_text.strip()
                        # Normalize excessive whitespace
                        generated_text = ' '.join(generated_text.split())
                        
                        # Debug first sample
                        if batch_idx == 0 and i == 0:
                            print(f"Final generated text: '{generated_text[:200]}...'")
                            print("================================")
                    else:
                        generated_text = ""
                    generated_texts.append(generated_text)

                reference_texts.extend(references)
        
        # More evaluation metrics need to be included
        
        # Simple evaluation metrics
        avg_gen_length = sum(len(text.split()) for text in generated_texts) / len(generated_texts)
        avg_ref_length = sum(len(text.split()) for text in reference_texts) / len(reference_texts)

        # generated_texts = []
        # reference_texts = []

        answer_evals = answer_evaluation_batch(generated_texts, reference_texts, model=self.embedder_model)
        # Obtain the evaluation score for each metrics
        # Here, we need to take average.
        # rouge_score_avg = 0
        # sbert_scores_avg = 0
        # bleu_scores_avg = 0

        # for rouge_scores, sbert_scores, bleu_scores in answer_evals:
        #     rouge_score_avg += rouge_scores
        #     sbert_scores_avg += sbert_scores
        #     bleu_scores_avg += bleu_scores
        # rouge_score_avg = rouge_score_avg / len(answer_evals)
        # sbert_scores_avg = sbert_scores_avg / len(answer_evals)
        # bleu_scores_avg = bleu_scores_avg / len(answer_evals)

        rouge_avg, sbert_avg, bleu_avg = calculate_averages_numpy(answer_evals)
        
        results = {
            'avg_generated_length': avg_gen_length,
            'avg_reference_length': avg_ref_length,
            'num_samples': len(generated_texts),
            'generated_samples': list(zip(reference_texts[:3], generated_texts[:3])),
            'rouge_avg': rouge_avg,
            'sbert_avg': sbert_avg,
            'bleu_avg': bleu_avg
        }
        
        # Log to wandb if available
        if wandb.run:
            wandb.log({
                'eval/rouge_score': rouge_avg,
                'eval/sbert_score': sbert_avg,
                'eval/bleu_score': bleu_avg,
                'eval/avg_generated_length': avg_gen_length,
                'eval/avg_reference_length': avg_ref_length
            })
        
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        return results

        
    def _collate_fn(self, batch):
        global device


        input_ids = torch.stack([item['input_ids'].to(device) for item in batch])
        attention_mask = torch.stack([item['attention_mask'].to(device) for item in batch])
        labels = torch.stack([item['labels'].to(device) for item in batch])
        external_embeddings = torch.stack([item['external_embeddings'].to(device) for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'external_embeddings': external_embeddings
        }


class EarlyStoppingCallback(TrainerCallback):
    """Enhanced early stopping callback with loss convergence detection"""
    
    def __init__(self, 
                 early_stopping_patience: int = 3,
                 early_stopping_threshold: float = 0.001,
                 min_delta: float = 0.0,
                 convergence_window: int = 5,
                 convergence_threshold: float = 0.05):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.min_delta = min_delta
        self.convergence_window = convergence_window
        self.convergence_threshold = convergence_threshold
        
        self.loss_history = deque(maxlen=convergence_window)
        self.best_metric = None
        self.best_step = 0
        self.patience_counter = 0
        self.should_stop = False
        
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs is None:
            return
            
        current_loss = logs.get("train_loss")
        if current_loss is None:
            return
            
        self.loss_history.append(current_loss)
        
        # Check for convergence
        if len(self.loss_history) == self.convergence_window:
            loss_variance = np.var(list(self.loss_history))
            loss_trend = np.mean(list(self.loss_history)[-3:]) - np.mean(list(self.loss_history)[:3])
            
            # Log convergence metrics
            wandb.log({
                "train/loss_variance": loss_variance,
                "train/loss_trend": loss_trend,
                "train/convergence_metric": loss_variance + abs(loss_trend),
                "step": state.global_step
            })
            
            # Check if loss has converged
            if loss_variance < self.convergence_threshold and abs(loss_trend) < self.convergence_threshold:
                print(f"🔥 Loss convergence detected! Variance: {loss_variance:.6f}, Trend: {loss_trend:.6f}")
                wandb.log({"train/converged": 1, "step": state.global_step})
                control.should_training_stop = True
                self.should_stop = True
                return
        
        # Traditional early stopping logic
        if self.best_metric is None or current_loss < self.best_metric - self.min_delta:
            self.best_metric = current_loss
            self.best_step = state.global_step
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        # Log early stopping metrics
        wandb.log({
            "train/best_loss": self.best_metric,
            "train/patience_counter": self.patience_counter,
            "train/steps_since_best": state.global_step - self.best_step,
            "step": state.global_step
        })
        
        if self.patience_counter >= self.early_stopping_patience:
            print(f"🛑 Early stopping triggered! No improvement for {self.patience_counter} evaluations.")
            wandb.log({"train/early_stopped": 1, "step": state.global_step})
            control.should_training_stop = True
            self.should_stop = True


class BestGenerationCallback(TrainerCallback):
    def __init__(self, generation_evaluator, save_dir: str = "./best_generation_model", 
                 eval_steps: int = 100):
        self.generation_evaluator = generation_evaluator
        self.save_dir = save_dir
        self.eval_steps = eval_steps
        self.last_eval_step = 0
        self.best_rouge_score = 0.0
        os.makedirs(save_dir, exist_ok=True)

    def on_log(self, args, state, control, model, logs=None, **kwargs):
        if (state.global_step > 0 and state.global_step - self.last_eval_step >= self.eval_steps):
            print(f"\n🔍 Evaluating paragraph generation at step {state.global_step}...")
            
            eval_results = self.generation_evaluator.evaluate_generation()
            
            current_rouge = eval_results.get('rouge_avg', 0.0)
            
            print(f"📊 Generated {eval_results['num_samples']} samples")
            print(f"📊 Avg generated length: {eval_results['avg_generated_length']:.1f} words")
            print(f"📊 Avg reference length: {eval_results['avg_reference_length']:.1f} words")
            print(f"📊 ROUGE Score: {current_rouge:.4f}")
            print(f"📊 SBERT Score: {eval_results.get('sbert_avg', 0.0):.4f}")
            print(f"📊 BLEU Score: {eval_results.get('bleu_avg', 0.0):.4f}")
            
            # Save best model based on ROUGE score
            if current_rouge > self.best_rouge_score:
                self.best_rouge_score = current_rouge
                print(f"🌟 New best ROUGE score: {current_rouge:.4f}! Saving best model...")
                best_model_dir = os.path.join(self.save_dir, "best_model")
                model.save_pretrained(best_model_dir)
                
                wandb.log({
                    "eval/best_rouge_score": self.best_rouge_score,
                    "step": state.global_step
                })
            
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


class WandbLoggingCallback(TrainerCallback):
    """Enhanced WandB logging callback"""
    
    def __init__(self):
        self.step_times = deque(maxlen=100)
        self.last_time = time.time()
        
    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.time()
        
    def on_step_end(self, args, state, control, **kwargs):
        step_time = time.time() - self.step_start_time
        self.step_times.append(step_time)
        
        # Log training speed metrics
        if len(self.step_times) > 10:
            avg_step_time = np.mean(list(self.step_times))
            wandb.log({
                "train/step_time": step_time,
                "train/avg_step_time": avg_step_time,
                "train/steps_per_second": 1.0 / avg_step_time,
                "step": state.global_step
            })
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
            
        # Enhanced logging with additional metrics
        log_dict = {}
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                if key == "learning_rate":
                    log_dict["train/learning_rate"] = value
                elif key == "train_loss":
                    log_dict["train/loss"] = value
                elif key == "grad_norm":
                    log_dict["train/grad_norm"] = value
                else:
                    log_dict[f"train/{key}"] = value
        
        log_dict["step"] = state.global_step
        log_dict["epoch"] = state.epoch
        
        wandb.log(log_dict)


def calculate_averages_numpy(answer_evals: List[Dict[str, float]]) -> Tuple[float, float, float]:
    """
    Calculate averages from list of dictionaries containing evaluation scores.
    
    Args:
        answer_evals: List of dicts like [{"rouge_score": 0.8, "sbert_score": 0.9, "bleu_score": 0.7}, ...]
        
    Returns:
        Tuple of (rouge_avg, sbert_avg, bleu_avg)
    """
    if not answer_evals:
        return 0.0, 0.0, 0.0
    
    # Extract scores into separate arrays
    rouge_scores = np.array([item["rouge_score"] for item in answer_evals])
    sbert_scores = np.array([item["sbert_score"] for item in answer_evals])
    bleu_scores = np.array([item["bleu_score"] for item in answer_evals])
    
    # Calculate means
    rouge_avg = np.mean(rouge_scores)
    sbert_avg = np.mean(sbert_scores)
    bleu_avg = np.mean(bleu_scores)
    
    return rouge_avg, sbert_avg, bleu_avg


# def create_validation_split(data_path: str, val_ratio: float = 0.1, seed: int = 42):
#     """Create train/validation split from training data"""
#     with open(data_path, "r", encoding="utf-8") as f:
#         data = json.load(f)
    
#     random.seed(seed)
#     random.shuffle(data)
    
#     val_size = int(len(data) * val_ratio)
#     val_data = data[:val_size]
#     train_data = data[val_size:]
    
#     # Save splits
#     # train_split_path = data_path.replace('.json', '_train_split.json')
#     # val_split_path = data_path.replace('.json', '_val_split.json')
    
#     with open(train_split_path, 'w', encoding='utf-8') as f:
#         json.dump(train_data, f, ensure_ascii=False, indent=2)
        
#     with open(val_split_path, 'w', encoding='utf-8') as f:
#         json.dump(val_data, f, ensure_ascii=False, indent=2)
    
#     print(f"Created train split: {len(train_data)} samples -> {train_split_path}")
#     print(f"Created validation split: {len(val_data)} samples -> {val_split_path}")
    
#     return train_split_path, val_split_path


def parse_args():
    parser = argparse.ArgumentParser(description="Paragraph generation training")
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="./data/paragraph_generation/tasks/paragraph_generation_training_exp.json",
        help="Path to training data (.json or .jsonl).",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="./data/paragraph_generation/tasks/paragraph_generation_testing_exp.json",
        help="Path to testing data (.json or .jsonl).",
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        default="./data/paragraph_generation/tasks/paragraph_generation_validation_exp.json",
        help="Path to testing data (.json or .jsonl).",
    )
    parser.add_argument(
        "--k_step_random_walk",
        type=int,
        default=2,
        help="Number of steps to perform in the random walk (e.g., 2 means compute X, AX, A^2X).",
    )
    parser.add_argument(
        "--final_model_path",
        type=str,
        default="./data/paragraph_generation/final_paragraph_generation_model",
        help="Path to save final model.",
    )
    parser.add_argument(
        "--experiment_result_dir",
        type=str,
        help="Directory to save evaluation JSONs.",
    )
    parser.add_argument(
        "--include_figure",
        action="store_true",
        help="Include figures during train/test.",
    )
    parser.add_argument(
        "--include_table",
        action="store_true",
        help="Include tables during train/test.",
    )
    parser.add_argument(
        "--callback_eval_steps",
        type=int,
        default=50,
        help="Steps between callback evaluations.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="paragraph-generation-fuller",
        help="WandB project name.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="WandB run name.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=5,
        help="Early stopping patience.",
    )
    parser.add_argument(
        "--convergence_window",
        type=int,
        default=10,
        help="Window size for loss convergence detection.",
    )
    parser.add_argument(
        "--convergence_threshold",
        type=float,
        default=0.05,
        help="Threshold for loss convergence detection.",
    )
    parser.add_argument(
        "--batch_validation_frequency",
        type=int,
        default=10,
        help="Validate every N training steps.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available! Please check GPU and CUDA installation.")
    
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)} (CUDA device {torch.cuda.current_device()})")
    else:
        print("Using CPU")
    
    # Initialize WandB
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "model_name": "Qwen/Qwen3-0.6B",
            "max_length": 2048,
            "batch_size": 2,
            "learning_rate": 5e-5,
            "num_epochs": 3,
            "lora_r": 16,
            "lora_alpha": 32,
            "early_stopping_patience": args.early_stopping_patience,
            "convergence_window": args.convergence_window,
            "convergence_threshold": args.convergence_threshold,
            "include_figure": args.include_figure,
            "include_table": args.include_table,
        }
    )
    
    # Configuration
    base_model_name = "Qwen/Qwen3-0.6B"  # Use a smaller model for testing
    
    # Data paths
    train_data_path = args.train_data_path
    test_data_path = args.test_data_path
    val_data_path = args.val_data_path
    experiment_result_dir = args.experiment_result_dir
    final_model_path = args.final_model_path

    include_figure = args.include_figure
    include_table = args.include_table

    k_step_random_walk = args.k_step_random_walk

    # Create sample data if files don't exist
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"train_data_path {train_data_path} does not exist")
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"test_data_path {test_data_path} does not exist")
    
    
    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    
    embedder_name = "all-mpnet-base-v2"
    embedder = SentenceTransformer(embedder_name)
    embedding_dim = embedder.get_sentence_embedding_dimension()

    max_length = 2048
    
    # Initialize model
    print("🔧 Initializing model...")
    model = MultiModalQwenGenerator(
        base_model_name=base_model_name,
        embedding_dim=embedding_dim,
        use_lora=True,
        lora_config=lora_config
    )
    
    # Log model parameters to wandb
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    wandb.log({
        "model/total_parameters": total_params,
        "model/trainable_parameters": trainable_params,
        "model/trainable_ratio": trainable_params / total_params
    })
    
    # Create datasets
    train_dataset = PreEmbeddedParagraphGenerationDataset(
        data_path=train_data_path, max_length=max_length, k_step_random_walk=k_step_random_walk, include_figure=args.include_figure, include_table=args.include_table
    )
    val_dataset = PreEmbeddedParagraphGenerationDataset(
        data_path=val_data_path, max_length=max_length, k_step_random_walk=k_step_random_walk, include_figure=args.include_figure, include_table=args.include_table
    )
    test_dataset = PreEmbeddedParagraphGenerationDataset(
        data_path=test_data_path, max_length=max_length, k_step_random_walk=k_step_random_walk, include_figure=args.include_figure, include_table=args.include_table
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Data collator for language modeling
    def collate_fn(batch):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_ids = torch.stack([item['input_ids'].to(device) for item in batch])
        attention_mask = torch.stack([item['attention_mask'].to(device) for item in batch])
        labels = torch.stack([item['labels'].to(device) for item in batch])

        external_embeddings = torch.stack([
            item['external_embeddings'].to(device) for item in batch
        ])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'external_embeddings': external_embeddings
        }
    
    # Create evaluator and callbacks
    evaluator = ParagraphGenerationEvaluator(model, model.tokenizer, test_dataset)
    
    # Initialize callbacks
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=args.early_stopping_patience,
        convergence_window=args.convergence_window,
        convergence_threshold=args.convergence_threshold
    )
    
    generation_callback = BestGenerationCallback(
        evaluator,
        save_dir="./paragraph_generation_models",
        eval_steps=args.callback_eval_steps
    )
    
    wandb_callback = WandbLoggingCallback()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./paragraph_generation_output",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=10,  # Increase epochs since we have early stopping
        learning_rate=5e-5,
        logging_steps=5,  # More frequent logging
        save_strategy="no",  # We handle saving in callback
        eval_strategy="no",
        warmup_steps=100,
        max_grad_norm=1.0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        fp16=True,
        dataloader_num_workers=0,
        gradient_accumulation_steps=4,
        report_to="wandb",  # Enable wandb integration
        logging_dir="./logs",
        load_best_model_at_end=False,  # We handle this in callbacks
    )
    
    # Create trainer with validation dataset
    trainer = ParagraphGenerationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,  # Add validation dataset
        val_collate_fn=collate_fn,  # Add validation collate function
        data_collator=collate_fn,
        callbacks=[early_stopping_callback, generation_callback, wandb_callback],
    )
    
    print("🚀 Starting paragraph generation training...")
    
    # Initial evaluation
    print("📊 Initial evaluation...")
    os.makedirs(experiment_result_dir, exist_ok=True)

    initial_results = evaluator.evaluate_generation(
        batch_size=2, 
        save_path=f"{experiment_result_dir}/initial_results.json"
    )
    print(f"Initial results: {initial_results}")
    
    # Log initial results to wandb
    wandb.log({
        "eval/initial_rouge": initial_results.get('rouge_avg', 0.0),
        "eval/initial_sbert": initial_results.get('sbert_avg', 0.0),
        "eval/initial_bleu": initial_results.get('bleu_avg', 0.0),
    })
    
    # Train with early stopping and convergence detection
    try:
        trainer.train()
        print("✅ Training completed successfully!")
        
        # Check why training stopped
        if early_stopping_callback.should_stop:
            if len(early_stopping_callback.loss_history) == early_stopping_callback.convergence_window:
                print("🔥 Training stopped due to loss convergence")
                wandb.log({"training/stop_reason": "convergence"})
            else:
                print("🛑 Training stopped due to early stopping")
                wandb.log({"training/stop_reason": "early_stopping"})
        else:
            print("⏰ Training completed all epochs")
            wandb.log({"training/stop_reason": "completed"})
            
    except KeyboardInterrupt:
        print("⚠️ Training interrupted by user")
        wandb.log({"training/stop_reason": "interrupted"})
    
    # Final evaluation
    print("📊 Final evaluation...")
    final_results = evaluator.evaluate_generation(
        batch_size=2, 
        save_path=f"{experiment_result_dir}/final_results.json"
    )
    print(f"Final results: {final_results}")
    
    # Log final results
    wandb.log({
        "eval/final_rouge": final_results.get('rouge_avg', 0.0),
        "eval/final_sbert": final_results.get('sbert_avg', 0.0),
        "eval/final_bleu": final_results.get('bleu_avg', 0.0),
    })
    
    # Save final model
    model.save_pretrained(final_model_path)
    print(f"💾 Final model saved to {final_model_path}")
    
    # Save model artifact to wandb
    artifact = wandb.Artifact("final_model", type="model")
    artifact.add_dir(final_model_path)
    wandb.log_artifact(artifact)
    
    # Finish wandb run
    wandb.finish()
    
    print("🎉 Training pipeline completed!")


if __name__ == "__main__":
    main()