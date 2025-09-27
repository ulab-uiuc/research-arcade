import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Optional, Tuple
import numpy as np
from torch.utils.data import Dataset
import pickle
import os
import json

class EfficientTokenEmbedder:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B", cache_dir: str = "./token_embedding_cache"):
        """
        Initialize the token-level prompt embedder with pre-computed fixed token embeddings.
        
        Args:
            model_name: The Qwen model to use for embeddings
            cache_dir: Directory to cache pre-computed token embeddings
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Define the fixed prompt template parts
        self.fixed_template_parts = {
            'prefix': "You are reconstructing one missing LaTeX paragraph in a research paper.\nPaper title: ",
            'middle': "\nSection name of the paragraph: ",
            'suffix': "\n\n# Task\nWrite exactly one LaTeX-formatted paragraph.\n\n# HARD REQUIREMENTS\n- Maintain objective, concise academic tone; ensure logical continuity with context.\n- Output exactly one paragraph."
        }
        
        # Pre-compute fixed token embeddings and token IDs
        self.fixed_tokens, self.fixed_embeddings = self._precompute_fixed_token_embeddings()
        
    def _get_token_embeddings(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get token IDs and corresponding embeddings for a text.
        
        Returns:
            token_ids: [seq_len] tensor of token IDs
            token_embeddings: [seq_len, hidden_dim] tensor of token embeddings
        """
        # Tokenize
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            add_special_tokens=False,  # We'll handle special tokens manually
            truncation=False
        )
        
        token_ids = inputs['input_ids'].squeeze(0).to(torch.long)   # Remove batch dimension
        
        # Get embeddings from the model's embedding layer
        with torch.no_grad():
            token_embeddings = self.model.get_input_embeddings()(token_ids)

        return token_ids, token_embeddings
    
    def _precompute_fixed_token_embeddings(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Pre-compute token IDs and embeddings for fixed parts of the prompt."""
        cache_file = os.path.join(self.cache_dir, f"fixed_token_embeddings_{self.model_name.replace('/', '_')}.pkl")
        
        # Try to load from cache
        if os.path.exists(cache_file):
            print("Loading fixed token embeddings from cache...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        print("Computing fixed token embeddings...")
        fixed_tokens = {}
        fixed_embeddings = {}
        
        for part_name, text in self.fixed_template_parts.items():
            token_ids, token_embeds = self._get_token_embeddings(text)
            fixed_tokens[part_name] = token_ids
            fixed_embeddings[part_name] = token_embeds
            print(f"Fixed part '{part_name}': {len(token_ids)} tokens")
        
        # Save to cache
        cache_data = (fixed_tokens, fixed_embeddings)
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
            
        return fixed_tokens, fixed_embeddings
    
    def get_variable_token_embeddings(self, title: str, section: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Get token IDs and embeddings for variable parts."""
        title_tokens, title_embeds = self._get_token_embeddings(title)
        section_tokens, section_embeds = self._get_token_embeddings(section)
        
        variable_tokens = {'title': title_tokens, 'section': section_tokens}
        variable_embeddings = {'title': title_embeds, 'section': section_embeds}
        
        return variable_tokens, variable_embeddings
    
    def construct_full_prompt_embeddings(self, title: str, section: str, 
                                       max_length: int = 1024) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Construct the full prompt by combining fixed and variable token embeddings.
        
        Returns:
            input_ids: [seq_len] Combined token IDs
            input_embeddings: [seq_len, hidden_dim] Combined token embeddings  
            attention_mask: [seq_len] Attention mask
        """
        # Get variable embeddings
        var_tokens, var_embeddings = self.get_variable_token_embeddings(title, section)
        
        # Combine in order: prefix + title + middle + section + suffix
        combined_token_ids = torch.cat([
            self.fixed_tokens['prefix'],
            var_tokens['title'],
            self.fixed_tokens['middle'], 
            var_tokens['section'],
            self.fixed_tokens['suffix']
        ], dim=0)
        
        combined_embeddings = torch.cat([
            self.fixed_embeddings['prefix'],
            var_embeddings['title'],
            self.fixed_embeddings['middle'],
            var_embeddings['section'],
            self.fixed_embeddings['suffix']
        ], dim=0)
        
        # Truncate if too long
        if len(combined_token_ids) > max_length:
            combined_token_ids = combined_token_ids[:max_length]
            combined_embeddings = combined_embeddings[:max_length]
        
        # Create attention mask (all 1s since no padding yet)
        attention_mask = torch.ones(len(combined_token_ids), dtype=torch.long)
        
        return combined_token_ids, combined_embeddings, attention_mask
    
    def construct_full_prompt_tokens(
        self,
        title: str,
        section: str,
        max_length: int = 1024
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct the full prompt by combining fixed and variable token IDs only.

        Returns:
            input_ids: [seq_len] Combined token IDs
            attention_mask: [seq_len] Attention mask
        """
        # Get variable token IDs
        var_tokens, _ = self.get_variable_token_embeddings(title, section)

        # Combine in order: prefix + title + middle + section + suffix
        combined_token_ids = torch.cat([
            self.fixed_tokens['prefix'],
            var_tokens['title'],
            self.fixed_tokens['middle'],
            var_tokens['section'],
            self.fixed_tokens['suffix']
        ], dim=0)
        
        # Truncate if too long
        if len(combined_token_ids) > max_length:
            combined_token_ids = combined_token_ids[:max_length]
        
        # Attention mask: all ones
        attention_mask = torch.ones(len(combined_token_ids), dtype=torch.long)

        return combined_token_ids, attention_mask

    def pad_embeddings(self, input_ids: torch.Tensor, input_embeddings: torch.Tensor, 
                      attention_mask: torch.Tensor, max_length: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad token embeddings to max_length."""
        current_length = len(input_ids)
        
        if current_length >= max_length:
            return input_ids[:max_length], input_embeddings[:max_length], attention_mask[:max_length]
        
        # Pad with pad_token_id and corresponding embeddings
        pad_length = max_length - current_length
        pad_token_id = self.tokenizer.pad_token_id
        
        # Get pad token embedding
        with torch.no_grad():
            pad_embedding = self.model.get_input_embeddings()(torch.tensor([pad_token_id]))
        
        # Pad sequences
        padded_ids = torch.cat([
            input_ids,
            torch.full((pad_length,), pad_token_id, dtype=input_ids.dtype)
        ])
        
        padded_embeddings = torch.cat([
            input_embeddings,
            pad_embedding.repeat(pad_length, 1)
        ])
        
        padded_mask = torch.cat([
            attention_mask,
            torch.zeros(pad_length, dtype=attention_mask.dtype)
        ])
        
        return padded_ids, padded_embeddings, padded_mask
    
    def construct_full_sequence_with_target(self, title: str, section: str, target_paragraph: str, 
                                          context_token: str = "<|context|>", max_length: int = 1024) -> Dict[str, torch.Tensor]:
        """
        Construct the complete sequence: context + prompt + target for training.
        
        Returns:
            Dictionary with input_ids, input_embeddings, attention_mask, labels
        """
        # Add context token and generation prompt
        context_text = f"{context_token} Generate the missing paragraph:"
        context_tokens, context_embeds = self._get_token_embeddings(context_text)
        
        # Get prompt embeddings (fixed + variable parts)
        prompt_ids, prompt_embeds, _ = self.construct_full_prompt_embeddings(title, section, max_length//2)
        
        # Get target embeddings
        target_text = " " + target_paragraph + self.tokenizer.eos_token
        target_tokens, target_embeds = self._get_token_embeddings(target_text)
        
        # Combine all parts: context + prompt + target
        full_ids = torch.cat([context_tokens, prompt_ids, target_tokens])
        full_embeddings = torch.cat([context_embeds, prompt_embeds, target_embeds])
        
        # Truncate if necessary
        if len(full_ids) > max_length:
            full_ids = full_ids[:max_length]
            full_embeddings = full_embeddings[:max_length]
        
        # Create attention mask
        attention_mask = torch.ones(len(full_ids), dtype=torch.long)
        
        # Pad to max_length
        full_ids, full_embeddings, attention_mask = self.pad_embeddings(
            full_ids, full_embeddings, attention_mask, max_length
        )
        
        # Create labels (mask context + prompt, only train on target)
        labels = full_ids.clone()
        context_prompt_length = len(context_tokens) + len(prompt_ids)
        labels[:context_prompt_length] = -100  # Mask context and prompt
        
        return {
            'input_ids': full_ids,
            'input_embeddings': full_embeddings,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def get_context_embeddings_for_generation(self, title: str, section: str, 
                                            context_token: str = "<|context|>", max_length: int = 1024) -> Dict[str, torch.Tensor]:
        """
        Get context embeddings for generation (without target paragraph).
        
        Returns:
            Dictionary with input_ids, input_embeddings, attention_mask for generation
        """
        # Add context token and generation prompt
        context_text = f"{context_token} Generate the missing paragraph:"
        context_tokens, context_embeds = self._get_token_embeddings(context_text)
        
        # Get prompt embeddings (fixed + variable parts)
        prompt_ids, prompt_embeds, _ = self.construct_full_prompt_embeddings(title, section, max_length//2)
        
        # Combine context + prompt (no target for generation)
        full_ids = torch.cat([context_tokens, prompt_ids])
        full_embeddings = torch.cat([context_embeds, prompt_embeds])
        
        # Create attention mask
        attention_mask = torch.ones(len(full_ids), dtype=torch.long)
        
        # Pad if needed
        if len(full_ids) < max_length:
            full_ids, full_embeddings, attention_mask = self.pad_embeddings(
                full_ids, full_embeddings, attention_mask, max_length
            )
        
        return {
            'input_ids': full_ids,
            'input_embeddings': full_embeddings,
            'attention_mask': attention_mask
        }


class EnhancedParagraphGenerationDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, token_embedder: EfficientTokenEmbedder, 
                 max_length: int = 1024):
        """
        Enhanced dataset using efficient token embeddings.
        
        Args:
            data_path: Path to JSON data file
            tokenizer: Tokenizer (should match the one in token_embedder)
            token_embedder: EfficientTokenEmbedder instance
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.token_embedder = token_embedder

        # Load data
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Extract fields - adjust field names according to your data structure
        title = item.get("paper_title", item.get("title", ""))
        section = item.get("section_name", item.get("paper_section", ""))
        target_paragraph = item.get("target_paragraph", "")
        
        # Use efficient token embedding construction
        sequence_data = self.token_embedder.construct_full_sequence_with_target(
            title=title,
            section=section,
            target_paragraph=target_paragraph,
            context_token="<|context|>",
            max_length=self.max_length
        )

        return {
            "input_ids": sequence_data['input_ids'],
            "input_embeddings": sequence_data['input_embeddings'].float(),
            "attention_mask": sequence_data['attention_mask'],
            "labels": sequence_data['labels']
        }


class TokenEmbeddingModel(nn.Module):
    """
    Model wrapper that can accept pre-computed token embeddings instead of token IDs.
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, input_ids=None, input_embeddings=None, attention_mask=None, labels=None, **kwargs):
        """
        Forward pass that can use either input_ids or pre-computed input_embeddings.
        """
        if input_embeddings is not None:
            # Use pre-computed embeddings
            return self.base_model(
                inputs_embeds=input_embeddings,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )
        else:
            # Use regular input_ids
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )
    
    def generate(self, input_ids=None, input_embeddings=None, attention_mask=None, **kwargs):
        """Generation method that can use pre-computed embeddings."""
        if input_embeddings is not None:
            return self.base_model.generate(
                inputs_embeds=input_embeddings,
                attention_mask=attention_mask,
                **kwargs
            )
        else:
            return self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )

def save_paragraph_to_json(token_ids, embeddings, attention_mask, save_path):
    # Convert to CPU + numpy + list
    token_ids_list = token_ids.cpu().tolist()
    attn_mask_list = attention_mask.cpu().tolist()
    embeddings_list = embeddings.cpu().tolist()  # careful: this can be large!

    data = {
        "token_ids": token_ids_list,
        "attention_mask": attn_mask_list,
        "embeddings": embeddings_list
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Saved paragraph encodings to {save_path}")


def example_usage():
    """Example of how to use the efficient token embedder."""
    
    # Initialize the token embedder
    token_embedder = EfficientTokenEmbedder(model_name="Qwen/Qwen3-0.6B")
    
    # Example variables
    title = "Attention Is All You Need"
    section = "Related Work"
    target = "This section reviews the relevant literature in attention mechanisms."
    
    print("=== Token-level Embedding Construction ===")
    
    # Method 1: Training sequence (context + prompt + target)
    training_data = token_embedder.construct_full_sequence_with_target(
        title=title,
        section=section, 
        target_paragraph=target,
        max_length=512
    )
    
    print(f"Training sequence shape:")
    print(f"  input_ids: {training_data['input_ids'].shape}")
    print(f"  input_embeddings: {training_data['input_embeddings'].shape}")
    print(f"  attention_mask: {training_data['attention_mask'].shape}")
    print(f"  labels: {training_data['labels'].shape}")
    
    # Method 2: Generation context (context + prompt only)
    generation_data = token_embedder.get_context_embeddings_for_generation(
        title=title,
        section=section,
        max_length=512
    )
    
    print(f"\nGeneration context shape:")
    print(f"  input_ids: {generation_data['input_ids'].shape}")
    print(f"  input_embeddings: {generation_data['input_embeddings'].shape}")
    print(f"  attention_mask: {generation_data['attention_mask'].shape}")
    
    # Show token efficiency
    print(f"\n=== Efficiency Analysis ===")
    for part_name, tokens in token_embedder.fixed_tokens.items():
        print(f"Fixed part '{part_name}': {len(tokens)} tokens")
    
    print(f"Variable 'title': {len(token_embedder._get_token_embeddings(title)[0])} tokens")
    print(f"Variable 'section': {len(token_embedder._get_token_embeddings(section)[0])} tokens")
    
    # Demonstrate dataset usage
    print(f"\n=== Dataset Integration ===")
    
    # Create sample data file
    sample_data = [
        {
            "paper_title": "Attention Is All You Need",
            "section_name": "Related Work", 
            "target_paragraph": "This section reviews relevant literature."
        },
        {
            "paper_title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "section_name": "Introduction",
            "target_paragraph": "Language model pre-training has been highly effective."
        }
    ]
    
    # Save sample data
    with open("sample_data.json", "w") as f:
        json.dump(sample_data, f)
    
    # Create dataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    dataset = EnhancedParagraphGenerationDataset(
        data_path="sample_data.json",
        tokenizer=tokenizer,
        token_embedder=token_embedder,
        max_length=512
    )
    
    # Test dataset
    sample_item = dataset[0]
    print(f"Dataset item shapes:")
    for key, value in sample_item.items():
        print(f"  {key}: {value.shape}")
    
    # Cleanup
    os.remove("sample_data.json")
    
    print(f"\n=== Success! ===")
    print("Token embeddings are pre-computed and cached for efficiency!")

def main():

    etb = EfficientTokenEmbedder()
    paper_title = "Robustness of Proof of Team Sprint (PoTS) Against Attacks: A Simulation-Based Analysis"
    section_name = "Introduction"
    # title: str, section
    combined_token_ids, combined_embeddings, attention_mask = etb.construct_full_prompt_embeddings(title=paper_title, section=section_name)

    print(combined_token_ids, combined_embeddings, attention_mask)
    save_path = "./embedding.json"
    save_paragraph_to_json(token_ids=combined_token_ids, embeddings=combined_embeddings, attention_mask=attention_mask, save_path=save_path)


if __name__ == "__main__":
    main()