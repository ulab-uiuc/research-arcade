import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import numpy as np

# Solution 1: Implicit Negative Sampling for Embedding Framework
class ImplicitNegativeLoss(nn.Module):
    """
    Loss function that creates implicit negatives from positive samples
    """
    def __init__(self, temperature=0.2, negative_sampling_ratio=5):
        super().__init__()
        self.temperature = temperature
        self.negative_sampling_ratio = negative_sampling_ratio
    
    def forward(self, user_embeddings, positive_item_embeddings, all_item_embeddings=None):
        batch_size = user_embeddings.size(0)
        
        # Normalize embeddings
        user_embeddings = F.normalize(user_embeddings, p=2, dim=-1)
        positive_item_embeddings = F.normalize(positive_item_embeddings, p=2, dim=-1)
        
        # Compute positive similarities
        pos_sim = torch.sum(user_embeddings * positive_item_embeddings, dim=-1) / self.temperature
        
        # Strategy 1: In-batch negatives (use other positives in batch as negatives)
        if all_item_embeddings is None:
            # Use other items in the batch as implicit negatives
            all_pos_items = F.normalize(positive_item_embeddings, p=2, dim=-1)
            neg_sim = torch.mm(user_embeddings, all_pos_items.t()) / self.temperature
            
            # Remove self-similarities (diagonal)
            mask = torch.eye(batch_size, device=neg_sim.device).bool()
            neg_sim = neg_sim.masked_fill(mask, -1e9)
            
        else:
            # Strategy 2: Random sampling from item catalog
            num_negatives = min(len(all_item_embeddings), 
                              batch_size * self.negative_sampling_ratio)
            neg_indices = torch.randperm(len(all_item_embeddings))[:num_negatives]
            negative_items = F.normalize(all_item_embeddings[neg_indices], p=2, dim=-1)
            
            neg_sim = torch.mm(user_embeddings, negative_items.t()) / self.temperature
        
        # Compute InfoNCE loss
        losses = []
        for i in range(batch_size):
            if all_item_embeddings is None:
                # In-batch negatives: exclude self
                valid_neg_sim = torch.cat([neg_sim[i][:i], neg_sim[i][i+1:]])
            else:
                valid_neg_sim = neg_sim[i]
            
            all_sim = torch.cat([pos_sim[i:i+1], valid_neg_sim])
            loss = -pos_sim[i] + torch.logsumexp(all_sim, dim=0)
            losses.append(loss)
        
        return torch.stack(losses).mean()


# Solution 2: Self-Supervised Learning with Masking
class MaskedReconstructionLoss(nn.Module):
    """
    Mask random items in sequence and predict them
    """
    def __init__(self, mask_prob=0.15):
        super().__init__()
        self.mask_prob = mask_prob
        self.mse_loss = nn.MSELoss()
    
    def forward(self, sequence_embeddings, predicted_embeddings, mask=None):
        if mask is None:
            # Create random mask
            mask = torch.rand(sequence_embeddings.shape[:-1]) < self.mask_prob
        
        # Only compute loss on masked positions
        masked_targets = sequence_embeddings[mask]
        masked_predictions = predicted_embeddings[mask]
        
        return self.mse_loss(masked_predictions, masked_targets)


# Solution 3: Prompt-Based with Positive-Only Training
class PositiveOnlyPromptModel(nn.Module):
    def __init__(self, base_model_name: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def create_positive_prompt(self, user_history, target_item, prompt_style="completion"):
        """
        Create prompts that only use positive examples
        """
        if prompt_style == "completion":
            # Sequence completion task
            prompt = f"User interaction sequence: {' -> '.join(user_history)} -> {target_item}"
            
        elif prompt_style == "explanation":
            # Explanation task (why this item follows)
            prompt = f"Given user history: {user_history}\n"
            prompt += f"The user interacted with {target_item} because:"
            
        elif prompt_style == "pattern":
            # Pattern recognition task
            prompt = f"Pattern: {' -> '.join(user_history)}\n"
            prompt += f"Next item following this pattern: {target_item}"
            
        elif prompt_style == "preference":
            # Preference modeling task
            prompt = f"User preferences based on history {user_history}:\n"
            prompt += f"The user would like: {target_item}"
            
        return prompt
    
    def forward(self, user_history, target_item, prompt_style="completion"):
        prompt = self.create_positive_prompt(user_history, target_item, prompt_style)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Standard language modeling loss
        outputs = self.model(**inputs, labels=inputs['input_ids'])
        return outputs.loss


# Solution 4: Curriculum Learning with Synthetic Negatives
class CurriculumNegativeSampler:
    """
    Gradually introduce harder negatives during training
    """
    def __init__(self, item_embeddings, curriculum_stages=3):
        self.item_embeddings = item_embeddings
        self.curriculum_stages = curriculum_stages
        self.current_stage = 0
        self.stage_steps = 0
        self.steps_per_stage = 1000  # Adjust based on your training
    
    def sample_negatives(self, positive_embedding, num_negatives=10):
        """Sample negatives with curriculum difficulty"""
        
        if self.current_stage == 0:
            # Stage 1: Random negatives (easiest)
            indices = torch.randperm(len(self.item_embeddings))[:num_negatives]
            return self.item_embeddings[indices]
        
        elif self.current_stage == 1:
            # Stage 2: Somewhat similar negatives (medium difficulty)
            similarities = torch.mm(positive_embedding.unsqueeze(0), 
                                  self.item_embeddings.t()).squeeze()
            # Sample from middle similarity range
            sorted_indices = torch.argsort(similarities)
            mid_start = len(sorted_indices) // 3
            mid_end = 2 * len(sorted_indices) // 3
            candidates = sorted_indices[mid_start:mid_end]
            
            if len(candidates) >= num_negatives:
                indices = candidates[torch.randperm(len(candidates))[:num_negatives]]
            else:
                indices = candidates
            return self.item_embeddings[indices]
        
        else:
            # Stage 3: Very similar negatives (hardest)
            similarities = torch.mm(positive_embedding.unsqueeze(0), 
                                  self.item_embeddings.t()).squeeze()
            # Sample from high similarity items (but not the exact positive)
            _, sorted_indices = torch.sort(similarities, descending=True)
            # Skip the most similar (likely the positive itself)
            hard_negatives = sorted_indices[1:num_negatives+1]
            return self.item_embeddings[hard_negatives]
    
    def update_curriculum(self):
        """Update curriculum stage"""
        self.stage_steps += 1
        if (self.stage_steps > self.steps_per_stage and 
            self.current_stage < self.curriculum_stages - 1):
            self.current_stage += 1
            self.stage_steps = 0
            print(f"Curriculum advanced to stage {self.current_stage}")


# Solution 5: Autoencoder-based Reconstruction
class AutoencoderRecommender(nn.Module):
    """
    Learn representations through reconstruction of positive interactions
    """
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # For binary interaction prediction
        )
        
        self.reconstruction_loss = nn.BCELoss()
    
    def forward(self, user_interaction_vector):
        """
        user_interaction_vector: Binary vector indicating interactions
        """
        # Encode user preferences
        user_embedding = self.encoder(user_interaction_vector)
        
        # Decode to reconstruct interactions
        reconstructed = self.decoder(user_embedding)
        
        # Reconstruction loss on positive interactions only
        positive_mask = user_interaction_vector > 0
        if positive_mask.sum() > 0:
            loss = self.reconstruction_loss(
                reconstructed[positive_mask], 
                user_interaction_vector[positive_mask]
            )
        else:
            loss = torch.tensor(0.0, requires_grad=True)
        
        return loss, user_embedding, reconstructed


# Usage examples for different approaches
def train_with_positive_only():
    """Example training loop for positive-only data"""
    
    # Approach 1: Use implicit negatives
    model = MultiModalQwenEmbedding(...)  # Your existing model
    loss_fn = ImplicitNegativeLoss()
    
    for batch in dataloader:
        user_embeddings = model(batch['input_ids'], batch['attention_mask'], 
                               batch['external_embeddings'])
        
        loss = loss_fn(user_embeddings, batch['positive_item_embeddings'], 
                      all_item_embeddings)  # Pass full item catalog
        
        loss.backward()
        optimizer.step()
    
    # Approach 2: Curriculum learning with synthetic negatives
    sampler = CurriculumNegativeSampler(all_item_embeddings)
    
    for batch in dataloader:
        # Generate curriculum-based negatives
        synthetic_negatives = []
        for pos_emb in batch['positive_item_embeddings']:
            negs = sampler.sample_negatives(pos_emb, num_negatives=10)
            synthetic_negatives.append(negs)
        
        # Train with synthetic negatives
        user_embeddings = model(batch['input_ids'], batch['attention_mask'], 
                               batch['external_embeddings'])
        loss = infonce_loss(user_embeddings, batch['positive_item_embeddings'],
                           torch.stack(synthetic_negatives))
        
        loss.backward()
        optimizer.step()
        sampler.update_curriculum()
    
    # Approach 3: Prompt-based positive-only training
    prompt_model = PositiveOnlyPromptModel("Qwen/Qwen2-7B-Instruct")
    
    for batch in dataloader:
        total_loss = 0
        for user_hist, target in zip(batch['user_histories'], batch['targets']):
            loss = prompt_model(user_hist, target, prompt_style="completion")
            total_loss += loss
        
        (total_loss / len(batch)).backward()
        optimizer.step()

# Evaluation strategies for positive-only models
def evaluate_positive_only_model(model, test_data, method="reconstruction"):
    """
    Evaluation strategies when you only have positive samples
    """
    if method == "reconstruction":
        # Measure how well model reconstructs held-out positive interactions
        total_reconstruction_error = 0
        for user_vector in test_data:
            # Hold out some positive interactions
            mask = torch.rand(len(user_vector)) < 0.2  # Hold out 20%
            train_vector = user_vector.copy()
            train_vector[mask] = 0
            
            # Reconstruct
            _, _, reconstructed = model(train_vector)
            
            # Measure error on held-out positives
            error = F.mse_loss(reconstructed[mask], user_vector[mask])
            total_reconstruction_error += error.item()
        
        return total_reconstruction_error / len(test_data)
    
    elif method == "next_item_prediction":
        # Predict next item in sequence (temporal splitting)
        correct_predictions = 0
        total_predictions = len(test_data)
        
        for sequence in test_data:
            if len(sequence) < 2:
                continue
                
            # Use all but last item as input
            context = sequence[:-1]
            target = sequence[-1]
            
            # Get model prediction
            prediction = model.predict_next(context)
            
            if prediction == target:
                correct_predictions += 1
        
        return correct_predictions / total_predictions