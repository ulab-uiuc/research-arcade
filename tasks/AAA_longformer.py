import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import LongformerModel, LongformerConfig
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import wandb
from sklearn.metrics import roc_auc_score, matthews_corrcoef
import numpy as np
import random

class SimplifiedLongformerModel(nn.Module):
    def __init__(self, in_channels, out_channels=768, dropout=0.1):
        super(SimplifiedLongformerModel, self).__init__()
        
        # Since we're working with graph embeddings, not raw text,
        # we'll use simple projection layers instead of Longformer
        # This makes it comparable to the HAT approach
        
        # Projection layers for different node types
        self.paragraph_proj = nn.Linear(in_channels, out_channels)
        self.figure_proj = nn.Linear(in_channels, out_channels)
        self.table_proj = nn.Linear(in_channels, out_channels)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Final prediction layer (same as HAT)
        self.classifier = nn.Linear(out_channels * 2, 1)
        
    def forward(self, x_dict, target_node_type, target_node_idx, return_all_embeddings=False):
        """
        Forward pass using graph embeddings (same input format as HAT)
        
        Args:
            x_dict: Dictionary of node embeddings (from graph.x_dict)
            target_node_type: 'figure' or 'table'
            target_node_idx: Index of target node
            return_all_embeddings: Whether to return all embeddings
        """
        
        # Project different node types to same dimension
        paragraph_embeddings = self.paragraph_proj(self.dropout(x_dict['paragraph']))
        figure_embeddings = self.figure_proj(self.dropout(x_dict['figure'])) if 'figure' in x_dict else None
        table_embeddings = self.table_proj(self.dropout(x_dict['table'])) if 'table' in x_dict else None
        
        if return_all_embeddings:
            return {
                'paragraph': paragraph_embeddings,
                'figure': figure_embeddings,
                'table': table_embeddings
            }
        
        # Get target node embedding
        if target_node_type == 'figure' and figure_embeddings is not None:
            if target_node_idx >= len(figure_embeddings):
                raise ValueError(f"Figure index {target_node_idx} out of range (max: {len(figure_embeddings)-1})")
            target_embedding = figure_embeddings[target_node_idx]
        elif target_node_type == 'table' and table_embeddings is not None:
            if target_node_idx >= len(table_embeddings):
                raise ValueError(f"Table index {target_node_idx} out of range (max: {len(table_embeddings)-1})")
            target_embedding = table_embeddings[target_node_idx]
        else:
            raise ValueError(f"Invalid target node type: {target_node_type}")
        
        # Compute similarity scores with all paragraphs (same as HAT)
        target_embedding_expanded = target_embedding.unsqueeze(0).repeat(paragraph_embeddings.size(0), 1)
        combined_features = torch.cat([target_embedding_expanded, paragraph_embeddings], dim=1)
        
        # Predict probability for each paragraph
        scores = self.classifier(combined_features).squeeze(-1)
        
        return scores

def contrastive_loss(target_embedding, paragraph_embeddings, true_paragraph_idx, temperature=0.1):
    """
    Contrastive loss for figure/table to paragraph matching
    """
    # Compute cosine similarity
    target_norm = F.normalize(target_embedding, p=2, dim=0)
    paragraph_norm = F.normalize(paragraph_embeddings, p=2, dim=1)
    
    similarities = torch.matmul(paragraph_norm, target_norm.unsqueeze(-1)).squeeze(-1)
    similarities = similarities / temperature
    
    # Use cross-entropy loss
    loss = F.cross_entropy(similarities.unsqueeze(0), true_paragraph_idx.unsqueeze(0))
    
    return loss

def calculate_binary_metrics(scores, true_idx, threshold=0.5):
    """
    Calculate Accuracy, AUC-ROC, and MCC for binary classification
    """
    scores_np = scores.detach().cpu().numpy()
    
    # Create binary labels (1 for true paragraph, 0 for all others)
    y_true = np.zeros(len(scores_np))
    y_true[true_idx.item()] = 1
    
    # Apply sigmoid to convert scores to probabilities
    y_prob = torch.sigmoid(scores).detach().cpu().numpy()
    
    # Binary predictions using threshold
    y_pred = (y_prob >= threshold).astype(int)
    
    # Calculate Accuracy
    accuracy = (y_pred == y_true).mean()
    
    # Calculate AUC-ROC
    try:
        if len(np.unique(y_true)) > 1:
            auc_roc = roc_auc_score(y_true, y_prob)
        else:
            auc_roc = 0.5
    except:
        auc_roc = 0.5
    
    # Calculate MCC
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
        if np.isnan(mcc):
            mcc = 0.0
    except:
        mcc = 0.0
    
    return accuracy, auc_roc, mcc

def sample_data(graphs, targets, sample_ratio=0.1, seed=42):
    """Sample a subset of the data"""
    random.seed(seed)
    np.random.seed(seed)
    
    total_samples = len(graphs)
    sample_size = int(total_samples * sample_ratio)
    
    indices = list(range(total_samples))
    sampled_indices = random.sample(indices, sample_size)
    
    sampled_graphs = [graphs[i] for i in sampled_indices]
    sampled_targets = [targets[i] for i in sampled_indices]
    
    print(f"Sampled {sample_size} out of {total_samples} samples ({sample_ratio*100:.1f}%)")
    
    return sampled_graphs, sampled_targets

def load_data(graphs_path, targets_path):
    """Load preprocessed data"""
    graphs = torch.load(graphs_path, weights_only=False)
    targets = torch.load(targets_path, weights_only=False)
    
    return graphs, targets

def train_epoch(model, train_graphs, train_targets, optimizer, device, accumulation_steps=16):
    """Train for one epoch"""
    model.train()
    epoch_losses = []
    accumulated_loss = 0.0
    optimizer.zero_grad()
    
    # Metrics
    total_accuracy = 0.0
    total_auc_roc = 0.0
    total_mcc = 0.0
    num_targets = 0
    
    for idx, (graph, target_data) in enumerate(tqdm(zip(train_graphs, train_targets), 
                                                      total=len(train_graphs), 
                                                      desc="Training")):
        graph = graph.to(device)
        
        # Handle both old and new target formats (same as HAT)
        if 'targets' in target_data:
            targets_list = target_data['targets']
        else:
            targets_list = []
            if 'figure_nodes' in target_data:
                for fig_idx in target_data['figure_nodes']:
                    targets_list.append({
                        'type': 'figure',
                        'node_idx': fig_idx,
                        'paragraph_idx': None
                    })
            if 'table_nodes' in target_data:
                for table_idx in target_data['table_nodes']:
                    targets_list.append({
                        'type': 'table',
                        'node_idx': table_idx,
                        'paragraph_idx': None
                    })
        
        # Process each target figure/table in this paper
        for target_info in targets_list:
            target_node_type = target_info['type']
            target_idx = target_info['node_idx']
            true_paragraph_idx = target_info.get('paragraph_idx')
            
            if true_paragraph_idx is None:
                continue
            
            try:
                # Get embeddings using graph.x_dict (same as HAT)
                embeddings = model(graph.x_dict, target_node_type, target_idx, return_all_embeddings=True)
                
                if target_node_type == 'figure':
                    target_embedding = embeddings['figure'][target_idx]
                else:
                    target_embedding = embeddings['table'][target_idx]
                
                # Compute loss
                true_paragraph_idx_tensor = torch.tensor(true_paragraph_idx, device=device)
                loss = contrastive_loss(target_embedding, embeddings['paragraph'], 
                                       true_paragraph_idx_tensor, temperature=0.1)
                loss = loss / accumulation_steps
                loss.backward()
                
                accumulated_loss += loss.item()
                
                # Calculate metrics (no grad needed)
                with torch.no_grad():
                    # Compute similarity scores
                    target_norm = F.normalize(target_embedding, p=2, dim=0)
                    paragraph_norm = F.normalize(embeddings['paragraph'], p=2, dim=1)
                    scores = torch.matmul(paragraph_norm, target_norm.unsqueeze(-1)).squeeze(-1)
                    
                    acc, auc, mcc = calculate_binary_metrics(scores, true_paragraph_idx_tensor)
                    total_accuracy += acc
                    total_auc_roc += auc
                    total_mcc += mcc
                    num_targets += 1
                    
            except (ValueError, IndexError) as e:
                # Skip invalid targets
                continue
        
        # Update parameters
        if (idx + 1) % accumulation_steps == 0 or (idx + 1) == len(train_graphs):
            if accumulated_loss > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()
            
            if accumulated_loss > 0:
                epoch_losses.append(accumulated_loss)
            accumulated_loss = 0.0
    
    avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
    avg_accuracy = total_accuracy / num_targets if num_targets > 0 else 0.0
    avg_auc_roc = total_auc_roc / num_targets if num_targets > 0 else 0.0
    avg_mcc = total_mcc / num_targets if num_targets > 0 else 0.0
    
    return avg_loss, avg_accuracy, avg_auc_roc, avg_mcc

def test(model, test_graphs, test_targets, device):
    """Test the model"""
    model.eval()
    test_losses = []
    
    # Metrics
    total_accuracy = 0.0
    total_auc_roc = 0.0
    total_mcc = 0.0
    num_targets = 0
    
    with torch.no_grad():
        for graph, target_data in tqdm(zip(test_graphs, test_targets), 
                                       total=len(test_graphs), 
                                       desc="Testing"):
            graph = graph.to(device)
            
            # Handle both old and new target formats
            if 'targets' in target_data:
                targets_list = target_data['targets']
            else:
                targets_list = []
                if 'figure_nodes' in target_data:
                    for fig_idx in target_data['figure_nodes']:
                        targets_list.append({
                            'type': 'figure',
                            'node_idx': fig_idx,
                            'paragraph_idx': None
                        })
                if 'table_nodes' in target_data:
                    for table_idx in target_data['table_nodes']:
                        targets_list.append({
                            'type': 'table',
                            'node_idx': table_idx,
                            'paragraph_idx': None
                        })
            
            for target_info in targets_list:
                target_node_type = target_info['type']
                target_idx = target_info['node_idx']
                true_paragraph_idx = target_info.get('paragraph_idx')
                
                if true_paragraph_idx is None:
                    continue
                
                try:
                    embeddings = model(graph.x_dict, target_node_type, target_idx, return_all_embeddings=True)
                    
                    if target_node_type == 'figure':
                        target_embedding = embeddings['figure'][target_idx]
                    else:
                        target_embedding = embeddings['table'][target_idx]
                    
                    true_paragraph_idx_tensor = torch.tensor(true_paragraph_idx, device=device)
                    loss = contrastive_loss(target_embedding, embeddings['paragraph'], 
                                           true_paragraph_idx_tensor, temperature=0.1)
                    test_losses.append(loss.item())
                    
                    # Calculate metrics
                    target_norm = F.normalize(target_embedding, p=2, dim=0)
                    paragraph_norm = F.normalize(embeddings['paragraph'], p=2, dim=1)
                    scores = torch.matmul(paragraph_norm, target_norm.unsqueeze(-1)).squeeze(-1)
                    
                    acc, auc, mcc = calculate_binary_metrics(scores, true_paragraph_idx_tensor)
                    total_accuracy += acc
                    total_auc_roc += auc
                    total_mcc += mcc
                    num_targets += 1
                    
                except (ValueError, IndexError) as e:
                    continue
    
    avg_loss = sum(test_losses) / len(test_losses) if test_losses else 0.0
    avg_accuracy = total_accuracy / num_targets if num_targets > 0 else 0.0
    avg_auc_roc = total_auc_roc / num_targets if num_targets > 0 else 0.0
    avg_mcc = total_mcc / num_targets if num_targets > 0 else 0.0
    
    return avg_loss, avg_accuracy, avg_auc_roc, avg_mcc

def main():
    parser = argparse.ArgumentParser(description='Train Simplified Longformer Model')
    parser.add_argument('--train_graphs', type=str, required=True, help='Path to training graphs')
    parser.add_argument('--train_targets', type=str, required=True, help='Path to training targets')
    parser.add_argument('--test_graphs', type=str, default=None, help='Path to test graphs')
    parser.add_argument('--test_targets', type=str, default=None, help='Path to test targets')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.005, help='Weight decay')
    parser.add_argument('--out_channels', type=int, default=768, help='Output channels')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--accumulation_steps', type=int, default=16, help='Gradient accumulation steps')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--test_interval', type=int, default=3, help='Run test evaluation every N epochs')
    parser.add_argument('--sample_ratio', type=float, default=0.1, help='Ratio of data to sample')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    parser.add_argument('--wandb_project', type=str, default='simplified-longformer', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity')
    parser.add_argument('--wandb_name', type=str, default=None, help='W&B run name')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=vars(args)
        )
    
    # Load data
    print("Loading data...")
    train_graphs, train_targets = load_data(args.train_graphs, args.train_targets)
    
    # Sample training data
    print("Sampling training data...")
    train_graphs, train_targets = sample_data(train_graphs, train_targets, args.sample_ratio, args.seed)
    
    print(f"Training samples: {len(train_graphs)}")
    
    # Load and sample test data if provided
    test_graphs, test_targets = None, None
    if args.test_graphs and args.test_targets:
        print("Loading test data...")
        test_graphs, test_targets = load_data(args.test_graphs, args.test_targets)
        print("Sampling test data...")
        test_graphs, test_targets = sample_data(test_graphs, test_targets, args.sample_ratio, args.seed)
        print(f"Test samples: {len(test_graphs)}")
    
    # Initialize model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Auto-detect input dimension from the first graph (same as HAT)
    first_graph = train_graphs[0]
    input_dim = None
    for node_type in first_graph.node_types:
        if hasattr(first_graph[node_type], 'x'):
            input_dim = first_graph[node_type].x.shape[1]
            break
    
    if input_dim is None:
        raise ValueError("Could not determine input dimension from graph")
    
    print(f"Detected input dimension: {input_dim}")
    
    model = SimplifiedLongformerModel(
        in_channels=input_dim,
        out_channels=args.out_channels,
        dropout=args.dropout
    )
    model.to(device)
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0001)
    
    # Training loop
    best_train_mcc = -1.0
    best_test_mcc = -1.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss, train_acc, train_auc, train_mcc = train_epoch(
            model, train_graphs, train_targets, optimizer, device, args.accumulation_steps
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Train - Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | AUC: {train_auc:.4f} | MCC: {train_mcc:.4f}")
        print(f"LR: {current_lr:.6f}")
        
        # Test evaluation
        test_loss, test_acc, test_auc, test_mcc = None, None, None, None
        if test_graphs is not None and (epoch + 1) % args.test_interval == 0:
            print("Running test evaluation...")
            test_loss, test_acc, test_auc, test_mcc = test(model, test_graphs, test_targets, device)
            print(f"Test  - Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | AUC: {test_auc:.4f} | MCC: {test_mcc:.4f}")
            
            if test_mcc > best_test_mcc:
                best_test_mcc = test_mcc
        
        # Log to wandb
        if not args.no_wandb:
            log_dict = {
                'epoch': epoch + 1,
                'train/loss': train_loss,
                'train/accuracy': train_acc,
                'train/auc_roc': train_auc,
                'train/mcc': train_mcc,
                'learning_rate': current_lr
            }
            
            if test_loss is not None:
                log_dict.update({
                    'test/loss': test_loss,
                    'test/accuracy': test_acc,
                    'test/auc_roc': test_auc,
                    'test/mcc': test_mcc
                })
            
            wandb.log(log_dict)
        
        # Save best model
        if train_mcc > best_train_mcc:
            best_train_mcc = train_mcc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_mcc': train_mcc,
                'train_accuracy': train_acc,
                'train_auc_roc': train_auc
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"✓ Saved best model (Train MCC: {train_mcc:.4f})")
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(args.output_dir, 'final_model.pth'))
    
    # Finish wandb
    if not args.no_wandb:
        wandb.finish()
    
    print(f"\nTraining complete!")
    print(f"Best training MCC: {best_train_mcc:.4f}")
    if test_graphs is not None:
        print(f"Best test MCC: {best_test_mcc:.4f}")

if __name__ == "__main__":
    main()