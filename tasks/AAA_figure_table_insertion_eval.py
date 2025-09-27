import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HANConv
import json
import numpy as np
import os

class FigureTableInsertionModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, metadata):
        super(FigureTableInsertionModel, self).__init__()
        
        self.conv = HANConv(in_channels, hidden_channels, heads=8, dropout=0.6, metadata=metadata)
        
        self.paragraph_proj = nn.Linear(hidden_channels, out_channels)
        self.figure_proj = nn.Linear(hidden_channels, out_channels)
        self.table_proj = nn.Linear(hidden_channels, out_channels)
        self.cited_paper_proj = nn.Linear(hidden_channels, out_channels)
        
        self.classifier = nn.Linear(out_channels * 2, 1)
        
    def forward(self, x_dict, edge_index_dict, target_node_type, target_node_idx, return_all_embeddings=False):
        out = self.conv(x_dict, edge_index_dict)
        
        paragraph_embeddings = self.paragraph_proj(out['paragraph'])
        figure_embeddings = self.figure_proj(out['figure']) if 'figure' in out else None
        table_embeddings = self.table_proj(out['table']) if 'table' in out else None
        cited_paper_embeddings = self.cited_paper_proj(out['cited_paper']) if 'cited_paper' in out else None
        
        if return_all_embeddings:
            return {
                'paragraph': paragraph_embeddings,
                'figure': figure_embeddings,
                'table': table_embeddings,
                'cited_paper': cited_paper_embeddings
            }
        
        if target_node_type == 'figure' and figure_embeddings is not None:
            target_embedding = figure_embeddings[target_node_idx]
        elif target_node_type == 'table' and table_embeddings is not None:
            target_embedding = table_embeddings[target_node_idx]
        else:
            raise ValueError(f"Invalid target node type: {target_node_type}")
        
        target_embedding_expanded = target_embedding.unsqueeze(0).repeat(paragraph_embeddings.size(0), 1)
        combined_features = torch.cat([target_embedding_expanded, paragraph_embeddings], dim=1)
        scores = self.classifier(combined_features).squeeze(-1)
        
        return scores

def load_test_data(paper_graphs_path, target_info_path):
    """Load test data"""
    paper_graphs = torch.load(paper_graphs_path, weights_only=False)
    target_info = torch.load(target_info_path, weights_only=False)
    return paper_graphs, target_info

def top_k_accuracy(predictions, true_paragraph_idx, k=5):
    """
    Calculate top-k accuracy for paragraph prediction
    
    Args:
        predictions: torch.Tensor of shape [num_paragraphs] with scores
        true_paragraph_idx: int, the correct paragraph index
        k: int, top-k value
    
    Returns:
        bool: True if correct paragraph is in top-k predictions
    """
    top_k_indices = torch.topk(predictions, k=min(k, len(predictions))).indices
    return true_paragraph_idx in top_k_indices.cpu().numpy()

def calculate_metrics(predictions, true_paragraph_idx, k_values=[1, 3, 5]):
    """Calculate various metrics"""
    metrics = {}
    
    for k in k_values:
        metrics[f'top_{k}_accuracy'] = top_k_accuracy(predictions, true_paragraph_idx, k)
    
    # Calculate MRR (Mean Reciprocal Rank)
    sorted_indices = torch.argsort(predictions, descending=True)
    rank = (sorted_indices == true_paragraph_idx).nonzero(as_tuple=True)[0].item() + 1
    metrics['mrr'] = 1.0 / rank
    metrics['rank'] = rank
    
    return metrics

# Load test data
TEST_PAPER_GRAPHS_PATH = "/path/to/test_paper_graphs.pt"
TEST_TARGET_INFO_PATH = "/path/to/test_target_info.pt"

test_graphs, test_targets = load_test_data(TEST_PAPER_GRAPHS_PATH, TEST_TARGET_INFO_PATH)

# Load model
MODEL_PATH = "/path/to/figure_table_insertion_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FigureTableInsertionModel(
    in_channels=768, 
    hidden_channels=128, 
    out_channels=768, 
    metadata=test_graphs[0].metadata()
)
model.load_state_dict(torch.load(MODEL_PATH))
model.to(device)
model.eval()

# Evaluation
all_results = []
aggregate_metrics = {
    'top_1_accuracy': [],
    'top_3_accuracy': [],
    'top_5_accuracy': [],
    'mrr': [],
    'rank': []
}

with torch.no_grad():
    for paper_idx, (graph, target_data) in enumerate(zip(test_graphs, test_targets)):
        graph = graph.to(device)
        
        paper_results = {
            'paper_id': paper_idx,
            'predictions': []
        }
        
        # Process each target figure/table in this paper
        for target_idx, true_paragraph_idx in zip(target_data['target_nodes'], target_data['ground_truth_paragraphs']):
            # You need to determine the target node type from your data
            # This is a placeholder - adjust based on your data structure
            target_node_type = 'figure'  # or 'table'
            
            try:
                # Get prediction scores for all paragraphs
                scores = model(graph.x_dict, graph.edge_index_dict, target_node_type, target_idx)
                
                # Calculate metrics
                metrics = calculate_metrics(scores, true_paragraph_idx)
                
                # Store results
                prediction_result = {
                    'target_node_idx': target_idx,
                    'target_node_type': target_node_type,
                    'true_paragraph_idx': true_paragraph_idx,
                    'predicted_scores': scores.cpu().numpy().tolist(),
                    'top_5_predicted_paragraphs': torch.topk(scores, k=min(5, len(scores))).indices.cpu().numpy().tolist(),
                    'metrics': metrics
                }
                
                paper_results['predictions'].append(prediction_result)
                
                # Aggregate metrics
                for key in aggregate_metrics:
                    aggregate_metrics[key].append(metrics[key])
                
            except Exception as e:
                print(f"Error processing paper {paper_idx}, target {target_idx}: {e}")
                continue
        
        all_results.append(paper_results)
        
        # Print progress
        if (paper_idx + 1) % 100 == 0:
            print(f"Processed {paper_idx + 1}/{len(test_graphs)} papers")

# Calculate final aggregate metrics
final_metrics = {}
for key, values in aggregate_metrics.items():
    if values:
        final_metrics[f'avg_{key}'] = np.mean(values)
        final_metrics[f'std_{key}'] = np.std(values)

print("\n=== Final Evaluation Results ===")
for key, value in final_metrics.items():
    print(f"{key}: {value:.4f}")

# Save detailed results
output_dir = "/path/to/output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save all predictions and metrics
with open(os.path.join(output_dir, "figure_table_insertion_results.json"), 'w') as f:
    json.dump({
        'final_metrics': final_metrics,
        'detailed_results': all_results
    }, f, indent=2)

print(f"\nDetailed results saved to {os.path.join(output_dir, 'figure_table_insertion_results.json')}")

# Print summary statistics
print(f"\nTotal predictions made: {len(aggregate_metrics['top_1_accuracy'])}")
print(f"Average Top-1 Accuracy: {final_metrics['avg_top_1_accuracy']:.4f}")
print(f"Average Top-3 Accuracy: {final_metrics['avg_top_3_accuracy']:.4f}")
print(f"Average Top-5 Accuracy: {final_metrics['avg_top_5_accuracy']:.4f}")
print(f"Average MRR: {final_metrics['avg_mrr']:.4f}")
print(f"Average Rank: {final_metrics['avg_rank']:.2f}")