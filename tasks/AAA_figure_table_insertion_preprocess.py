import json
import numpy as np
import torch
from torch_geometric.data import HeteroData
from collections import defaultdict
from tqdm import tqdm

def convert_hierarchical_json_to_graph(paper_data):
    """
    Convert hierarchical JSON paper data to graph format
    
    Args:
        paper_data: Dictionary with hierarchical structure containing:
            - title, title_embedding
            - sections: list of {section_title, section_embedding, paragraphs: [...]}
            - paragraphs: list of {paragraph_id, content, paragraph_embedding, figures: [...], tables: [...]}
    
    Returns:
        HeteroData object with node features and edge indices
    """
    data = HeteroData()
    
    # Storage for nodes and edges
    node_mappings = {
        'paragraph': {},
        'figure': {},
        'table': {},
        'section': {},
        'paper': {}  # Add paper node type
    }
    
    node_features = {
        'paragraph': [],
        'figure': [],
        'table': [],
        'section': [],
        'paper': []  # Add paper node features
    }
    
    edge_lists = defaultdict(list)
    
    # Add paper node (single node representing the entire paper)
    paper_node_idx = 0
    node_mappings['paper']['paper_0'] = paper_node_idx
    if 'title_embedding' in paper_data:
        node_features['paper'].append(paper_data['title_embedding'])
    else:
        # If no title embedding, use zero embedding as placeholder
        node_features['paper'].append([0.0] * 1024)  # Adjust dimension as needed
    
    # Process each section
    for section_idx, section in enumerate(paper_data['sections']):
        # Add section node
        section_id = f"section_{section_idx}"
        section_node_idx = len(node_mappings['section'])
        node_mappings['section'][section_id] = section_node_idx
        node_features['section'].append(section['section_embedding'])
        
        # Connect section to paper (bidirectional)
        edge_lists[('section', 'section-to-paper', 'paper')].append([section_node_idx, paper_node_idx])
        edge_lists[('paper', 'paper-to-section', 'section')].append([paper_node_idx, section_node_idx])
        
        # Process paragraphs in this section and track their indices
        section_para_indices = []
        
        for para_local_idx, para in enumerate(section['paragraphs']):
            para_id = para['paragraph_id']
            para_node_idx = len(node_mappings['paragraph'])
            node_mappings['paragraph'][para_id] = para_node_idx
            node_features['paragraph'].append(para['paragraph_embedding'])
            section_para_indices.append(para_node_idx)
            
            # Connect paragraph to section (bidirectional)
            edge_lists[('paragraph', 'paragraph-to-section', 'section')].append([para_node_idx, section_node_idx])
            edge_lists[('section', 'section-to-paragraph', 'paragraph')].append([section_node_idx, para_node_idx])
            
            # Process figures in this paragraph
            for fig in para['figures']:
                fig_id = fig['label']
                fig_node_idx = len(node_mappings['figure'])
                node_mappings['figure'][fig_id] = fig_node_idx
                node_features['figure'].append(fig['figure_embedding'])
                
                # Connect figure to paragraph (bidirectional)
                edge_lists[('figure', 'figure-to-paragraph', 'paragraph')].append([fig_node_idx, para_node_idx])
                edge_lists[('paragraph', 'paragraph-to-figure', 'figure')].append([para_node_idx, fig_node_idx])
            
            # Process tables in this paragraph
            for table in para['tables']:
                table_id = table['label']
                table_node_idx = len(node_mappings['table'])
                node_mappings['table'][table_id] = table_node_idx
                node_features['table'].append(table['table_embedding'])
                
                # Connect table to paragraph (bidirectional)
                edge_lists[('table', 'table-to-paragraph', 'paragraph')].append([table_node_idx, para_node_idx])
                edge_lists[('paragraph', 'paragraph-to-table', 'table')].append([para_node_idx, table_node_idx])
        
        # Connect consecutive paragraphs within section based on their order
        for i in range(len(section_para_indices) - 1):
            curr_para_idx = section_para_indices[i]
            next_para_idx = section_para_indices[i + 1]
            # Bidirectional paragraph connections
            edge_lists[('paragraph', 'paragraph-to-paragraph', 'paragraph')].append([curr_para_idx, next_para_idx])
            edge_lists[('paragraph', 'paragraph-to-paragraph', 'paragraph')].append([next_para_idx, curr_para_idx])
    
    # Convert node features to tensors
    for node_type, features in node_features.items():
        if features:
            data[node_type].x = torch.tensor(features, dtype=torch.float32)
    
    # Define expected edge types (including paper-section connections)
    expected_edge_types = [
        ('paragraph', 'paragraph-to-paragraph', 'paragraph'),
        ('paragraph', 'paragraph-to-figure', 'figure'),
        ('paragraph', 'paragraph-to-table', 'table'),
        ('paragraph', 'paragraph-to-section', 'section'),
        ('figure', 'figure-to-paragraph', 'paragraph'),
        ('table', 'table-to-paragraph', 'paragraph'),
        ('section', 'section-to-paragraph', 'paragraph'),
        ('section', 'section-to-paper', 'paper'),  # New
        ('paper', 'paper-to-section', 'section'),  # New
    ]
    
    # Convert edge lists to edge indices with empty tensors for missing edge types
    for edge_type in expected_edge_types:
        if edge_type in edge_lists and edge_lists[edge_type]:
            data[edge_type].edge_index = torch.tensor(edge_lists[edge_type], dtype=torch.long).t().contiguous()
        else:
            data[edge_type].edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    # Store mappings for later use
    data.node_mappings = node_mappings
    
    # Add compatibility attributes for training code
    data.x_dict = {node_type: data[node_type].x for node_type in data.node_types}
    data.edge_index_dict = {edge_type: data[edge_type].edge_index for edge_type in data.edge_types}
    
    return data

def is_empty_embedding(embedding, threshold=1e-6):
    """Check if an embedding is all zeros or near-zero"""
    if isinstance(embedding, list):
        embedding = torch.tensor(embedding)
    return torch.abs(embedding).sum().item() < threshold

def create_figure_table_insertion_dataset(json_path, output_path):
    """
    Create dataset for figure/table insertion task from hierarchical JSON
    
    Args:
        json_path: Path to hierarchical_papers_data_preprocessed.json
        output_path: Path to save processed graphs
    """
    # Load data
    with open(json_path, 'r') as f:
        papers_data = json.load(f)
    
    paper_graphs = []
    target_info = []
    
    skipped_figures = 0
    skipped_tables = 0
    
    for paper_idx, paper in enumerate(tqdm(papers_data, desc="Converting papers to graphs")):
        try:
            # Convert to graph
            graph = convert_hierarchical_json_to_graph(paper)
            
            # Collect target information (figures and tables) - ONLY non-empty embeddings
            targets = {
                'figure_nodes': [],
                'table_nodes': [],
                'figure_ground_truth': [],
                'table_ground_truth': []
            }
            
            # Extract ground truth: which paragraph each figure/table belongs to
            for section in paper['sections']:
                for para in section['paragraphs']:
                    para_idx = graph.node_mappings['paragraph'][para['paragraph_id']]
                    
                    # Record figures - ONLY if they have valid (non-zero) embeddings
                    for fig in para['figures']:
                        if not is_empty_embedding(fig['figure_embedding']):
                            fig_idx = graph.node_mappings['figure'][fig['label']]
                            targets['figure_nodes'].append(fig_idx)
                            targets['figure_ground_truth'].append(para_idx)
                        else:
                            skipped_figures += 1
                    
                    # Record tables - ONLY if they have valid (non-zero) embeddings
                    for table in para['tables']:
                        if not is_empty_embedding(table['table_embedding']):
                            table_idx = graph.node_mappings['table'][table['label']]
                            targets['table_nodes'].append(table_idx)
                            targets['table_ground_truth'].append(para_idx)
                        else:
                            skipped_tables += 1
            
            paper_graphs.append(graph)
            target_info.append(targets)
            
        except Exception as e:
            print(f"Error processing paper {paper_idx} ({paper.get('arxiv_id', 'unknown')}): {e}")
            continue
    
    # Save processed data
    torch.save(paper_graphs, f"{output_path}_graphs.pt")
    torch.save(target_info, f"{output_path}_targets.pt")
    
    print(f"Processed {len(paper_graphs)} papers")
    print(f"Skipped {skipped_figures} figures with empty embeddings")
    print(f"Skipped {skipped_tables} tables with empty embeddings")
    print(f"Saved to {output_path}_graphs.pt and {output_path}_targets.pt")
    
    return paper_graphs, target_info

def create_attribute_matrix_format(json_path):
    """
    Alternative: Create attribute matrix format matching the old preprocessing code
    This creates adjacency matrices and attribute matrices
    """
    with open(json_path, 'r') as f:
        papers_data = json.load(f)
    
    all_adjacency_matrices = []
    all_attribute_matrices = []
    all_node_types = []
    all_target_nodes = []
    all_ground_truths = []
    
    for paper in tqdm(papers_data, desc="Creating attribute matrices"):
        # Build node list and adjacency matrix
        nodes = []
        node_type_list = []
        adjacency_dict = defaultdict(list)
        
        # Add all nodes
        global_idx = 0
        node_id_to_idx = {}
        
        # Add paragraphs
        for section in paper['sections']:
            for para in section['paragraphs']:
                node_id_to_idx[para['paragraph_id']] = global_idx
                nodes.append({
                    'type': 'paragraph',
                    'embedding': para['paragraph_embedding'],
                    'text': para['content']
                })
                node_type_list.append('paragraph')
                global_idx += 1
        
        # Add figures
        figure_targets = []
        figure_ground_truths = []
        for section in paper['sections']:
            for para in section['paragraphs']:
                para_idx = node_id_to_idx[para['paragraph_id']]
                for fig in para['figures']:
                    fig_id = f"fig_{fig['label']}"
                    fig_idx = global_idx
                    node_id_to_idx[fig_id] = fig_idx
                    nodes.append({
                        'type': 'figure',
                        'embedding': fig['figure_embedding'],
                        'text': fig['caption']
                    })
                    node_type_list.append('figure')
                    
                    # Add edges
                    adjacency_dict[fig_idx].append(para_idx)
                    adjacency_dict[para_idx].append(fig_idx)
                    
                    # Record target
                    figure_targets.append(fig_idx)
                    figure_ground_truths.append(para_idx)
                    
                    global_idx += 1
        
        # Add tables (similar to figures)
        table_targets = []
        table_ground_truths = []
        for section in paper['sections']:
            for para in section['paragraphs']:
                para_idx = node_id_to_idx[para['paragraph_id']]
                for table in para['tables']:
                    table_id = f"table_{table['label']}"
                    table_idx = global_idx
                    node_id_to_idx[table_id] = table_idx
                    nodes.append({
                        'type': 'table',
                        'embedding': table['table_embedding'],
                        'text': table.get('caption', '')
                    })
                    node_type_list.append('table')
                    
                    # Add edges
                    adjacency_dict[table_idx].append(para_idx)
                    adjacency_dict[para_idx].append(table_idx)
                    
                    # Record target
                    table_targets.append(table_idx)
                    table_ground_truths.append(para_idx)
                    
                    global_idx += 1
        
        # Create adjacency matrix
        num_nodes = len(nodes)
        adjacency_matrix = np.zeros((num_nodes, num_nodes))
        for src, dsts in adjacency_dict.items():
            for dst in dsts:
                adjacency_matrix[src, dst] = 1
        
        # Create attribute matrix (embeddings)
        attribute_matrix = np.array([node['embedding'] for node in nodes])
        
        all_adjacency_matrices.append(adjacency_matrix)
        all_attribute_matrices.append(attribute_matrix)
        all_node_types.append(node_type_list)
        all_target_nodes.append(figure_targets + table_targets)
        all_ground_truths.append(figure_ground_truths + table_ground_truths)
    
    return {
        'adjacency_matrices': all_adjacency_matrices,
        'attribute_matrices': all_attribute_matrices,
        'node_types': all_node_types,
        'target_nodes': all_target_nodes,
        'ground_truth_paragraphs': all_ground_truths
    }

# Usage example
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', type=str, required=True,
                       help='Path to hierarchical_papers_data_preprocessed.json')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Output path for processed graphs')
    parser.add_argument('--format', choices=['hetero', 'matrix'], default='hetero',
                       help='Output format: hetero (HeteroData) or matrix (adjacency/attribute matrices)')
    args = parser.parse_args()
    
    if args.format == 'hetero':
        # Create HeteroData format (recommended)
        paper_graphs, target_info = create_figure_table_insertion_dataset(
            args.input_json, 
            args.output_path
        )
    else:
        # Create matrix format (for compatibility with old code)
        data_dict = create_attribute_matrix_format(args.input_json)
        np.save(args.output_path + '_matrices.npy', data_dict)
        print(f"Saved matrices to {args.output_path}_matrices.npy")