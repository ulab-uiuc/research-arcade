import torch
import argparse
import os

def add_ground_truth_from_edges(graphs, targets):
    """
    Attempt to infer ground truth paragraph indices from graph edges.
    
    This assumes your graphs have edges like:
    - ('figure', 'cites', 'paragraph') or ('figure', 'references', 'paragraph')
    - ('table', 'cites', 'paragraph') or ('table', 'references', 'paragraph')
    
    The connected paragraph is used as ground truth.
    """
    print("Attempting to infer ground truth from graph edges...")
    
    new_targets = []
    stats = {'added': 0, 'failed': 0, 'already_exists': 0}
    
    for idx, (graph, target_data) in enumerate(zip(graphs, targets)):
        new_target = {'targets': []}
        
        # Handle old format
        if 'targets' not in target_data:
            # Convert old format
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
        else:
            targets_list = target_data['targets']
        
        # Process each target
        for t in targets_list:
            target_type = t['type']
            node_idx = t['node_idx']
            para_idx = t.get('paragraph_idx')
            
            # If already has ground truth, keep it
            if para_idx is not None:
                new_target['targets'].append(t)
                stats['already_exists'] += 1
                continue
            
            # Try to find ground truth from edges
            found_para_idx = None
            edge_index_dict = graph.edge_index_dict if hasattr(graph, 'edge_index_dict') else {}
            
            # Check different possible edge types
            possible_edges = [
                (target_type, 'cites', 'paragraph'),
                (target_type, 'references', 'paragraph'),
                (target_type, 'related_to', 'paragraph'),
                (target_type, 'inserted_after', 'paragraph'),
            ]
            
            for edge_type in possible_edges:
                if edge_type in edge_index_dict:
                    edge_index = edge_index_dict[edge_type]
                    # Find edges where source is our target node
                    mask = edge_index[0] == node_idx
                    if mask.any():
                        # Get the first connected paragraph
                        found_para_idx = edge_index[1][mask][0].item()
                        break
            
            if found_para_idx is not None:
                new_target['targets'].append({
                    'type': target_type,
                    'node_idx': node_idx,
                    'paragraph_idx': found_para_idx
                })
                stats['added'] += 1
            else:
                # No ground truth found, skip or add with None
                new_target['targets'].append({
                    'type': target_type,
                    'node_idx': node_idx,
                    'paragraph_idx': None
                })
                stats['failed'] += 1
        
        new_targets.append(new_target)
    
    print(f"\nGround Truth Inference Results:")
    print(f"  Already had ground truth: {stats['already_exists']}")
    print(f"  Successfully inferred: {stats['added']}")
    print(f"  Could not infer: {stats['failed']}")
    
    return new_targets, stats

def add_random_ground_truth(graphs, targets):
    """
    Add random paragraph indices as ground truth (for testing only!)
    
    WARNING: This is only for testing the training pipeline.
    The model won't learn meaningful patterns with random labels!
    """
    print("⚠️  WARNING: Adding RANDOM ground truth for testing purposes only!")
    print("    The model will NOT learn meaningful patterns with random labels!\n")
    
    new_targets = []
    
    for idx, (graph, target_data) in enumerate(zip(graphs, targets)):
        new_target = {'targets': []}
        
        # Get number of paragraphs in this graph
        num_paragraphs = graph['paragraph'].x.shape[0]
        
        # Handle old format
        if 'targets' not in target_data:
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
        else:
            targets_list = target_data['targets']
        
        # Add random paragraph index
        for t in targets_list:
            if t.get('paragraph_idx') is None:
                # Assign random paragraph
                import random
                t['paragraph_idx'] = random.randint(0, num_paragraphs - 1)
            new_target['targets'].append(t)
        
        new_targets.append(new_target)
    
    return new_targets

def main():
    parser = argparse.ArgumentParser(description='Fix missing ground truth in target data')
    parser.add_argument('--graphs', type=str, required=True, help='Path to graphs file')
    parser.add_argument('--targets', type=str, required=True, help='Path to targets file')
    parser.add_argument('--output_targets', type=str, required=True, help='Output path for fixed targets')
    parser.add_argument('--method', type=str, choices=['infer', 'random'], default='infer',
                       help='Method to add ground truth: infer from edges or random (testing only)')
    args = parser.parse_args()
    
    print("Loading data...")
    graphs = torch.load(args.graphs, weights_only=False)
    targets = torch.load(args.targets, weights_only=False)
    
    print(f"Loaded {len(graphs)} graphs and {len(targets)} targets\n")
    
    if args.method == 'infer':
        new_targets, stats = add_ground_truth_from_edges(graphs, targets)
        
        if stats['failed'] > 0:
            print(f"\n⚠️  Warning: {stats['failed']} targets could not be inferred")
            print("    Consider checking your graph structure or using original data source")
    else:
        new_targets = add_random_ground_truth(graphs, targets)
    
    # Save fixed targets
    torch.save(new_targets, args.output_targets)
    print(f"\n✓ Saved fixed targets to: {args.output_targets}")
    
    # Verify
    print("\nVerifying fixed data...")
    valid_count = sum(
        1 for target_data in new_targets
        for t in target_data.get('targets', [])
        if t.get('paragraph_idx') is not None
    )
    total_count = sum(
        len(target_data.get('targets', []))
        for target_data in new_targets
    )
    
    print(f"Valid targets: {valid_count}/{total_count}")
    
    if valid_count == 0:
        print("\n❌ Still no valid ground truth! You need to:")
        print("   1. Check your original data source")
        print("   2. Manually add ground truth paragraph indices")
        print("   3. Ensure your graph has proper edges connecting figures/tables to paragraphs")
    elif valid_count < total_count:
        print(f"\n⚠️  Only {valid_count}/{total_count} have ground truth")
        print("   Training will only use targets with valid paragraph indices")
    else:
        print("\n✓ All targets now have ground truth!")

if __name__ == "__main__":
    main()