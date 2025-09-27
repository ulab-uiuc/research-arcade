import torch
import argparse
import os
from pathlib import Path
import random
import glob

def merge_graph_files(graph_files, target_files, output_dir):
    """
    Merge multiple graph and target files into single files
    
    Args:
        graph_files: List of paths to *_graphs.pt files
        target_files: List of paths to *_targets.pt files
        output_dir: Directory to save merged files
    """
    all_graphs = []
    all_targets = []
    
    print("Loading and merging files...")
    for graph_file, target_file in zip(graph_files, target_files):
        print(f"  Loading {graph_file}")
        graphs = torch.load(graph_file, weights_only=False)
        targets = torch.load(target_file, weights_only=False)
        
        all_graphs.extend(graphs)
        all_targets.extend(targets)
        
        print(f"    Added {len(graphs)} graphs")
    
    print(f"\nTotal graphs: {len(all_graphs)}")
    
    # Save merged files
    os.makedirs(output_dir, exist_ok=True)
    merged_graphs_path = os.path.join(output_dir, "merged_graphs.pt")
    merged_targets_path = os.path.join(output_dir, "merged_targets.pt")
    
    torch.save(all_graphs, merged_graphs_path)
    torch.save(all_targets, merged_targets_path)
    
    print(f"\nMerged files saved:")
    print(f"  Graphs: {merged_graphs_path}")
    print(f"  Targets: {merged_targets_path}")
    
    return all_graphs, all_targets

def split_dataset(graphs, targets, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, 
                  output_dir="./data", shuffle=True, seed=42):
    """
    Split graphs and targets into train/val/test sets
    
    Args:
        graphs: List of graph objects
        targets: List of target dictionaries
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        output_dir: Directory to save split datasets
        shuffle: Whether to shuffle before splitting
        seed: Random seed for reproducibility
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    n_samples = len(graphs)
    indices = list(range(n_samples))
    
    # Shuffle if requested
    if shuffle:
        random.seed(seed)
        random.shuffle(indices)
    
    # Calculate split points
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    # Split indices
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # Split data
    train_graphs = [graphs[i] for i in train_indices]
    train_targets = [targets[i] for i in train_indices]
    
    val_graphs = [graphs[i] for i in val_indices]
    val_targets = [targets[i] for i in val_indices]
    
    test_graphs = [graphs[i] for i in test_indices]
    test_targets = [targets[i] for i in test_indices]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_graphs)} samples ({len(train_graphs)/n_samples*100:.1f}%)")
    print(f"  Val:   {len(val_graphs)} samples ({len(val_graphs)/n_samples*100:.1f}%)")
    print(f"  Test:  {len(test_graphs)} samples ({len(test_graphs)/n_samples*100:.1f}%)")
    
    # Save splits
    os.makedirs(output_dir, exist_ok=True)
    
    torch.save(train_graphs, os.path.join(output_dir, "train_graphs.pt"))
    torch.save(train_targets, os.path.join(output_dir, "train_targets.pt"))
    
    torch.save(val_graphs, os.path.join(output_dir, "val_graphs.pt"))
    torch.save(val_targets, os.path.join(output_dir, "val_targets.pt"))
    
    torch.save(test_graphs, os.path.join(output_dir, "test_graphs.pt"))
    torch.save(test_targets, os.path.join(output_dir, "test_targets.pt"))
    
    print(f"\nSplit datasets saved to {output_dir}")
    
    return {
        'train': (train_graphs, train_targets),
        'val': (val_graphs, val_targets),
        'test': (test_graphs, test_targets)
    }

def print_dataset_statistics(graphs, targets, name="Dataset"):
    """Print statistics about the dataset"""
    print(f"\n{name} Statistics:")
    print(f"  Number of papers: {len(graphs)}")
    
    total_figures = sum(len(t['figure_nodes']) for t in targets)
    total_tables = sum(len(t['table_nodes']) for t in targets)
    
    print(f"  Total figures: {total_figures}")
    print(f"  Total tables: {total_tables}")
    print(f"  Total targets: {total_figures + total_tables}")
    
    if graphs:
        # Count node types
        total_paragraphs = sum(g['paragraph'].x.shape[0] for g in graphs)
        total_sections = sum(g['section'].x.shape[0] if 'section' in g.node_types else 0 for g in graphs)
        
        print(f"  Total paragraphs: {total_paragraphs}")
        print(f"  Total sections: {total_sections}")
        
        # Average stats
        avg_paras = total_paragraphs / len(graphs)
        avg_figs = total_figures / len(graphs)
        avg_tables = total_tables / len(graphs)
        
        print(f"  Avg paragraphs/paper: {avg_paras:.1f}")
        print(f"  Avg figures/paper: {avg_figs:.1f}")
        print(f"  Avg tables/paper: {avg_tables:.1f}")

def main():
    parser = argparse.ArgumentParser(description='Merge and split graph datasets')
    parser.add_argument('files', nargs='*', help='Graph files to merge (can specify multiple files directly)')
    parser.add_argument('--input_pattern', type=str, default=None,
                       help='Pattern to match graph files (use glob pattern)')
    parser.add_argument('--output_dir', type=str, default='./data/insertion_processed',
                       help='Output directory for split datasets')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no_shuffle', action='store_true', help='Do not shuffle before splitting')
    parser.add_argument('--merge_only', action='store_true', help='Only merge files without splitting')
    args = parser.parse_args()
    
    # Determine which files to process
    graph_files = []
    
    # If files are provided as positional arguments
    if args.files:
        graph_files = args.files
    # If input_pattern is provided
    elif args.input_pattern:
        graph_files = sorted(glob.glob(args.input_pattern))
    else:
        # Default pattern
        graph_files = sorted(glob.glob('./json/graph_matrix_*_graphs.pt'))
    
    if not graph_files:
        print(f"No files found!")
        print("Please specify files either as:")
        print("  python script.py file1.pt file2.pt file3.pt")
        print("Or with a pattern:")
        print("  python script.py --input_pattern './json/graph_matrix_*_graphs.pt'")
        return
    
    # Filter out non-existent graph files and check for corresponding target files
    valid_graph_files = []
    valid_target_files = []
    
    print("Checking files...")
    for graph_file in graph_files:
        if not os.path.exists(graph_file):
            print(f"  ⚠️  Skipping (not found): {graph_file}")
            continue
        
        target_file = graph_file.replace('_graphs.pt', '_targets.pt')
        if not os.path.exists(target_file):
            print(f"  ⚠️  Skipping {graph_file} (target file not found): {target_file}")
            continue
        
        valid_graph_files.append(graph_file)
        valid_target_files.append(target_file)
        print(f"  ✓ Valid: {graph_file}")
    
    if not valid_graph_files:
        print("\nError: No valid graph/target file pairs found!")
        return
    
    print(f"\nFound {len(valid_graph_files)} valid file pairs to process")
    
    # Merge files
    all_graphs, all_targets = merge_graph_files(valid_graph_files, valid_target_files, args.output_dir)
    
    # Print merged statistics
    print_dataset_statistics(all_graphs, all_targets, "Merged Dataset")
    
    if args.merge_only:
        print("\nMerge complete (no splitting performed)")
        return
    
    # Split dataset
    splits = split_dataset(
        all_graphs, 
        all_targets,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        output_dir=args.output_dir,
        shuffle=not args.no_shuffle,
        seed=args.seed
    )
    
    # Print statistics for each split
    print_dataset_statistics(splits['train'][0], splits['train'][1], "Training Set")
    print_dataset_statistics(splits['val'][0], splits['val'][1], "Validation Set")
    print_dataset_statistics(splits['test'][0], splits['test'][1], "Test Set")
    
    print("\nDone! You can now use these files for training:")
    print(f"  --train_graphs {os.path.join(args.output_dir, 'train_graphs.pt')}")
    print(f"  --train_targets {os.path.join(args.output_dir, 'train_targets.pt')}")
    print(f"  --val_graphs {os.path.join(args.output_dir, 'val_graphs.pt')}")
    print(f"  --val_targets {os.path.join(args.output_dir, 'val_targets.pt')}")
    print(f"  --test_graphs {os.path.join(args.output_dir, 'test_graphs.pt')}")
    print(f"  --test_targets {os.path.join(args.output_dir, 'test_targets.pt')}")

if __name__ == "__main__":
    main()