import torch
import argparse

def convert_targets_format(old_targets):
    """
    Convert from old format:
    {
        'figure_nodes': [2, 3, 5, ...],
        'table_nodes': [1, 2, 3, ...],
        'figure_ground_truth': [44, 44, 44, ...],
        'table_ground_truth': [44, 44, 44, ...]
    }
    
    To new format:
    {
        'targets': [
            {'type': 'figure', 'node_idx': 2, 'paragraph_idx': 44},
            {'type': 'figure', 'node_idx': 3, 'paragraph_idx': 44},
            {'type': 'table', 'node_idx': 1, 'paragraph_idx': 44},
            ...
        ]
    }
    """
    new_targets = []
    
    for target_data in old_targets:
        new_target = {'targets': []}
        
        # Convert figures
        if 'figure_nodes' in target_data and 'figure_ground_truth' in target_data:
            figure_nodes = target_data['figure_nodes']
            figure_gt = target_data['figure_ground_truth']
            
            for node_idx, para_idx in zip(figure_nodes, figure_gt):
                new_target['targets'].append({
                    'type': 'figure',
                    'node_idx': node_idx,
                    'paragraph_idx': para_idx
                })
        
        # Convert tables
        if 'table_nodes' in target_data and 'table_ground_truth' in target_data:
            table_nodes = target_data['table_nodes']
            table_gt = target_data['table_ground_truth']
            
            for node_idx, para_idx in zip(table_nodes, table_gt):
                new_target['targets'].append({
                    'type': 'table',
                    'node_idx': node_idx,
                    'paragraph_idx': para_idx
                })
        
        new_targets.append(new_target)
    
    return new_targets

def main():
    parser = argparse.ArgumentParser(description='Convert target format from old to new')
    parser.add_argument('--input', type=str, required=True, help='Input targets file (old format)')
    parser.add_argument('--output', type=str, required=True, help='Output targets file (new format)')
    args = parser.parse_args()
    
    print(f"Loading old format targets from: {args.input}")
    old_targets = torch.load(args.input, weights_only=False)
    
    print(f"Number of samples: {len(old_targets)}")
    print(f"\nOld format example:")
    print(f"  Keys: {list(old_targets[0].keys())}")
    print(f"  Figure nodes: {len(old_targets[0].get('figure_nodes', []))}")
    print(f"  Table nodes: {len(old_targets[0].get('table_nodes', []))}")
    
    print("\nConverting to new format...")
    new_targets = convert_targets_format(old_targets)
    
    print(f"\nNew format example:")
    print(f"  Keys: {list(new_targets[0].keys())}")
    print(f"  Number of targets: {len(new_targets[0]['targets'])}")
    if new_targets[0]['targets']:
        print(f"  First target: {new_targets[0]['targets'][0]}")
    
    # Verify conversion
    total_targets = sum(len(t['targets']) for t in new_targets)
    valid_targets = sum(
        1 for t in new_targets 
        for target in t['targets'] 
        if target.get('paragraph_idx') is not None
    )
    
    print(f"\nConversion statistics:")
    print(f"  Total targets: {total_targets}")
    print(f"  Valid targets with ground truth: {valid_targets}")
    print(f"  Success rate: {valid_targets/total_targets*100:.1f}%")
    
    # Save new format
    torch.save(new_targets, args.output)
    print(f"\n✓ Saved new format to: {args.output}")
    
    if valid_targets == total_targets:
        print("✓ All targets have ground truth - ready for training!")
    else:
        print(f"⚠️  Warning: {total_targets - valid_targets} targets missing ground truth")

if __name__ == "__main__":
    main()