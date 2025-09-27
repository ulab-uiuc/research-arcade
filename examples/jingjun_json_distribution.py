import json

def distribute_and_save(input_file, num_groups):
    # Read from file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Initialize groups
    groups = [[] for _ in range(num_groups)]
    
    # Distribute using modulo
    for i, item in enumerate(data):
        group_index = i % num_groups
        groups[group_index].append(item)
    
    # Save each group to separate file
    for i, group in enumerate(groups):
        with open(f'./json/arxiv_old_{i+1}.json', 'w') as f:
            json.dump(group, f, indent=2)
    
    print(f"Saved {num_groups} files: arxiv_old_1.json to arxiv_old_{num_groups}.json")

# Usage
distribute_and_save('./json/arxiv_old.json', 10)