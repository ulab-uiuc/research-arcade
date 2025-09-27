import os
import json
import sys

source_path = "./data/paragraph_generation/paragraph_generation_training.jsonl"
existing_work_path = "./data/paragraph_generation/paragraph_generation_training_preprocessed.jsonl"

# Check if files exist
file_exists = os.path.exists(source_path) and os.path.exists(existing_work_path)
print(f"Files exist: {file_exists}")

if not file_exists:
    print("Required files not found!")
    exit()


# Distribute into 4 different files
n = 4

# Count existing work done
count_existing = 0
if os.path.exists(existing_work_path):
    with open(existing_work_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                count_existing += 1

print(f"Existing work completed: {count_existing}")

# Load all data from source file
json_array = []
with open(source_path, 'r') as source_file:
    for line in source_file:
        line = line.strip()
        if line:
            json_array.append(json.loads(line))

count_total = len(json_array)
count_remaining = count_total - count_existing
chunk_size = count_remaining // n

print(f"Total items: {count_total}")
print(f"Remaining items: {count_remaining}")
print(f"Chunk size per GPU: {chunk_size}")

# Create directory if it doesn't exist
os.makedirs("./data/paragraph_generation", exist_ok=True)

# Distribute data into separate files
for i in range(n):
    start_index = count_existing + (chunk_size * i)
    
    if i == n - 1:  # Last chunk gets any remaining items
        end_index = count_total
    else:
        end_index = start_index + chunk_size
    
    json_objs = json_array[start_index:end_index]
    
    save_path = f"./data/paragraph_generation/paragraph_generation_training_gpu_{i}.jsonl"
    
    with open(save_path, 'w') as output_file:
        for obj in json_objs:
            output_file.write(json.dumps(obj) + '\n')
    
    print(f"GPU {i}: {len(json_objs)} items saved to {save_path}")
    
print("Distribution complete!")