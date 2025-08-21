import json
import math

# Load arxiv ids from file
with open("./arxiv_id_list.json", "r") as f:
    arxiv_ids = json.load(f)

# Number of splits
num_splits = 60
chunk_size = math.ceil(len(arxiv_ids) / num_splits)

# Split and save
for i in range(num_splits):
    start = i * chunk_size
    end = start + chunk_size
    chunk = arxiv_ids[start:end]
    
    filename = f"./arxiv_id_openreview/arxiv_id_list{i+1}.json"
    with open(filename, "w") as f:
        json.dump(chunk, f, indent=2)
    
    print(f"Saved {len(chunk)} ids to {filename}")
