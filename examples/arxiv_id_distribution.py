import json
import math

# Load arxiv ids from file
with open("./json/arxiv_id_insertion.json", "r") as f:
    arxiv_ids = json.load(f)

# Number of splits
num_splits = 7
chunk_size = math.ceil(len(arxiv_ids) / num_splits)

# Split and save
for i in range(num_splits):
    start = i * chunk_size
    end = start + chunk_size
    chunk = arxiv_ids[start:end]
    
    filename = f"./json/arxiv_id_insertion_{i+1}.json"
    with open(filename, "w") as f:
        json.dump(chunk, f, indent=2)
    
    print(f"Saved {len(chunk)} ids to {filename}")
