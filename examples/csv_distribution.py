import json
import math

# Load arxiv ids from CSV-like file
with open("./csv/arxiv_ids_ct_csv.csv", "r") as f:
    arxiv_ids = [line.strip() for line in f if line.strip()]

# Number of splits
num_splits = 100
chunk_size = math.ceil(len(arxiv_ids) / num_splits)

# Split and save
for i in range(num_splits):
    start = i * chunk_size
    end = start + chunk_size
    chunk = arxiv_ids[start:end]
    
    filename = f"./csv/arxiv_id_list{i+1}.csv"
    with open(filename, "w") as f:
        json.dump(chunk, f, indent=2)
    
    print(f"Saved {len(chunk)} ids to {filename}")
