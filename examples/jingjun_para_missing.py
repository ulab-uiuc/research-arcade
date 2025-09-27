import json

data_path1 = "./json/arxiv_2022.json"
data_path2 = "./json/arxiv_old_in2022.json"
dest_file = "./json/missing.json"

with open(data_path1, 'r') as f:
    arxiv_ids_existing = set(json.load(f))

with open(data_path2, 'r') as f:
    arxiv_ids = json.load(f)

# Choose arxiv ids that are in arxiv_ids but not in arxiv_ids_existing
arxiv_ids_missing = [id for id in arxiv_ids if id not in arxiv_ids_existing]

with open(dest_file, 'a') as f:
    json.dump(arxiv_ids_missing, f)