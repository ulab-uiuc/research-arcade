import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

file_path = "./json/openreview_arxiv_2025.json"
split = 10

arxiv_ids = []

if os.path.exists(file_path):
    # Load the JSON list
    with open(file_path, 'r') as f:
        json_list = json.load(f)
        for json_obj in json_list:
            arxiv_url = json_obj.get("arxiv_id", None)
            if arxiv_url:
                arxiv_id = arxiv_url.split("/")[-1]
                arxiv_ids.append(arxiv_id)

    chunk_size = (len(arxiv_ids) + split - 1) // split  

    for i in range(split):
        chunk = arxiv_ids[i * chunk_size : (i + 1) * chunk_size]
        if not chunk:
            continue
        
        output_path = f"./json/arxiv_id_openreview_{i+1}.json"
        with open(output_path, "w") as out_f:
            json.dump(chunk, out_f, indent=2)

        print(f"Wrote {len(chunk)} IDs to {output_path}")
