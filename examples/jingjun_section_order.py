# First obtain the paper arxiv id list
import csv
import os
import json


paper_csv_path = "./data/openreview/papers.csv"
save_path = "./jingjun/paper_section_order.csv"

arxiv_ids = []
with open(paper_csv_path, 'r') as csv_file:
    reader = csv.reader(csv_file)
    _ = next(reader)  # Skip header
    for row in reader:
        arxiv_ids.append(row[1])

n_id = len(arxiv_ids)
n_exists = 0

section_id = 1
with open(save_path, 'w', newline='') as output_file:
    writer = csv.writer(output_file)
    # Write header
    writer.writerow(["section_id", "arxiv_id", "section_name", "section_order"])
    
    for arxiv_id in arxiv_ids:
        json_path = f"./download/output/{arxiv_id}.json"
        exists = os.path.exists(json_path)
        if exists:
            n_exists += 1
            with open(json_path, 'r') as json_file:
                json_object = json.load(json_file)
                section_order = 1
                contents = json_object["sections"]
                sections = list(contents.keys())
                for section in sections:
                    # The key is the section name
                    line = [section_id, arxiv_id, section, section_order]
                    writer.writerow(line)
                    section_id += 1
                    section_order += 1

print(f"Total papers: {n_id}")
print(f"Existing JSON files: {n_exists}")