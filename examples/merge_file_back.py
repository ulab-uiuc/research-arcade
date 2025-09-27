import json
import os

target_json_path = "./data/paragraph_generation/tasks/paragraph_generation_training_embedding.json"

with open(target_json_path, 'w', encoding="utf-8") as target_file:
    for i in range(1, 4):
        source_file_name = f"./data/paragraph_generation/tasks/paragraph_generation_training_embedding_part{i}.json"
        if not os.path.exists(source_file_name):
            print(f"Warning: {source_file_name} not found, skipping...")
            continue
        try:
            with open(source_file_name, 'r', encoding="utf-8") as source_file:
                for line in source_file:
                    line = line.strip()
                    if not line:
                        continue
                    # If file is JSON array instead of JSONL
                    if line.startswith("[") or line.startswith("]") or line.endswith(","):
                        # skip array brackets/commas
                        line = line.strip(",[] ")
                        if not line:
                            continue
                    if line:
                        obj = json.loads(line)
                        target_file.write(json.dumps(obj, ensure_ascii=False) + "\n")
            print(f"Appended {source_file_name}")
        except Exception as e:
            print(f"Error processing {source_file_name}: {e}")
