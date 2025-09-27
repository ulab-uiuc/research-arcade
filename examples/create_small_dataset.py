import json
import os


file_path = "./data/paragraph_generation/tasks/paragraph_generation_validation_embedded.json"
dest_path = "./data/paragraph_generation/tasks/paragraph_generation_validation_embedded_smoke.json"
k = 10
subset = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        k -= 1
        subset.append(json.loads(line))
        if k <= 0:
            break

with open(dest_path, "w") as f:
    for obj in subset:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

