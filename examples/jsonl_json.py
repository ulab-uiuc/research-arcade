import json

data_path = "./data/paragraph_generation/tasks/paragraph_generation_testing_embedded.json"
with open(data_path, "r", encoding="utf-8") as f:
    for line in f: 
        data = json.loads(line)
        print(data)

