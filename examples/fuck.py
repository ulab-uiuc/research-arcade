import sys
import json
# import


training_data_path = f"./data/paragraph_generation/task/paragraph_generation_training_figure_table.json"



train_data = []
with open(training_data_path, 'r') as json_file:
    raw = json.load(json_file)
    # print(raw)
    for json_line in raw:
        train_data.append(json_line)

# print(train_data)
print(len(train_data))
sys.exit()
