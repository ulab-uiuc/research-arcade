
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from tasks.paragraph_generation import _data_fetching

paragraph_key_ids = ["21829759", "21831811", "21854471", "56348", "2848716", "21846899"]

data_path = "./jsonl/paragraph_generation2.jsonl"

k_neighbour = 5

_data_fetching(paragraph_key_ids=paragraph_key_ids, data_path=data_path, k_neighbour=k_neighbour)

