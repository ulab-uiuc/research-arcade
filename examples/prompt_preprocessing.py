import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from tasks.paragraph_generation_local_vllm import _data_extraction_non_vlm_save


written = _data_extraction_non_vlm_save(jsonl_file_path = "./jsonl/paragraph_generation2.jsonl", file_save_path = "./jsonl/paragraph_generation2_preprocessed2.jsonl")

print(f"The number of json objects written to the jsonl file is {written}")
