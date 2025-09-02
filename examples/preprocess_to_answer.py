import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tasks.paragraph_generation_local_vllm import llm_generate_evaluate_from_preprocessed_jsonl


def main():
    jsonl_path = "./jsonl/paragraph_generation2_preprocessed2.jsonl"
    result_path = "./csv/csv_paragraph_generation2_result.csv"

    llm_generate_evaluate_from_preprocessed_jsonl(jsonl_path=jsonl_path, result_path=result_path, model_name="Qwen/Qwen3-8B")

if __name__ == "__main__": 
    main()