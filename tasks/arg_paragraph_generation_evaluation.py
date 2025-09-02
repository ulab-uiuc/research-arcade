import sys
import os
import argparse
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tasks.paragraph_generation_local_vllm import llm_generate_evaluate_from_preprocessed_jsonl


def main():
    parser = argparse.ArgumentParser(description="Generate and evaluate paragraphs from a preprocessed JSONL file.")
    parser.add_argument(
        "--jsonl_path",
        type=str,
        required=True,
        help="Path to the preprocessed JSONL file."
    )
    parser.add_argument(
        "--result_path",
        type=str,
        required=True,
        help="Path to save the evaluation results (CSV)."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Model name to use (default: Qwen/Qwen3-8B)."
    )

    args = parser.parse_args()

    llm_generate_evaluate_from_preprocessed_jsonl(
        jsonl_path=args.jsonl_path,
        result_path=args.result_path,
        model_name=args.model_name
    )


if __name__ == "__main__":
    main()
