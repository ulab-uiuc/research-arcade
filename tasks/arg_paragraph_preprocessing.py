import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tasks.paragraph_generation_local_vllm import _data_extraction_non_vlm_save


def main():
    parser = argparse.ArgumentParser(description="Preprocess JSONL for non-VLM models.")
    parser.add_argument(
        "--jsonl_path",
        type=str,
        required=True,
        help="Input JSONL file path."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Output preprocessed JSONL file path."
    )

    args = parser.parse_args()


    written = _data_extraction_non_vlm_save(
        jsonl_file_path=args.jsonl_path,
        file_save_path=args.save_path,
        start_from_prev_stop=True
    )

    print(f"The number of json objects written to the jsonl file is {written}")


if __name__ == "__main__":
    main()
