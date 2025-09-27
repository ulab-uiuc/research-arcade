import json
import argparse
import os

def split_json(input_path: str, output_prefix: str):
    # Load full JSON array
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of objects.")

    n = len(data)
    if n < 3:
        raise ValueError("Not enough objects to split into three parts.")

    # Compute split indices
    split1 = n // 3
    split2 = 2 * n // 3

    part1 = data[:split1]
    part2 = data[split1:split2]
    part3 = data[split2:]

    # Save each part into its own file
    for i, part in enumerate([part1, part2, part3], start=1):
        out_path = f"{output_prefix}_part{i}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(part, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(part)} objects to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Split a JSON list into three parts.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSON file.")
    parser.add_argument("--output_prefix", type=str, required=True, help="Prefix for output JSON files.")
    args = parser.parse_args()

    split_json(args.input_path, args.output_prefix)


if __name__ == "__main__":
    main()
