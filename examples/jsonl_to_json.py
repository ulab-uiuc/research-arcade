import json
import argparse

def jsonl_to_json(input_file, output_file):
    """
    Convert JSONL file to JSON array format.
    
    Args:
        input_file (str): Path to input JSONL file
        output_file (str): Path to output JSON file
    """
    json_array = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        json_obj = json.loads(line)
                        json_array.append(json_obj)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON on line {line_num}: {e}")
                        continue
        
        # Write the array to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_array, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully converted {len(json_array)} JSON objects from {input_file} to {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
    except Exception as e:
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert JSONL file to JSON array format")
    parser.add_argument(
        "--input", 
        "-i",
        type=str, 
        required=True,
        help="Input JSONL file path"
    )
    parser.add_argument(
        "--output", 
        "-o",
        type=str, 
        required=True,
        help="Output JSON file path"
    )
    
    args = parser.parse_args()
    
    jsonl_to_json(args.input, args.output)

if __name__ == "__main__":
    main()